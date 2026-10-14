// podlab - local arena for Mad Pod Racing bots.
//
// The bots live in bots/a.rs (baseline) and bots/b.rs (candidate slot). They are compiled into this
// program unchanged (include!) and started as child processes of this same executable ("--bot a|b"),
// so they talk to the arena over stdin/stdout exactly like on CodinGame.
//
// The arena plays every candidate from experiments.txt against the baseline, both sides, on identical
// random maps, in parallel, with a fixed number of simulations per turn (CSB_SIMS) so that results do not
// depend on how busy your computer is. It stops after --minutes and writes results.txt.
#![allow(dead_code)]

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

mod bot_a {
    include!("../bots/a.rs");
    pub fn entry() {
        main();
    }
}

mod bot_b {
    include!("../bots/b.rs");
    pub fn entry() {
        main();
    }
}

const POD_RSQ: f64 = 800.0 * 800.0;
const CP_RSQ: f64 = 600.0 * 600.0;
const SHIELD_TIMER: i32 = 4;
const MAX_TURNS: usize = 800;
const LAPS: i32 = 3;
const TURN_TIMEOUT: Duration = Duration::from_millis(5000);

// ------------------------------------------------------------------------------------------------
// Game physics (same rules as the simulation inside the bot)
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct Pod {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    angle: i32, // -1 = not yet set (first turn)
    next: usize,
    passed: i32,
    shield: i32,
    boosted: bool,
    idle: i32, // turns left before this pod is considered stuck
}

#[derive(Clone, Copy)]
enum Th {
    Thrust(i32),
    Shield,
    Boost,
}

#[derive(Clone, Copy)]
struct Cmd {
    tx: f64,
    ty: f64,
    th: Th,
}

#[derive(Clone, Copy, PartialEq)]
enum Outcome {
    P0,
    P1,
    Draw,
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        let mut r = Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1);
        for _ in 0..8 {
            r.next();
        }
        r
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }
    fn int(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }
}

fn make_map(seed: u64) -> Vec<(f64, f64)> {
    let mut rng = Rng::new(seed);
    let n = rng.int(3, 8) as usize;
    let mut cps: Vec<(f64, f64)> = Vec::new();
    while cps.len() < n {
        let mut chosen = (0.0, 0.0);
        for _try in 0..200 {
            let x = rng.range(1000.0, 15000.0).round();
            let y = rng.range(1000.0, 8000.0).round();
            chosen = (x, y);
            let ok = cps.iter().all(|c| {
                let dx = c.0 - x;
                let dy = c.1 - y;
                dx * dx + dy * dy >= 2000.0 * 2000.0
            });
            if ok {
                break;
            }
        }
        cps.push(chosen);
    }
    cps
}

fn start_pods(cps: &[(f64, f64)]) -> [Pod; 4] {
    let c0 = cps[0];
    let c1 = cps[1];
    let dx = c1.0 - c0.0;
    let dy = c1.1 - c0.1;
    let len = (dx * dx + dy * dy).sqrt().max(1.0);
    let px = -dy / len;
    let py = dx / len;
    let offs = [-1500.0f64, -500.0, 500.0, 1500.0];
    // line-up across the track: P0.pod0, P1.pod0, P0.pod1, P1.pod1 -> pod indices 0, 2, 1, 3
    let idx = [0usize, 2, 1, 3];
    let mut pods = [Pod::default(); 4];
    for k in 0..4 {
        pods[idx[k]] = Pod {
            x: (c0.0 + px * offs[k]).round(),
            y: (c0.1 + py * offs[k]).round(),
            angle: -1,
            next: 1,
            idle: 100,
            ..Pod::default()
        };
    }
    pods
}

fn collide_time(a: &Pod, b: &Pod) -> f64 {
    let px = b.x - a.x;
    let py = b.y - a.y;
    let pl2 = px * px + py * py;
    if pl2 <= POD_RSQ {
        return 0.0;
    }
    let vx = b.vx - a.vx;
    let vy = b.vy - a.vy;
    let dot = px * vx + py * vy;
    if dot > 0.0 {
        return 10.0;
    }
    let vl2 = vx * vx + vy * vy;
    if vl2 < 1e-12 {
        return 10.0;
    }
    let disc = dot * dot - vl2 * (pl2 - POD_RSQ);
    if disc < 0.0 {
        return 10.0;
    }
    (-dot - disc.sqrt()) / vl2
}

fn bounce(a: &mut Pod, b: &mut Pod) {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let mut dd = (dx * dx + dy * dy).sqrt();
    let (nx, ny) = if dd > 1e-9 {
        (dx / dd, dy / dd)
    } else {
        dd = 0.0;
        (1.0, 0.0)
    };
    let rvx = a.vx - b.vx;
    let rvy = a.vy - b.vy;
    let m1 = if a.shield == SHIELD_TIMER { 0.1 } else { 1.0 };
    let m2 = if b.shield == SHIELD_TIMER { 0.1 } else { 1.0 };
    let mut force = (nx * rvx + ny * rvy) / (m1 + m2);
    if force < 120.0 {
        force += 120.0;
    } else {
        force += force;
    }
    let ix = nx * (-force);
    let iy = ny * (-force);
    a.vx += ix * m1;
    a.vy += iy * m1;
    b.vx -= ix * m2;
    b.vy -= iy * m2;
    if dd <= 800.0 {
        let push = (800.0 - dd) / 2.0 + 0.00001;
        a.x -= nx * push;
        a.y -= ny * push;
        b.x += nx * push;
        b.y += ny * push;
    }
}

fn cp_collide(p1: (f64, f64), p2: (f64, f64), cp: (f64, f64)) -> bool {
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let pd2 = dx * dx + dy * dy;
    let mut ppx = p1.0;
    let mut ppy = p1.1;
    if pd2 != 0.0 {
        let u = ((cp.0 - p1.0) * dx + (cp.1 - p1.1) * dy) / pd2;
        if u > 1.0 {
            ppx = p2.0;
            ppy = p2.1;
        } else if u > 0.0 {
            ppx = p1.0 + u * dx;
            ppy = p1.1 + u * dy;
        }
    }
    let ex = ppx - cp.0;
    let ey = ppy - cp.1;
    ex * ex + ey * ey < CP_RSQ
}

fn check_cp(p: &mut Pod, from: (f64, f64), cps: &[(f64, f64)], total: i32) {
    if p.passed >= total {
        return;
    }
    let cp = cps[p.next % cps.len()];
    if cp_collide(from, (p.x, p.y), cp) {
        p.passed += 1;
        p.next = (p.next + 1) % cps.len();
    }
}

/// Rotate toward the target (max 18 degrees, unlimited on the very first turn), then apply thrust.
fn rotate_accel(p: &mut Pod, c: &Cmd) {
    let dx = c.tx - p.x;
    let dy = c.ty - p.y;
    let target = dy.atan2(dx).to_degrees();
    if p.angle < 0 {
        p.angle = (target.round() as i32).rem_euclid(360);
    } else {
        let mut da = target - p.angle as f64;
        da = (da + 540.0).rem_euclid(360.0) - 180.0;
        let da = da.clamp(-18.0, 18.0);
        p.angle = ((p.angle as f64 + da).round() as i32).rem_euclid(360);
    }

    let mut thrust = 0.0;
    if p.shield > 0 {
        // cooldown: engine off, SHIELD / BOOST commands are ignored
    } else {
        match c.th {
            Th::Shield => {
                p.shield = SHIELD_TIMER;
            }
            Th::Boost => {
                if !p.boosted {
                    p.boosted = true;
                    thrust = 650.0;
                } else {
                    thrust = 200.0;
                }
            }
            Th::Thrust(t) => {
                thrust = t.clamp(0, 200) as f64;
            }
        }
    }
    let a = (p.angle as f64).to_radians();
    p.vx += a.cos() * thrust;
    p.vy += a.sin() * thrust;
}

fn simulate(pods: &mut [Pod; 4], cps: &[(f64, f64)], total: i32) {
    let mut starts = [(0.0f64, 0.0f64); 4];
    for i in 0..4 {
        starts[i] = (pods[i].x, pods[i].y);
    }
    let mut t = 1.0f64;
    let mut iter = 0;

    while t > 0.0 {
        let mut first = t;
        let mut ci = 0usize;
        let mut cj = 0usize;

        if iter < 8 {
            for i in (1..4).rev() {
                for j in (0..i).rev() {
                    let tx = collide_time(&pods[i], &pods[j]);
                    if tx <= first {
                        first = tx;
                        ci = i;
                        cj = j;
                    }
                }
            }
        }
        iter += 1;

        for p in pods.iter_mut() {
            p.x += p.vx * first;
            p.y += p.vy * first;
        }
        t -= first;

        if ci != cj {
            let (l, r) = pods.split_at_mut(ci);
            bounce(&mut r[0], &mut l[cj]);
        }

        if t > 0.0 {
            for i in 0..4 {
                check_cp(&mut pods[i], starts[i], cps, total);
                starts[i] = (pods[i].x, pods[i].y);
            }
        }
    }

    for i in 0..4 {
        let p = &mut pods[i];
        p.vx = (p.vx * 0.85).trunc();
        p.vy = (p.vy * 0.85).trunc();
        p.x = (p.x + 0.5).floor();
        p.y = (p.y + 0.5).floor();
        if p.shield > 0 {
            p.shield -= 1;
        }
        check_cp(p, starts[i], cps, total);
    }
}

// ------------------------------------------------------------------------------------------------
// Bot processes
// ------------------------------------------------------------------------------------------------

#[derive(Clone)]
struct Spec {
    name: String,
    which: String, // "a" or "b"
    env: Vec<(String, String)>,
    sims_abs: Option<u32>,
    sims_mult: f64,
}

impl Spec {
    fn base() -> Spec {
        Spec {
            name: "baseline".to_string(),
            which: "a".to_string(),
            env: Vec::new(),
            sims_abs: None,
            sims_mult: 1.0,
        }
    }

    fn eff_sims(&self, global: u32) -> u32 {
        match self.sims_abs {
            Some(s) => s,
            None => ((global as f64) * self.sims_mult).round().max(1.0) as u32,
        }
    }
}

struct Proc {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<String>,
}

impl Proc {
    fn spawn(exe: &Path, spec: &Spec, sims: u32) -> Result<Proc, String> {
        let mut cmd = Command::new(exe);
        cmd.arg("--bot").arg(&spec.which);
        cmd.env("CSB_DEBUG", "0");
        cmd.env("CSB_SIMS", sims.to_string());
        for (k, v) in &spec.env {
            cmd.env(k, v);
        }
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("cannot start the bot process: {}", e))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "bot process has no stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "bot process has no stdout".to_string())?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(l) => {
                        if tx.send(l).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        Ok(Proc { child, stdin, rx })
    }

    fn send(&mut self, s: &str) -> std::io::Result<()> {
        self.stdin.write_all(s.as_bytes())?;
        self.stdin.flush()
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn parse_cmd(line: &str) -> Option<Cmd> {
    let mut it = line.split_whitespace();
    let tx: f64 = it.next()?.parse().ok()?;
    let ty: f64 = it.next()?.parse().ok()?;
    let t = it.next()?;
    let th = if t == "SHIELD" {
        Th::Shield
    } else if t == "BOOST" {
        Th::Boost
    } else {
        Th::Thrust(t.parse::<i32>().ok()?)
    };
    Some(Cmd { tx, ty, th })
}

fn input_for(pods: &[Pod; 4], me: usize) -> String {
    let order = if me == 0 { [0usize, 1, 2, 3] } else { [2usize, 3, 0, 1] };
    let mut s = String::new();
    for &i in order.iter() {
        let p = &pods[i];
        s.push_str(&format!(
            "{} {} {} {} {} {}\n",
            p.x as i64, p.y as i64, p.vx as i64, p.vy as i64, p.angle, p.next
        ));
    }
    s
}

fn forfeit(crashed: [bool; 2], turns: usize) -> (Outcome, [bool; 2], usize) {
    let oc = if crashed[0] && crashed[1] {
        Outcome::Draw
    } else if crashed[0] {
        Outcome::P1
    } else {
        Outcome::P0
    };
    (oc, crashed, turns)
}

/// Plays one game. Returns (outcome, which bots crashed or timed out, turns played).
fn play(cps: &[(f64, f64)], specs: [&Spec; 2], sims: u32, exe: &Path) -> (Outcome, [bool; 2], usize) {
    let n = cps.len();
    let total = LAPS * n as i32;

    let p0 = match Proc::spawn(exe, specs[0], specs[0].eff_sims(sims)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return (Outcome::P1, [true, false], 0);
        }
    };
    let p1 = match Proc::spawn(exe, specs[1], specs[1].eff_sims(sims)) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return (Outcome::P0, [false, true], 0);
        }
    };
    let mut procs = [p0, p1];

    let mut init = format!("{}\n{}\n", LAPS, n);
    for c in cps.iter() {
        init.push_str(&format!("{} {}\n", c.0 as i64, c.1 as i64));
    }
    let mut crashed = [false, false];
    for k in 0..2 {
        if procs[k].send(&init).is_err() {
            crashed[k] = true;
        }
    }
    if crashed[0] || crashed[1] {
        return forfeit(crashed, 0);
    }

    let mut pods = start_pods(cps);
    let mut turn = 0usize;
    while turn < MAX_TURNS {
        turn += 1;

        for k in 0..2 {
            let inp = input_for(&pods, k);
            if procs[k].send(&inp).is_err() {
                crashed[k] = true;
            }
        }
        if crashed[0] || crashed[1] {
            return forfeit(crashed, turn);
        }

        let mut cm = [Cmd { tx: 0.0, ty: 0.0, th: Th::Thrust(0) }; 4];
        for k in 0..2 {
            for j in 0..2 {
                match procs[k].rx.recv_timeout(TURN_TIMEOUT) {
                    Ok(line) => match parse_cmd(&line) {
                        Some(c) => cm[2 * k + j] = c,
                        None => crashed[k] = true,
                    },
                    Err(_) => crashed[k] = true,
                }
                if crashed[k] {
                    break;
                }
            }
        }
        if crashed[0] || crashed[1] {
            return forfeit(crashed, turn);
        }

        let before = [pods[0].passed, pods[1].passed, pods[2].passed, pods[3].passed];
        for i in 0..4 {
            rotate_accel(&mut pods[i], &cm[i]);
        }
        simulate(&mut pods, cps, total);
        for i in 0..4 {
            if pods[i].passed > before[i] {
                pods[i].idle = 100;
            } else {
                pods[i].idle -= 1;
            }
        }

        let f0 = pods[0].passed >= total || pods[1].passed >= total;
        let f1 = pods[2].passed >= total || pods[3].passed >= total;
        if f0 && f1 {
            return (Outcome::Draw, crashed, turn);
        }
        if f0 {
            return (Outcome::P0, crashed, turn);
        }
        if f1 {
            return (Outcome::P1, crashed, turn);
        }

        let dead0 = pods[0].idle <= 0 && pods[1].idle <= 0;
        let dead1 = pods[2].idle <= 0 && pods[3].idle <= 0;
        if dead0 && dead1 {
            return (Outcome::Draw, crashed, turn);
        }
        if dead0 {
            return (Outcome::P1, crashed, turn);
        }
        if dead1 {
            return (Outcome::P0, crashed, turn);
        }
    }

    let s0 = pods[0].passed.max(pods[1].passed);
    let s1 = pods[2].passed.max(pods[3].passed);
    let oc = if s0 > s1 {
        Outcome::P0
    } else if s1 > s0 {
        Outcome::P1
    } else {
        Outcome::Draw
    };
    (oc, crashed, turn)
}

// ------------------------------------------------------------------------------------------------
// Experiments, scheduling, reporting
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Default)]
struct Tally {
    w: u32,
    d: u32,
    l: u32,
    crash_c: u32,
    crash_b: u32,
    turns: u64,
}

struct Shared {
    cands: Vec<Spec>,
    base: Spec,
    sims: u32,
    exe: PathBuf,
    seed: u64,
    deadline: Instant,
    max_jobs: usize,
    next_job: AtomicUsize,
    done: AtomicUsize,
    tallies: Mutex<Vec<Tally>>,
}

fn load_experiments(path: &str) -> Vec<Spec> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {}: {}", path, e);
            std::process::exit(1);
        }
    };
    let mut out: Vec<Spec> = Vec::new();
    for (ln, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, rest) = match line.split_once(':') {
            Some(x) => x,
            None => {
                eprintln!("{}:{}: expected 'name: SETTING=VALUE ...'", path, ln + 1);
                std::process::exit(1);
            }
        };
        let mut spec = Spec::base();
        spec.name = name.trim().to_string();
        for tok in rest.split_whitespace() {
            let (k, v) = match tok.split_once('=') {
                Some(x) => x,
                None => {
                    eprintln!("{}:{}: '{}' is not SETTING=VALUE", path, ln + 1, tok);
                    std::process::exit(1);
                }
            };
            match k {
                "bot" => spec.which = v.to_string(),
                "sims" => spec.sims_abs = v.parse().ok(),
                "simsx" => spec.sims_mult = v.parse().unwrap_or(1.0),
                _ => spec.env.push((k.to_string(), v.to_string())),
            }
        }
        out.push(spec);
    }
    out
}

fn render(names: &[String], t: &[Tally], header: &str) -> String {
    let mut rows: Vec<(f64, String)> = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let tt = &t[i];
        let n = (tt.w + tt.d + tt.l) as f64;
        if n < 1.0 {
            continue;
        }
        let w = tt.w as f64;
        let d = tt.d as f64;
        let p = (w + 0.5 * d) / n;
        // Wilson score interval: behaves sensibly for small n and for scores near 0% or 100%
        let z = 1.96f64;
        let denom = 1.0 + z * z / n;
        let centre = (p + z * z / (2.0 * n)) / denom;
        let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denom;
        let lo = centre - half;
        let hi = centre + half;
        let pc = p.max(0.5 / n).min(1.0 - 0.5 / n);
        let elo = 400.0 * (pc / (1.0 - pc)).log10();
        let verdict = if n < 40.0 {
            "too few games"
        } else if lo > 0.5 {
            "BETTER than baseline"
        } else if hi < 0.5 {
            "WORSE than baseline"
        } else {
            "unclear"
        };
        let mut line = format!(
            "{:<12} {:>5} games  W-D-L {:>4}-{:>3}-{:>4}  score {:>5.1}%  95% CI [{:>5.1}, {:>5.1}]  Elo {:>+5.0}  {}",
            name,
            tt.w + tt.d + tt.l,
            tt.w,
            tt.d,
            tt.l,
            100.0 * p,
            100.0 * lo,
            100.0 * hi,
            elo,
            verdict
        );
        if tt.crash_c + tt.crash_b > 0 {
            line.push_str(&format!(
                "  [bot crashes/timeouts: candidate {}, baseline {}]",
                tt.crash_c, tt.crash_b
            ));
        }
        rows.push((p, line));
    }
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = String::new();
    out.push_str(header);
    out.push('\n');
    out.push_str("score = share of points the candidate won against the baseline (win 1, draw 0.5); 50% = equal\n\n");
    for (_, line) in rows {
        out.push_str(&line);
        out.push('\n');
    }
    out
}

fn worker(shared: Arc<Shared>) {
    let m_count = shared.cands.len();
    let per_round = 2 * m_count;
    loop {
        if Instant::now() >= shared.deadline {
            break;
        }
        let j = shared.next_job.fetch_add(1, Ordering::SeqCst);
        if j >= shared.max_jobs {
            break;
        }
        let round = j / per_round;
        let within = j % per_round;
        let m = within / 2;
        let slot = within % 2; // candidate plays as P0 in slot 0 and as P1 in slot 1
        let cps = make_map(shared.seed + round as u64);
        let specs: [&Spec; 2] = if slot == 0 {
            [&shared.cands[m], &shared.base]
        } else {
            [&shared.base, &shared.cands[m]]
        };
        let (oc, crash, turns) = play(&cps, specs, shared.sims, &shared.exe);

        let mut t = shared.tallies.lock().unwrap();
        let tt = &mut t[m];
        match (oc, slot) {
            (Outcome::P0, 0) | (Outcome::P1, 1) => tt.w += 1,
            (Outcome::Draw, _) => tt.d += 1,
            _ => tt.l += 1,
        }
        if crash[slot] {
            tt.crash_c += 1;
        }
        if crash[1 - slot] {
            tt.crash_b += 1;
        }
        tt.turns += turns as u64;
        drop(t);
        shared.done.fetch_add(1, Ordering::SeqCst);
    }
}

fn take(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    match args.get(*i) {
        Some(v) => v.clone(),
        None => {
            eprintln!("{} needs a value", flag);
            std::process::exit(1);
        }
    }
}

fn print_help() {
    println!("podlab - local Mad Pod Racing arena");
    println!();
    println!("  --minutes N       how long to run (default 20)");
    println!("  --sims N          simulations per turn for every bot (default 4000; the real bot gets ~30000+)");
    println!("  --workers N       games played in parallel (default: half of your CPU threads)");
    println!("  --games N         stop a candidate after N games (default: run until the time is up)");
    println!("  --only a,b,c      only run these candidates from the experiments file");
    println!("  --experiments F   experiments file (default experiments.txt)");
    println!("  --out F           results file (default results.txt)");
    println!("  --seed N          map seed (default 12345)");
}

fn write_results(shared: &Shared, header: &str, out_path: &str) -> String {
    let names: Vec<String> = shared.cands.iter().map(|c| c.name.clone()).collect();
    let snapshot: Vec<Tally> = shared.tallies.lock().unwrap().clone();
    let text = render(&names, &snapshot, header);
    let _ = std::fs::write(out_path, &text);
    text
}

fn arena_main(args: Vec<String>) {
    let mut minutes = 20.0f64;
    let mut sims = 4000u32;
    let mut workers = 0usize;
    let mut games_cap = 1_000_000usize;
    let mut only: Option<Vec<String>> = None;
    let mut exp_path = "experiments.txt".to_string();
    let mut out_path = "results.txt".to_string();
    let mut seed = 12345u64;

    let mut i = 1;
    while i < args.len() {
        let flag = args[i].clone();
        match flag.as_str() {
            "--minutes" => minutes = take(&args, &mut i, &flag).parse().unwrap_or(20.0),
            "--sims" => sims = take(&args, &mut i, &flag).parse().unwrap_or(4000),
            "--workers" => workers = take(&args, &mut i, &flag).parse().unwrap_or(0),
            "--games" => games_cap = take(&args, &mut i, &flag).parse().unwrap_or(1_000_000),
            "--only" => {
                let v = take(&args, &mut i, &flag);
                only = Some(v.split(',').map(|s| s.trim().to_string()).collect());
            }
            "--experiments" => exp_path = take(&args, &mut i, &flag),
            "--out" => out_path = take(&args, &mut i, &flag),
            "--seed" => seed = take(&args, &mut i, &flag).parse().unwrap_or(12345),
            "-h" | "--help" => {
                print_help();
                return;
            }
            other => {
                eprintln!("unknown option {} (try --help)", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let mut cands = load_experiments(&exp_path);
    if let Some(names) = &only {
        cands.retain(|c| names.iter().any(|n| *n == c.name));
    }
    if cands.is_empty() {
        eprintln!("no candidates to run - check {} and --only", exp_path);
        std::process::exit(1);
    }
    let cores = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    if workers == 0 {
        workers = (cores / 2).max(1);
    }
    let exe = std::env::current_exe().expect("cannot locate this program's own executable");
    let base = Spec::base();

    println!(
        "podlab: {} candidates, {} workers ({} CPU threads detected), {} sims/turn, {:.0} minutes",
        cands.len(),
        workers,
        cores,
        sims,
        minutes
    );
    println!("self-test: one game, baseline vs baseline ...");
    let t0 = Instant::now();
    let test_map = make_map(seed);
    let (oc, crash, turns) = play(&test_map, [&base, &base], sims, &exe);
    if crash[0] || crash[1] {
        eprintln!("self-test FAILED: a bot process crashed, timed out or printed something unexpected.");
        eprintln!("Send me this message and the output above it.");
        std::process::exit(1);
    }
    let secs = t0.elapsed().as_secs_f64();
    let who = match oc {
        Outcome::P0 => "player 0 won",
        Outcome::P1 => "player 1 won",
        Outcome::Draw => "draw",
    };
    println!(
        "self-test OK: {} turns in {:.1}s ({}). Expect roughly {:.1} games/s in total.",
        turns,
        secs,
        who,
        workers as f64 / secs.max(0.01)
    );

    let m = cands.len();
    let rounds_cap = (games_cap / 2).max(1);
    let start = Instant::now();
    let shared = Arc::new(Shared {
        cands,
        base,
        sims,
        exe,
        seed,
        deadline: start + Duration::from_secs_f64(minutes * 60.0),
        max_jobs: rounds_cap.saturating_mul(2).saturating_mul(m),
        next_job: AtomicUsize::new(0),
        done: AtomicUsize::new(0),
        tallies: Mutex::new(vec![Tally::default(); m]),
    });
    run_experiments(shared, workers, minutes, sims, &out_path, start);
}

fn run_experiments(
    shared: Arc<Shared>,
    workers: usize,
    minutes: f64,
    sims: u32,
    out_path: &str,
    start: Instant,
) {
    let mut handles: Vec<thread::JoinHandle<()>> = Vec::new();
    for _ in 0..workers {
        let sh = Arc::clone(&shared);
        handles.push(thread::spawn(move || worker(sh)));
    }

    let mut last_print = Instant::now();
    let mut last_write = Instant::now();
    loop {
        thread::sleep(Duration::from_millis(500));
        let finished = handles.iter().all(|h| h.is_finished());
        let el = start.elapsed();
        let done = shared.done.load(Ordering::SeqCst);
        let header = format!(
            "podlab results | baseline = bots/a.rs defaults | {} sims/turn | {} workers | {:.0} min budget | {:.0}s elapsed | {} games",
            sims,
            workers,
            minutes,
            el.as_secs_f64(),
            done
        );
        if last_print.elapsed() >= Duration::from_secs(20) {
            last_print = Instant::now();
            println!(
                "[{:02}:{:02}] {} games played ({:.1} games/s)",
                el.as_secs() / 60,
                el.as_secs() % 60,
                done,
                done as f64 / el.as_secs_f64().max(1.0)
            );
        }
        if last_write.elapsed() >= Duration::from_secs(60) {
            last_write = Instant::now();
            let text = write_results(&shared, &header, out_path);
            println!("\n{}", text);
        }
        if finished {
            break;
        }
    }
    for h in handles {
        if h.join().is_err() {
            eprintln!("warning: a worker thread crashed");
        }
    }

    let done = shared.done.load(Ordering::SeqCst);
    let header = format!(
        "podlab results | baseline = bots/a.rs defaults | {} sims/turn | {} workers | {:.0} min budget | {:.0}s elapsed | {} games",
        sims,
        workers,
        minutes,
        start.elapsed().as_secs_f64(),
        done
    );
    let text = write_results(&shared, &header, out_path);
    println!("\n==================== FINAL ====================\n{}", text);
    println!("Saved to {}. Send me that file (or paste it).", out_path);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 3 && args[1] == "--bot" {
        match args[2].as_str() {
            "a" => bot_a::entry(),
            "b" => bot_b::entry(),
            other => eprintln!("unknown bot '{}'", other),
        }
        return;
    }
    arena_main(args);
}
