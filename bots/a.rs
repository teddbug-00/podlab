// Single-population GA for Mad Pod Racing  (v7 = v4_safe with the dynamic role weights switched OFF by default)
//
// v7: in self-play (podlab, 2399 games, 4000 sims/turn) turning the dynamic weights off beat v4_safe 60.6%
// (95% CI 58.6-62.5, ~+75 Elo). The blocker now always uses the fixed CSB_WBLOCK / CSB_WSECOND weights.
//
// (v4 + timing stats + safer time budget)
//
// TIME BUDGET: per-turn think time lowered from 73 ms to 50 ms (CSB_TIME). Replays showed turns of 88-90 ms
// (limit 75 ms) -> timeout losses, while the bot gets ~50k simulations per turn, far more than it needs.
// Everything below is v4 plus the stats line:
//
// This is v4 with ONE difference: a compact stats line is printed to stderr every turn (CSB_DEBUG now
// defaults to 1). Game behaviour is identical to v4. Read the last line of your bot's console output in a
// replay:  "turns N avg A min M maxt T slow S"  = turns since turn 2, average / minimum simulations per turn,
// slowest turn in ms, and how many turns took >= 65 ms (the limit is 75 ms). CSB_DEBUG=0 silences it,
// CSB_DEBUG=2 additionally prints the long v4 line (roles, weights, opponent plan, boost state).
//
// Changes vs v3:
//   BOOST gating. The GA used to fire BOOST on the first turn it improved the leaf score (= almost
//   immediately, usually wasting it on a short leg). Now a BOOST in a plan only counts if boost_ok()
//   holds in the simulated state at that ply, otherwise it is simulated as plain full thrust. The same
//   check guards the printed command, so the simulation and the real output always agree.
//   boost_ok() = pod would be well aligned with its checkpoint (after this turn's rotation) AND one of
//     - the leg is (nearly) the longest leg of the track and the pod is still far from the checkpoint
//     - last lap and far from the checkpoint ("use it or lose it")
//     - heading for the very last checkpoint
//
// Changes in v3 (vs v2):
//   1. Seeding fix: the blocker "aim at opponent racer" seed now goes into the WORST 8
//      individuals instead of the best 8 (which used to destroy the inherited elite plans),
//      tracks the heading while it rotates, and the opponent racer is chosen by potential().
//   2. Dynamic role weights: w_block / w_second depend on who is winning the race.
//        winning -> blocker cares less about blocking and more about racing itself
//        losing  -> blocker goes harder after the opponent leader
//   3. Adaptive opponent plan (co-evolution lite): at the start of each turn, a short
//      opponent plan (first CSB_OPLEN plies, 2 pods) is hill-climbed from the opponent's
//      side against my previous best plan (shifted one turn). The GA then runs against that
//      plan; plies after the plan use the scripted opp_move() as before. The plan can contain
//      SHIELD. It is skipped (scripted plan only) when no pod of mine is near an opponent.
//
// Env vars for local tuning:
//   existing: CSB_DEPTH CSB_POP CSB_WVEL CSB_WBLOCK CSB_WSECOND CSB_SIMS CSB_TIME
//             CSB_WFACE CSB_WVELFAR CSB_VELRAMP CSB_CHASE CSB_CHASEA CSB_DEBUG
//   new:      CSB_LEAD     distance lead (opp_rem - my_rem) that counts as "fully winning/losing"  (4000)
//             CSB_LSEC     when winning: w_second *= 1 + LSEC * k                                    (0 = off)
//             CSB_LBLK     when winning: w_block  *= 1 - LBLK * k                                    (0 = off)
//             CSB_BBLK     when losing:  w_block  *= 1 + BBLK * k                                    (0 = off)
//             CSB_OPLEN    opponent plan length in plies, 0 = feature off                            (3)
//             CSB_OPSHARE  fraction of the turn's time spent on the opponent hill-climb              (0.22)
//             CSB_OPDIST   skip the hill-climb if every (my pod, opp pod) pair is farther than this  (4000)
//
//             CSB_BOOSTGATE  1 = gate BOOST as described above, 0 = old behaviour                    (1)
//             CSB_BOOSTLEG   a leg counts as "long" if it is >= this fraction of the longest leg     (0.9)
//             CSB_BOOSTFAR   on a long leg, min distance to the checkpoint to boost                 (4000)
//             CSB_BOOSTLAST  last lap: min distance to the checkpoint to boost                      (3000)
//             CSB_BOOSTCOS   min cos(heading, direction to checkpoint) to boost                      (0.9)
//
// Switch features off for A/B tests:
//   boost gating off:     CSB_BOOSTGATE=0
//   dynamic weights ON:   CSB_LSEC=3 CSB_LBLK=0.6 CSB_BBLK=1   (they were the v3/v4 defaults)
//   opponent plan off:    CSB_OPLEN=0
use std::cmp::Ordering;
use std::f64::consts::PI;
use std::io::{self, BufRead};
use std::time::Instant;

const POD_RSQ: f64 = 800.0 * 800.0;
const CP_RSQ: f64 = 600.0 * 600.0;
const MIN_IMPULSE: f64 = 120.0;
const EPS: f64 = 0.00001;
const MAX_THRUST: i32 = 200;
const SHIELD_TIMER: i32 = 4;
const T_SHIELD: i32 = -1;
const T_BOOST: i32 = 650;
const MAX_DEPTH: usize = 12;
const MAX_OPP_LEN: usize = 6;
const RAD_TO_DEG: f64 = 180.0 / PI;
const COS_65: f64 = 0.422_618_261_740_699_4;
const ROLE_SWAP_HYSTERESIS: i32 = 3;
const CLOCK_CHECK_MASK: u32 = 15;

#[inline(always)]
fn floor_f(x: f64) -> f64 {
    let t = x as i64 as f64;
    if t > x { t - 1.0 } else { t }
}
#[inline(always)]
fn round_ref(x: f64) -> f64 { floor_f(x + 0.5) }
#[inline(always)]
fn trunc_f(x: f64) -> f64 { x as i64 as f64 }

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Vec2 {
    x: f64,
    y: f64,
}

impl Vec2 {
    #[inline(always)] fn new(x: f64, y: f64) -> Self { Self { x, y } }
    #[inline(always)] fn dot(self, o: Vec2) -> f64 { self.x * o.x + self.y * o.y }
    #[inline(always)] fn mag_sq(self) -> f64 { self.x * self.x + self.y * self.y }
    #[inline(always)] fn mag(self) -> f64 { self.mag_sq().sqrt() }
    #[inline(always)] fn dist(self, o: Vec2) -> f64 { (self - o).mag() }
    #[inline(always)] fn normalize(self) -> Vec2 {
        let m = self.mag();
        if m < 1e-9 { Self::new(0.0, 0.0) } else { Self::new(self.x / m, self.y / m) }
    }
}

impl std::ops::Add for Vec2 {
    type Output = Vec2;
    #[inline(always)]
    fn add(self, r: Vec2) -> Vec2 { Vec2::new(self.x + r.x, self.y + r.y) }
}
impl std::ops::Sub for Vec2 {
    type Output = Vec2;
    #[inline(always)]
    fn sub(self, r: Vec2) -> Vec2 { Vec2::new(self.x - r.x, self.y - r.y) }
}
impl std::ops::Mul<f64> for Vec2 {
    type Output = Vec2;
    #[inline(always)]
    fn mul(self, r: f64) -> Vec2 { Vec2::new(self.x * r, self.y * r) }
}

#[derive(Clone, Copy, Debug, Default)]
struct Move {
    angle: i32,
    thrust: i32,
}

/// Plan for the two opponent pods (index 0 = pod 2, index 1 = pod 3), first MAX_OPP_LEN plies.
type OppPlan = [[Move; MAX_OPP_LEN]; 2];

#[derive(Clone, Copy, Debug, Default)]
struct Pod {
    pos: Vec2,
    vel: Vec2,
    angle: i32,
    next: usize,
    passed: i32,
    shield: i32,
    boosted: bool,
}

struct TrigLut {
    sin: [f64; 360],
    cos: [f64; 360],
}

impl TrigLut {
    fn new() -> Self {
        let mut sin = [0.0; 360];
        let mut cos = [0.0; 360];
        for i in 0..360 {
            let r = i as f64 * PI / 180.0;
            sin[i] = r.sin();
            cos[i] = r.cos();
        }
        Self { sin, cos }
    }
}

impl Pod {
    #[inline(always)]
    fn apply_move(&mut self, m: Move, trig: &TrigLut) {
        self.angle += m.angle;
        if self.angle >= 360 {
            self.angle -= 360;
        } else if self.angle < 0 {
            self.angle += 360;
        }

        let mut thrust = m.thrust;
        if self.shield > 0 {
            // cooldown: engine off, SHIELD / BOOST commands are ignored
            thrust = 0;
        } else if thrust == T_SHIELD {
            self.shield = SHIELD_TIMER;
            thrust = 0;
        } else if thrust == T_BOOST {
            if !self.boosted {
                self.boosted = true;
                thrust = 650;
            } else {
                thrust = MAX_THRUST;
            }
        }

        let t = thrust as f64;
        let i = (self.angle as usize) % 360;
        self.vel.x += trig.cos[i] * t;
        self.vel.y += trig.sin[i] * t;
    }
}

#[derive(Clone, Copy, Default)]
struct World {
    pods: [Pod; 4],
}

#[derive(Clone, Copy, Default)]
struct Solution {
    moves: [[Move; MAX_DEPTH]; 2],
    score: f64,
}

#[inline(always)]
fn new_collide(a: &Pod, b: &Pod) -> f64 {
    let p = b.pos - a.pos;
    let pl2 = p.mag_sq();
    if pl2 <= POD_RSQ {
        return 0.0;
    }
    let v = b.vel - a.vel;
    let dot = p.dot(v);
    if dot > 0.0 {
        return 10.0;
    }
    let vl2 = v.mag_sq();
    if vl2 < 1e-12 {
        return 10.0;
    }
    let disc = dot * dot - vl2 * (pl2 - POD_RSQ);
    if disc < 0.0 {
        return 10.0;
    }
    (-dot - disc.sqrt()) / vl2
}

#[inline(always)]
fn bounce(a: &mut Pod, b: &mut Pod) {
    let d = b.pos - a.pos;
    let mut dd = d.mag();
    let n = if dd > 1e-9 {
        Vec2::new(d.x / dd, d.y / dd)
    } else {
        dd = 0.0;
        Vec2::new(1.0, 0.0)
    };
    let relv = a.vel - b.vel;
    let m1 = if a.shield == SHIELD_TIMER { 0.1 } else { 1.0 };
    let m2 = if b.shield == SHIELD_TIMER { 0.1 } else { 1.0 };
    let mut force = n.dot(relv) / (m1 + m2);
    if force < MIN_IMPULSE {
        force += MIN_IMPULSE;
    } else {
        force += force;
    }
    let imp = n * (-force);
    a.vel = a.vel + imp * m1;
    b.vel = b.vel - imp * m2;
    if dd <= 800.0 {
        let push = (800.0 - dd) / 2.0 + EPS;
        a.pos = a.pos - n * push;
        b.pos = b.pos + n * push;
    }
}

#[inline(always)]
fn cp_collide(p1: Vec2, p2: Vec2, cp: Vec2) -> bool {
    let d = p2 - p1;
    let pd2 = d.mag_sq();
    let mut pp = p1;
    if pd2 != 0.0 {
        let u = ((cp.x - p1.x) * d.x + (cp.y - p1.y) * d.y) / pd2;
        if u > 1.0 {
            pp = p2;
        } else if u > 0.0 {
            pp = Vec2::new(p1.x + u * d.x, p1.y + u * d.y);
        }
    }
    (pp - cp).mag_sq() < CP_RSQ
}

/// Would opponent pod `p` shield against pod `q` this turn?
/// Only if off cooldown and a meaningful hit is imminent. Thresholds are guesses; tune them.
#[inline(always)]
fn opp_wants_shield(p: &Pod, q: &Pod) -> bool {
    if p.shield > 0 {
        return false;
    }
    let d = q.pos - p.pos;
    let dist = d.mag();
    if dist < 1.0 {
        return false;
    }
    // > 0 when the pods are approaching each other (units / turn)
    let closing = -d.dot(q.vel - p.vel) / dist;
    closing > 300.0 && dist - closing < 850.0
}

struct Params {
    depth: usize,
    pop: usize,
    w_vel: f64,
    w_vel_far: f64,
    vel_ramp: f64,
    w_face: f64,
    w_block: f64,
    w_second: f64,
    chase_thresh: f64,
    chase_alpha: f64,
    // dynamic role weights
    lead_scale: f64,
    lead_sec: f64,
    lead_blk: f64,
    behind_blk: f64,
    // opponent plan
    op_len: usize,
    op_share: f64,
    op_dist: f64,
    // boost gating
    boost_gate: bool,
    boost_leg: f64,
    boost_far: f64,
    boost_last: f64,
    boost_cos: f64,
    max_sims: u32,
    time_ms: u128,
    debug: u32,
}

fn env_f(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Bot {
    cps: Vec<Vec2>,
    n: usize,
    total: i32,
    rem: Vec<f64>,
    pop: Vec<Solution>,
    rng: u32,
    trig: TrigLut,
    params: Params,
    first_turn: bool,
    racer_idx: usize,
    role_swap_counter: i32,
    shield: [i32; 2],
    boost_used: [bool; 2],
    last_next: [usize; 4],
    passed: [i32; 4],
    // opponent modelling
    chase: [f64; 4],            // EMA of (heading . to_me - heading . to_cp); only [2],[3] used
    opp_chaser: Option<usize>,  // pod index (2 or 3) modelled as chaser this turn
    opp_cd: [i32; 2],           // inferred opponent shield cooldown (same convention as Pod::shield)
    prev_vel: [Vec2; 4],
    prev_pos: [Vec2; 4],
    // adaptive opponent plan (persisted between turns, shifted one ply each turn)
    opp_plan: OppPlan,
    // per-turn role weights (recomputed from the race state)
    cur_wblock: f64,
    cur_wsecond: f64,
    // longest checkpoint-to-checkpoint leg (for boost gating)
    max_leg: f64,
    // timing stats for the debug line (turn 1 excluded: it has a longer budget)
    dbg_turns: u32,
    dbg_sum_sims: u64,
    dbg_min_sims: u32,
    dbg_max_ms: u128,
    dbg_slow: u32,
}

impl Bot {
    fn new() -> Self {
        let w_vel = env_f("CSB_WVEL", 1.15);
        let depth = (env_f("CSB_DEPTH", 6.0) as usize).clamp(2, MAX_DEPTH);
        let params = Params {
            depth,
            pop: env_f("CSB_POP", 90.0) as usize,
            w_vel,
            w_vel_far: env_f("CSB_WVELFAR", 1.8),
            vel_ramp: env_f("CSB_VELRAMP", 3000.0).max(1.0),
            w_face: env_f("CSB_WFACE", 100.0),
            w_block: env_f("CSB_WBLOCK", 0.28),
            w_second: env_f("CSB_WSECOND", 0.12),
            chase_thresh: env_f("CSB_CHASE", 0.15),
            chase_alpha: env_f("CSB_CHASEA", 0.2),
            lead_scale: env_f("CSB_LEAD", 4000.0).max(1.0),
            lead_sec: env_f("CSB_LSEC", 0.0),
            lead_blk: env_f("CSB_LBLK", 0.0),
            behind_blk: env_f("CSB_BBLK", 0.0),
            // plan must stay shorter than the GA depth: the last ply of a shifted individual is random
            op_len: (env_f("CSB_OPLEN", 3.0) as usize)
                .min(MAX_OPP_LEN)
                .min(depth - 1),
            op_share: env_f("CSB_OPSHARE", 0.22).clamp(0.0, 0.8),
            op_dist: env_f("CSB_OPDIST", 4000.0),
            boost_gate: env_f("CSB_BOOSTGATE", 1.0) > 0.0,
            boost_leg: env_f("CSB_BOOSTLEG", 0.9),
            boost_far: env_f("CSB_BOOSTFAR", 4000.0),
            boost_last: env_f("CSB_BOOSTLAST", 3000.0),
            boost_cos: env_f("CSB_BOOSTCOS", 0.9),
            max_sims: env_f("CSB_SIMS", 0.0) as u32,
            time_ms: env_f("CSB_TIME", 50.0) as u128,
            debug: env_f("CSB_DEBUG", 1.0) as u32,
        };
        let w_block0 = params.w_block;
        let w_second0 = params.w_second;
        let pop_n = params.pop;
        Bot {
            cps: Vec::new(),
            n: 0,
            total: 0,
            rem: Vec::new(),
            pop: vec![Solution::default(); pop_n],
            rng: 987654321,
            trig: TrigLut::new(),
            params,
            first_turn: true,
            racer_idx: 0,
            role_swap_counter: 0,
            shield: [0; 2],
            boost_used: [false; 2],
            last_next: [1; 4],
            passed: [0; 4],
            chase: [0.0; 4],
            opp_chaser: None,
            opp_cd: [0; 2],
            prev_vel: [Vec2::default(); 4],
            prev_pos: [Vec2::default(); 4],
            opp_plan: [[Move::default(); MAX_OPP_LEN]; 2],
            cur_wblock: w_block0,
            cur_wsecond: w_second0,
            max_leg: 0.0,
            dbg_turns: 0,
            dbg_sum_sims: 0,
            dbg_min_sims: u32::MAX,
            dbg_max_ms: 0,
            dbg_slow: 0,
        }
    }

    #[inline(always)]
    fn rand(&mut self) -> u32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        self.rng
    }

    #[inline(always)]
    fn rand_angle(&mut self) -> i32 {
        (self.rand() % 37) as i32 - 18
    }

    /// True when the current phase of the turn has used up its share of the budget.
    /// `frac` is the cumulative fraction of the turn's budget at which this phase ends.
    /// In CSB_SIMS mode the budget is counted in simulations instead of milliseconds.
    #[inline(always)]
    fn budget_hit(&self, sims: u32, start: &Instant, time_limit: u128, frac: f64) -> bool {
        if self.params.max_sims > 0 {
            sims >= (self.params.max_sims as f64 * frac) as u32
        } else {
            sims & CLOCK_CHECK_MASK == 0
                && (start.elapsed().as_micros() as f64) >= time_limit as f64 * 1000.0 * frac
        }
    }

    /// Opponent policy for pod `i` (2 or 3).
    /// Racer model: head to next checkpoint. Chaser model: head to my racer's predicted position.
    /// Both: shield when off cooldown and a hit on one of my pods is imminent.
    #[inline(always)]
    fn opp_move(&self, pods: &[Pod; 4], i: usize) -> Move {
        let p = &pods[i];
        let target = if self.opp_chaser == Some(i) {
            let r = &pods[self.racer_idx];
            r.pos + r.vel * 3.0
        } else {
            self.cps[p.next % self.n]
        };
        let d = target - p.pos;
        let dist_sq = d.mag_sq();
        if dist_sq < 1.0 {
            return Move {
                angle: 0,
                thrust: MAX_THRUST,
            };
        }

        let inv = 1.0 / dist_sq.sqrt();
        let idx = (p.angle as usize) % 360;
        let hx = self.trig.cos[idx];
        let hy = self.trig.sin[idx];
        let cos_diff = (hx * d.x + hy * d.y) * inv;
        let sin_diff = (hx * d.y - hy * d.x) * inv;

        let turn = if cos_diff < 0.0 {
            if sin_diff >= 0.0 { 18 } else { -18 }
        } else {
            ((sin_diff * RAD_TO_DEG).clamp(-18.0, 18.0)).round() as i32
        };

        if opp_wants_shield(p, &pods[0]) || opp_wants_shield(p, &pods[1]) {
            return Move {
                angle: turn,
                thrust: T_SHIELD,
            };
        }

        let thrust = if cos_diff < COS_65 { 45 } else { MAX_THRUST };
        Move { angle: turn, thrust }
    }

    #[inline(always)]
    fn check_cp(&self, p: &mut Pod, from: Vec2) {
        if p.passed >= self.total {
            return;
        }
        if cp_collide(from, p.pos, self.cps[p.next % self.n]) {
            p.passed += 1;
            p.next = (p.next + 1) % self.n;
        }
    }

    /// One game turn. `om` = explicit moves for opponent pods 2 and 3; None = scripted opp_move().
    /// Returns the opponent moves that were actually applied (used to record scripted plans).
    fn step(&self, w: &mut World, m0: Move, m1: Move, om: Option<[Move; 2]>) -> [Move; 2] {
        w.pods[0].apply_move(m0, &self.trig);
        w.pods[1].apply_move(m1, &self.trig);

        // apply_move only changes velocity/angle/shield, so positions are still start-of-turn
        let used = match om {
            Some(o) => o,
            None => {
                let snap = w.pods;
                [self.opp_move(&snap, 2), self.opp_move(&snap, 3)]
            }
        };
        w.pods[2].apply_move(used[0], &self.trig);
        w.pods[3].apply_move(used[1], &self.trig);

        let mut starts = [
            w.pods[0].pos,
            w.pods[1].pos,
            w.pods[2].pos,
            w.pods[3].pos,
        ];
        let mut t = 1.0;
        let mut iter = 0;

        while t > 0.0 {
            let mut first = t;
            let mut ci = 0usize;
            let mut cj = 0usize;

            if iter < 8 {
                for i in (1..4).rev() {
                    for j in (0..i).rev() {
                        let tx = new_collide(&w.pods[i], &w.pods[j]);
                        if tx <= first {
                            first = tx;
                            ci = i;
                            cj = j;
                        }
                    }
                }
            }
            iter += 1;

            for p in w.pods.iter_mut() {
                p.pos = p.pos + p.vel * first;
            }
            t -= first;

            if ci != cj {
                let (l, r) = w.pods.split_at_mut(ci);
                bounce(&mut r[0], &mut l[cj]);
            }

            if t > 0.0 {
                for i in 0..4 {
                    self.check_cp(&mut w.pods[i], starts[i]);
                    starts[i] = w.pods[i].pos;
                }
            }
        }

        for i in 0..4 {
            let p = &mut w.pods[i];
            p.vel.x = trunc_f(p.vel.x * 0.85);
            p.vel.y = trunc_f(p.vel.y * 0.85);
            p.pos.x = round_ref(p.pos.x);
            p.pos.y = round_ref(p.pos.y);
            if p.shield > 0 {
                p.shield -= 1;
            }
            self.check_cp(p, starts[i]);
        }

        used
    }

    /// Leaf potential: distance to go + velocity bonus (ramped by distance) - facing penalty.
    #[inline(always)]
    fn potential(&self, p: &Pod) -> f64 {
        let cp = self.cps[p.next % self.n];
        let to = cp - p.pos;
        let d = to.mag();
        let (toward, cos_h) = if d > 1.0 {
            let a = (p.angle as usize) % 360;
            (
                p.vel.dot(to) / d,
                (self.trig.cos[a] * to.x + self.trig.sin[a] * to.y) / d,
            )
        } else {
            (0.0, 1.0)
        };
        // ~w_vel near the checkpoint, ramping to w_vel_far when far away
        let k = ((d - 600.0) / self.params.vel_ramp).clamp(0.0, 1.0);
        let wv = self.params.w_vel + (self.params.w_vel_far - self.params.w_vel) * k;

        -(d + self.rem.get((p.passed + 1) as usize).copied().unwrap_or(0.0))
            + wv * toward
            - self.params.w_face * (1.0 - cos_h)
    }

    /// Total distance a pod still has to cover (to its next cp plus all later legs).
    #[inline(always)]
    fn remaining(&self, p: &Pod) -> f64 {
        self.cps[p.next % self.n].dist(p.pos)
            + self.rem.get((p.passed + 1) as usize).copied().unwrap_or(0.0)
    }

    /// Is this a good moment for pod `p` to BOOST, if it first turns by `turn` degrees?
    /// Needs good alignment with the checkpoint, plus a long straight, the last lap, or the final cp.
    fn boost_ok(&self, p: &Pod, turn: i32) -> bool {
        let cur = self.cps[p.next % self.n];
        let to = cur - p.pos;
        let d = to.mag();
        if d < 1000.0 {
            return false;
        }
        let a = (p.angle + turn).rem_euclid(360) as usize;
        let cos_h = (self.trig.cos[a] * to.x + self.trig.sin[a] * to.y) / d;
        if cos_h < self.params.boost_cos {
            return false;
        }
        let prev = self.cps[(p.next + self.n - 1) % self.n];
        let leg = prev.dist(cur);
        let last_lap = p.passed >= self.total - self.n as i32;
        let final_cp = p.passed + 1 >= self.total;
        (leg >= self.params.boost_leg * self.max_leg && d > self.params.boost_far.min(0.6 * leg))
            || (last_lap && d > self.params.boost_last.min(0.5 * self.max_leg))
            || final_cp
    }

    /// My two moves for ply `i` of `sol`; BOOST is downgraded to full thrust when boost_ok() is false,
    /// so the simulation and the printed command always agree.
    #[inline(always)]
    fn my_moves(&self, w: &World, sol: &Solution, i: usize) -> (Move, Move) {
        let mut m0 = sol.moves[0][i];
        let mut m1 = sol.moves[1][i];
        if self.params.boost_gate {
            if m0.thrust == T_BOOST && !self.boost_ok(&w.pods[0], m0.angle) {
                m0.thrust = MAX_THRUST;
            }
            if m1.thrust == T_BOOST && !self.boost_ok(&w.pods[1], m1.angle) {
                m1.thrust = MAX_THRUST;
            }
        }
        (m0, m1)
    }

    /// Simulate `depth` turns. Err(score) = somebody finished (terminal score from MY side).
    fn rollout(&self, sol: &Solution, start: &World, plan: &OppPlan) -> Result<World, f64> {
        let depth = self.params.depth;
        let op_len = self.params.op_len;
        let mut w = *start;

        for i in 0..depth {
            let om = if i < op_len {
                Some([plan[0][i], plan[1][i]])
            } else {
                None
            };
            let (m0, m1) = self.my_moves(&w, sol, i);
            self.step(&mut w, m0, m1, om);

            if w.pods[0].passed >= self.total || w.pods[1].passed >= self.total {
                return Err(1.0e7 + (depth - i) as f64 * 1.0e4);
            }
            if w.pods[2].passed >= self.total || w.pods[3].passed >= self.total {
                return Err(-1.0e7 - (depth - i) as f64 * 1.0e4);
            }
        }
        Ok(w)
    }

    /// My leaf score (higher = better for me).
    fn leaf(&self, w: &World) -> f64 {
        let racer = self.racer_idx;
        let blocker = 1 - racer;

        let pr = self.potential(&w.pods[racer]);
        let pb = self.potential(&w.pods[blocker]);
        let o0 = self.potential(&w.pods[2]);
        let o1 = self.potential(&w.pods[3]);
        let (leader, po) = if o0 >= o1 { (2, o0) } else { (3, o1) };

        let dist = w.pods[blocker].pos.dist(w.pods[leader].pos);

        let block_score = if dist < 2200.0 {
            let to_cp = self.cps[w.pods[leader].next % self.n] - w.pods[leader].pos;
            let to_blocker = w.pods[blocker].pos - w.pods[leader].pos;
            let align = to_cp.normalize().dot(to_blocker.normalize());
            dist * self.cur_wblock - align * 180.0
        } else {
            dist * self.cur_wblock
        };

        (pr - po) + self.cur_wsecond * pb - block_score
    }

    /// Opponent's leaf score (higher = better for THEM). Pure race differential, no blocking
    /// heuristic: collisions and shields still matter through the potentials of the pods.
    fn opp_leaf(&self, w: &World) -> f64 {
        let racer = self.racer_idx;
        let pr = self.potential(&w.pods[racer]);
        let pb = self.potential(&w.pods[1 - racer]);
        let o0 = self.potential(&w.pods[2]);
        let o1 = self.potential(&w.pods[3]);
        let (hi, lo) = if o0 >= o1 { (o0, o1) } else { (o1, o0) };
        (hi - pr) + self.params.w_second * (lo - pb)
    }

    fn evaluate_with(&self, sol: &Solution, start: &World, plan: &OppPlan) -> f64 {
        match self.rollout(sol, start, plan) {
            Err(s) => s,
            Ok(w) => self.leaf(&w),
        }
    }

    #[inline(always)]
    fn evaluate(&self, sol: &Solution, start: &World) -> f64 {
        self.evaluate_with(sol, start, &self.opp_plan)
    }

    /// Score of an opponent plan from the opponent's side, given my solution.
    fn opp_objective(&self, sol: &Solution, start: &World, plan: &OppPlan) -> f64 {
        match self.rollout(sol, start, plan) {
            Err(s) => -s,
            Ok(w) => self.opp_leaf(&w),
        }
    }

    /// Record what the scripted opponent would do over the first op_len plies against `sol`.
    fn scripted_plan(&self, start: &World, sol: &Solution) -> OppPlan {
        let mut plan = [[Move::default(); MAX_OPP_LEN]; 2];
        let mut w = *start;
        for i in 0..self.params.op_len {
            let (m0, m1) = self.my_moves(&w, sol, i);
            let used = self.step(&mut w, m0, m1, None);
            plan[0][i] = used[0];
            plan[1][i] = used[1];
        }
        plan
    }

    /// Hill-climb the opponent plan against `mine`. Candidates: last turn's plan (already shifted)
    /// and the scripted plan; then random single-move mutations (strict improvement only, so pods
    /// that do not matter keep their scripted behaviour). `sims` is shared with the GA budget.
    fn climb_opponent(
        &mut self,
        world: &World,
        mine: &Solution,
        start: &Instant,
        time_limit: u128,
        sims: &mut u32,
        climb: bool,
    ) {
        let len = self.params.op_len;
        let scripted = self.scripted_plan(world, mine);

        if self.first_turn || !climb {
            self.opp_plan = scripted;
            return;
        }

        let mut best = scripted;
        let mut best_s = self.opp_objective(mine, world, &best);
        let prev = self.opp_plan;
        let prev_s = self.opp_objective(mine, world, &prev);
        if prev_s > best_s {
            best = prev;
            best_s = prev_s;
        }

        loop {
            if self.budget_hit(*sims, start, time_limit, self.params.op_share) {
                break;
            }
            *sims += 1;

            let mut cand = best;
            let p = (self.rand() % 2) as usize;
            let t = (self.rand() % len as u32) as usize;
            let r = self.rand() % 100;

            let a = cand[p][t].angle;
            let na = if r < 50 {
                a + (self.rand() % 11) as i32 - 5
            } else if r < 75 {
                self.rand_angle()
            } else if r < 87 {
                18
            } else {
                -18
            };
            cand[p][t].angle = na.clamp(-18, 18);

            if self.rand() % 3 == 0 {
                let q = self.rand() % 100;
                cand[p][t].thrust = if q < 40 {
                    MAX_THRUST
                } else if q < 55 {
                    0
                } else if q < 80 {
                    (self.rand() % (MAX_THRUST as u32 + 1)) as i32
                } else {
                    T_SHIELD
                };
            }

            let s = self.opp_objective(mine, world, &cand);
            if s > best_s {
                best = cand;
                best_s = s;
            }
        }

        self.opp_plan = best;
    }

    /// Recompute w_block / w_second from the race state (k > 0: I am ahead, k < 0: I am behind).
    fn update_weights(&mut self, world: &World) {
        let my_rem = self.remaining(&world.pods[self.racer_idx]);
        let opp_rem = self
            .remaining(&world.pods[2])
            .min(self.remaining(&world.pods[3]));
        let k = ((opp_rem - my_rem) / self.params.lead_scale).clamp(-1.0, 1.0);
        let ahead = k.max(0.0);
        let behind = (-k).max(0.0);
        self.cur_wsecond = self.params.w_second * (1.0 + self.params.lead_sec * ahead);
        self.cur_wblock = self.params.w_block
            * (1.0 - self.params.lead_blk * ahead + self.params.behind_blk * behind);
    }

    /// Infer opponent shield cooldown from observed acceleration.
    /// Input velocity v_now = trunc(0.85 * (v_prev + thrust * dir)) when no collision happened, so
    /// (v_now / 0.85 - v_prev) is the thrust vector. Near-zero thrust with my pod close by on the
    /// previous turn => the opponent most likely shielded.
    fn observe_opponents(&mut self, world: &mut World) {
        for k in 0..2 {
            let i = k + 2;
            let was_cd = self.opp_cd[k] > 0;
            if was_cd {
                self.opp_cd[k] -= 1;
            }
            if !self.first_turn && !was_cd {
                let p = &world.pods[i];
                let a = (p.angle as usize) % 360;
                let dx = self.trig.cos[a];
                let dy = self.trig.sin[a];
                let ax = p.vel.x / 0.85 - self.prev_vel[i].x;
                let ay = p.vel.y / 0.85 - self.prev_vel[i].y;
                let along = ax * dx + ay * dy; // estimated thrust
                let perp = ax * dy - ay * dx;  // should be ~0 if no collision
                if along.abs() < 6.0 && perp.abs() < 6.0 {
                    let near = (0..2).any(|j| self.prev_pos[j].dist(self.prev_pos[i]) < 1700.0);
                    if near {
                        // activated last turn: shield counter is SHIELD_TIMER - 1 at input time
                        self.opp_cd[k] = SHIELD_TIMER - 1;
                    }
                }
            }
            world.pods[i].shield = self.opp_cd[k];
        }
    }

    /// Update chaser detection and decide which opponent pod (if any) is modelled as chaser.
    fn update_chase(&mut self, world: &World) {
        let me = world.pods[self.racer_idx];
        let alpha = self.params.chase_alpha;
        for i in 2..4 {
            let p = &world.pods[i];
            let a = (p.angle as usize) % 360;
            let h = Vec2::new(self.trig.cos[a], self.trig.sin[a]);
            let to_cp = (self.cps[p.next % self.n] - p.pos).normalize();
            let to_me = (me.pos - p.pos).normalize();
            let s = h.dot(to_me) - h.dot(to_cp);
            self.chase[i] += alpha * (s - self.chase[i]);
        }
        // only the weaker opponent pod can be the chaser
        let o0 = self.potential(&world.pods[2]);
        let o1 = self.potential(&world.pods[3]);
        let weaker = if o0 < o1 { 2 } else { 3 };
        self.opp_chaser = if self.chase[weaker] > self.params.chase_thresh {
            Some(weaker)
        } else {
            None
        };
    }

    fn run(&mut self) {
        let stdin = io::stdin();
        let mut lines = stdin.lock().lines();

        let laps: i32 = lines.next().unwrap().unwrap().trim().parse().unwrap();
        let ncp: usize = lines.next().unwrap().unwrap().trim().parse().unwrap();
        for _ in 0..ncp {
            let line = lines.next().unwrap().unwrap();
            let mut it = line.split_whitespace();
            let x: f64 = it.next().unwrap().parse().unwrap();
            let y: f64 = it.next().unwrap().parse().unwrap();
            self.cps.push(Vec2::new(x, y));
        }
        self.n = ncp.max(1);
        self.max_leg = (0..self.n)
            .map(|i| self.cps[i].dist(self.cps[(i + 1) % self.n]))
            .fold(0.0, f64::max);
        self.total = laps * self.n as i32;
        self.rem = vec![0.0; self.total as usize + 5];
        for i in (1..self.total as usize).rev() {
            let a = self.cps[i % self.n];
            let b = self.cps[(i + 1) % self.n];
            self.rem[i] = a.dist(b) + self.rem[i + 1];
        }

        loop {
            let mut start_time = Instant::now();
            let mut world = World::default();

            for i in 0..4 {
                let line = match lines.next() {
                    Some(Ok(l)) => l,
                    _ => return,
                };
                if i == 0 {
                    start_time = Instant::now();
                }
                let mut it = line.split_whitespace();
                let pod = &mut world.pods[i];
                pod.pos.x = it.next().unwrap().parse().unwrap();
                pod.pos.y = it.next().unwrap().parse().unwrap();
                pod.vel.x = it.next().unwrap().parse().unwrap();
                pod.vel.y = it.next().unwrap().parse().unwrap();
                let raw_angle: f64 = it.next().unwrap().parse().unwrap();
                pod.next = it.next().unwrap().parse().unwrap();

                if self.first_turn && raw_angle < 0.0 {
                    let d = self.cps[pod.next % self.n] - pod.pos;
                    pod.angle = ((d.y.atan2(d.x) * RAD_TO_DEG).round() as i32).rem_euclid(360);
                } else {
                    pod.angle = (raw_angle.round() as i32).rem_euclid(360);
                }

                if !self.first_turn && pod.next != self.last_next[i] {
                    self.passed[i] += 1;
                }
                self.last_next[i] = pod.next;
                pod.passed = self.passed[i];

                if i < 2 {
                    pod.shield = self.shield[i];
                    pod.boosted = self.boost_used[i];
                } else {
                    pod.boosted = true;
                }
            }

            // Opponent shield-cooldown inference (needs prev_* from the previous turn)
            self.observe_opponents(&mut world);
            for i in 0..4 {
                self.prev_vel[i] = world.pods[i].vel;
                self.prev_pos[i] = world.pods[i].pos;
            }

            // Role assignment with hysteresis
            let pot0 = self.potential(&world.pods[0]);
            let pot1 = self.potential(&world.pods[1]);
            let proposed = if pot0 >= pot1 { 0 } else { 1 };
            if self.first_turn {
                self.racer_idx = proposed;
            } else if proposed != self.racer_idx {
                self.role_swap_counter += 1;
                if self.role_swap_counter >= ROLE_SWAP_HYSTERESIS {
                    self.racer_idx = proposed;
                    self.role_swap_counter = 0;
                }
            } else {
                self.role_swap_counter = 0;
            }

            // Chaser detection (needs racer_idx)
            self.update_chase(&world);

            // Race-state dependent weights (needs racer_idx)
            self.update_weights(&world);

            let depth = self.params.depth;
            let pop_size = self.params.pop;
            let first = self.first_turn;
            let my_blocker = 1 - self.racer_idx;
            let op_len = self.params.op_len;

            // [FIX 1b] opponent racer by potential (ties no longer always go to pod 2)
            let opp_racer =
                if self.potential(&world.pods[2]) >= self.potential(&world.pods[3]) { 2 } else { 3 };

            let time_limit: u128 = if first { 850 } else { self.params.time_ms };
            let mut sim_count: u32 = 0;

            // Shift last turn's opponent plan by one ply
            if !first && op_len > 0 {
                for p in 0..2 {
                    self.opp_plan[p].copy_within(1..op_len, 0);
                }
            }

            // Seed population (scores are computed after the opponent phase so that every
            // individual is scored against the same opponent plan)
            for k in 0..pop_size {
                let mut s = self.pop[k];
                for p in 0..2 {
                    let begin = if first {
                        0
                    } else {
                        s.moves[p].copy_within(1..depth, 0);
                        depth - 1
                    };
                    for t in begin..depth {
                        s.moves[p][t] = Move {
                            angle: self.rand_angle(),
                            thrust: MAX_THRUST,
                        };
                    }
                }

                // [FIX 1a] aim-at-opponent-racer seed goes into the WORST 8 (the population is sorted
                // best-first), and the heading is tracked while the pod rotates.
                if k + 8 >= pop_size {
                    let target = world.pods[opp_racer].pos * 0.4
                        + self.cps[world.pods[opp_racer].next % self.n] * 0.6;
                    let d = target - world.pods[my_blocker].pos;
                    let aim = ((d.y.atan2(d.x) * RAD_TO_DEG).round() as i32).rem_euclid(360);
                    let mut heading = world.pods[my_blocker].angle;
                    for t in 0..depth.min(3) {
                        let mut diff = (aim - heading).rem_euclid(360);
                        if diff > 180 {
                            diff -= 360;
                        }
                        let ang = diff.clamp(-18, 18);
                        heading = (heading + ang).rem_euclid(360);
                        s.moves[my_blocker][t] = Move {
                            angle: ang,
                            thrust: MAX_THRUST,
                        };
                    }
                }

                self.pop[k] = s;
            }

            // Opponent phase: adapt the opponent's first plies against my previous best (shifted).
            // Skipped (scripted plan only) when nobody is close enough to interact within the horizon.
            let near = (0..2).any(|i| {
                (2..4).any(|j| world.pods[i].pos.dist(world.pods[j].pos) < self.params.op_dist)
            });
            if op_len > 0 {
                let mine = self.pop[0];
                self.climb_opponent(&world, &mine, &start_time, time_limit, &mut sim_count, near);
            }

            // Score + sort the seeded population against the (new) opponent plan
            for k in 0..pop_size {
                let sc = self.evaluate(&self.pop[k], &world);
                self.pop[k].score = sc;
            }
            self.pop
                .sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

            // GA loop
            let elite = (pop_size as u32 / 3).max(1);

            loop {
                if self.budget_hit(sim_count, &start_time, time_limit, 1.0) {
                    break;
                }
                sim_count += 1;

                let ei = (self.rand() % elite) as usize;
                let mut child = self.pop[ei];

                // Crossover
                if self.rand() % 100 < 60 {
                    let other = (self.rand() % elite) as usize;
                    for p in 0..2 {
                        for t in 0..depth {
                            if self.rand() % 2 == 0 {
                                child.moves[p][t] = self.pop[other].moves[p][t];
                            }
                        }
                    }
                }

                // Mutation
                let pod = (self.rand() % 2) as usize;
                let turn = (self.rand() % depth as u32) as usize;
                let r = self.rand() % 100;

                {
                    let a = child.moves[pod][turn].angle;
                    let na = if r < 55 {
                        a + (self.rand() % 11) as i32 - 5
                    } else if r < 80 {
                        self.rand_angle()
                    } else if r < 90 {
                        18
                    } else {
                        -18
                    };
                    child.moves[pod][turn].angle = na.clamp(-18, 18);
                }

                if self.rand() % 4 == 0 {
                    let q = self.rand() % 100;
                    child.moves[pod][turn].thrust = if q < 42 {
                        MAX_THRUST
                    } else if q < 58 {
                        0
                    } else if q < 88 {
                        (self.rand() % (MAX_THRUST as u32 + 1)) as i32
                    } else if q < 95 {
                        T_SHIELD
                    } else if !self.boost_used[pod] {
                        T_BOOST
                    } else {
                        MAX_THRUST
                    };
                }

                child.score = self.evaluate(&child, &world);

                if child.score > self.pop[pop_size - 1].score {
                    self.pop[pop_size - 1] = child;
                    for i in (1..pop_size).rev() {
                        if self.pop[i].score > self.pop[i - 1].score {
                            self.pop.swap(i, i - 1);
                        } else {
                            break;
                        }
                    }
                }
            }

            // Output
            let best = self.pop[0];
            for my in 0..2 {
                let m = best.moves[my][0];
                let pod = world.pods[my];
                let aim = (pod.angle + m.angle).rem_euclid(360) as usize % 360;
                let tx = (pod.pos.x + self.trig.cos[aim] * 10000.0) as i32;
                let ty = (pod.pos.y + self.trig.sin[aim] * 10000.0) as i32;
                let role = if my == self.racer_idx { "RACER" } else { "BLOCKER" };

                // during cooldown the game ignores SHIELD/BOOST: print plain thrust 0
                // and do not touch the timers, so the internal state stays in sync with the sim.
                let cooling = self.shield[my] > 0;

                if m.thrust == T_SHIELD && !cooling {
                    println!("{} {} SHIELD {}", tx, ty, role);
                    self.shield[my] = SHIELD_TIMER - 1;
                } else if m.thrust == T_BOOST
                    && !cooling
                    && !self.boost_used[my]
                    && (!self.params.boost_gate || self.boost_ok(&pod, m.angle))
                {
                    println!("{} {} BOOST {}", tx, ty, role);
                    self.boost_used[my] = true;
                } else {
                    let th = if cooling || m.thrust == T_SHIELD {
                        0
                    } else if m.thrust == T_BOOST {
                        MAX_THRUST
                    } else {
                        m.thrust.clamp(0, MAX_THRUST)
                    };
                    println!("{} {} {} {}", tx, ty, th, role);
                    self.shield[my] = (self.shield[my] - 1).max(0);
                }
            }

            if self.params.debug > 0 {
                let el = start_time.elapsed().as_millis();
                if !first {
                    self.dbg_turns += 1;
                    self.dbg_sum_sims += sim_count as u64;
                    self.dbg_min_sims = self.dbg_min_sims.min(sim_count);
                    self.dbg_max_ms = self.dbg_max_ms.max(el);
                    if el >= 65 {
                        self.dbg_slow += 1;
                    }
                }
                let avg = if self.dbg_turns > 0 {
                    self.dbg_sum_sims / self.dbg_turns as u64
                } else {
                    0
                };
                eprintln!(
                    "sims {} t {}ms | turns {} avg {} min {} maxt {}ms slow {}",
                    sim_count, el, self.dbg_turns, avg, self.dbg_min_sims, self.dbg_max_ms, self.dbg_slow
                );
            }
            if self.params.debug >= 2 {
                eprintln!(
                    "sims {} t {}ms racer {} chase [{:.2} {:.2}] chaser {:?} opp_cd {:?} wb {:.3} ws {:.3} near {} op0 {:?} op1 {:?} boost {:?}",
                    sim_count,
                    start_time.elapsed().as_millis(),
                    self.racer_idx,
                    self.chase[2],
                    self.chase[3],
                    self.opp_chaser,
                    self.opp_cd,
                    self.cur_wblock,
                    self.cur_wsecond,
                    near,
                    if op_len > 0 { Some(self.opp_plan[0][0]) } else { None },
                    if op_len > 0 { Some(self.opp_plan[1][0]) } else { None },
                    self.boost_used
                );
            }

            self.first_turn = false;
        }
    }
}

fn main() {
    let mut bot = Bot::new();
    bot.run();
}
