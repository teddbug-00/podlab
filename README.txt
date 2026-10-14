PODLAB - local test arena for your Mad Pod Racing bot
=====================================================

WHAT YOU NEED (once): Rust.   https://rustup.rs   (install it, then close and reopen your terminal)

HOW TO RUN
  Windows:    double-click  run.bat
  Mac/Linux:  ./run.sh
  (or in a terminal inside this folder:  cargo run --release)

The first run compiles for about a minute. Then it plays one test game, and then bot-vs-bot games on all your
CPU cores for 20 minutes. Progress prints every 20 seconds, and the table is saved to  results.txt  every minute.
WHEN IT IS DONE: send me results.txt (or paste the table at the bottom of the window).

HOW MANY GAMES TO EXPECT
  Each game uses two CPU threads. On a small machine that is about 0.4 games per second (~1400 per hour).
  Precision is roughly +-100/sqrt(games) percentage points, so 100 games per candidate can only show huge
  differences and ~1000 games shows differences of about 3 points. Fewer candidates = more games each:
     ./run.sh --only depth8,boost_off --minutes 45

NO FAST COMPUTER? SETUP-GITHUB.txt explains how to run every candidate in parallel on free GitHub machines.

WHAT IT TESTS
  bots/a.rs is the baseline (your v4_safe, unchanged). experiments.txt lists candidates, each one a bot setting
  changed. Every candidate plays the baseline on both sides of identical random maps. A candidate is only called
  BETTER/WORSE when the 95% confidence interval excludes 50%; "unclear" means "needs more games".

OPTIONS (after  --  when using cargo, or directly with run.bat / run.sh)
  --minutes 10          shorter or longer run
  --sims 8000           more simulations per turn: slower but closer to the real bot (it gets ~30000+ on CodinGame)
  --only depth8,no_opp  run just these candidates (more games each in the same time)
  --workers 4           games in parallel (default: half of your CPU threads)

NOTES
  * The arena uses a fixed number of simulations per turn instead of milliseconds, so results do not depend on
    how busy your computer is. It does mean timing/timeout behaviour is NOT tested here.
  * The opponent is your own baseline bot, not the real Legend field. Treat results as "better or worse than my
    current bot", not as a rank prediction. Confirm real wins on the ladder.
  * To test a new bot version: replace bots/b.rs with it, and add a line like   v7: bot=b   to experiments.txt.
