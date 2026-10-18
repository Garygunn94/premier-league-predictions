# Pre-registered test: do we see through luck? (2026/27)

Written 18 October 2026, before any live call it covers. The date of the commit that added
this file is the timestamp. The rule will not change during the season; if it ever has to,
the change gets its own dated commit and this file says so.

## Why

Our ratings are fitted to a 50/50 blend of goals and expected goals (xG). Back-tests on 707
matches from October 2024 to October 2026 hinted that the model does better than the
bookmakers in the games where one side's results have run furthest from its chances:

- In the third of games with the biggest luck gap (236), our ranked probability score was
  0.0039 better than the bookmakers'.
- In the other two thirds (471) it was 0.0029 worse.
- Neither is statistically significant (t ≈ ±1.1). This test is how we find out.

The code that scores it is `plpred/luck.py`.

## The rule

1. **Luck gap.** For each team, take its last 20 Premier League games before the match
   (at least 8 needed), with xG from Understat. Per game, work out
   (goals scored − xG for) − (goals conceded − xG against), then average it. The luck gap is
   the home side's figure minus the away side's. `plpred.luck.luck_gap` computes it.
2. **Recorded before kick-off.** Every published call stores its luck gap in the
   committed predictions file (`luck_gap` in `data/predictions/<date>.json`).
3. **In the test** if the absolute luck gap is **0.45 or more**. (0.45 was the top third of
   back-test games. It is fixed now, not re-estimated.)
4. **Which calls count.** Live calls only: predictions files dated 19 October 2026 or later,
   never replays or back-tests. Each game's call is the probabilities in the newest
   committed file for it before kick-off.
5. **Benchmark.** The bookmakers' average pre-match odds from football-data.co.uk
   (`AvgH/AvgD/AvgA`), margin removed by scaling to 100%, exactly as in the back-test.
6. **Measure.** Mean ranked probability score (RPS) over the games in the test, ours against
   the bookmakers'. Also reported: right results for our pick and for the bookmakers'
   favourite.
7. **When.** Scored after the last game of 2026/27. A running tally shows on the site's record
   page but is not the result.

## What counts as a result

- **Supported:** our mean RPS is lower than the bookmakers' over the test games, and a paired
  one-sided t-test on the per-game RPS differences gives p < 0.05.
- **Hint only:** lower, but p ≥ 0.05. We say so and carry the test into 2027/28 unchanged.
- **Not supported:** our RPS is equal or higher. We say so on the site.

With roughly 100 to 120 test games in a season, only a gap of about 0.01 RPS would reach
p < 0.05, so a single season is most likely to end as "hint only" or "not supported". We
publish whichever it is.
