# Inkwell & Algorithm

Data-driven Premier League predictions, with the reasons behind each call, published as
a newspaper-style site on GitHub Pages: https://garygunn94.github.io/premier-league-predictions/

Follow @InkwellAlgo on X and Instagram.

For every match the site gives a call (home win, draw or away win), how confident we
are, a predicted score, and the three biggest reasons for the call, each backed by
real numbers.

## Data

We only collect the current season and the two before it. Recent form matters most,
so the model weights the current season most heavily. The window is set by
`SEASONS_KEPT` in `plpred/seasons.py`.

| Source | What we use | Folder |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/englandm.php) | Results, shots, corners, cards, bookmaker odds | `data/raw/football-data-uk/`, combined in `data/matches.csv` |
| [Understat](https://understat.com/league/EPL) | Expected goals (xG) per match | `data/xg/` |
| [Fantasy Premier League](https://fantasy.premierleague.com/) | Player availability, injuries, suspensions, chance of playing | `data/upcoming/fpl_players.csv`, `key_absences.csv` |
| [API-Football](https://www.api-football.com/) | This round's fixtures and pre-match odds | `data/upcoming/fixtures.csv`, `odds.csv` |

The **Update data** workflow refreshes everything on Thursday and Saturday mornings
and commits the results, so every week's inputs stay in the history. It can also be
run by hand from the Actions tab.

API-Football needs a key saved as the secret `API_FOOTBALL_KEY` in the `prod`
environment (Settings > Environments > prod). Only the scheduled data job uses that
environment, and it makes about 12 calls a week: 2 per run for fixtures, plus odds
once per match. Pull-request checks never use the key. Without it, the other
sources still update.

## Model

`plpred/model.py` gives every team an attack and a defence rating, fitted to recent
matches with older games fading out (half-life 180 days, so this season counts most).
Ratings are fitted to a 50/50 blend of goals and xG, since xG says more about how a
team is really playing than a few results. The ratings give expected goals for any
match, which become win/draw/loss chances and a predicted score (Dixon-Coles model).

`python -m plpred.backtest` replays past weeks using only what was known at the time,
tunes the settings on last season, and checks them on this season. Results are saved
in `data/backtest/`. As of 18 Oct 2026:

| Season | Model accuracy | Bookmakers' favourite | Always home | Exact score |
|---|---|---|---|---|
| 2025/26 (tuning) | 49.2% | 49.5% | 42.6% | 9.7% |
| 2026/27 so far (unseen, 60 games) | 50.0% | 45.0% | 38.3% | 8.3% |

## Reasons

`python -m plpred.predict` turns this round's fixtures into
`data/predictions/<date>.json`: for every match, the call, its chances against the
bookmakers', likely scores, and why. The "why" has two parts:

- **How we got there.** Each match is split into factors: each side's attack and
  defence, and each side's team news. Starting from an average home side against an
  average visitor, every factor's exact share of the final percentage is worked out
  (Shapley values), so the steps always add up to our number.
- **Reasons a reader can check:** recent xG home and away, form, finishing against
  xG, key absences and head to head, plus a headline and what would change our mind.

Team news comes from the FPL feed. A missing regular reduces his team's attack by
half his share of its goal involvements, and a missing defender or keeper raises
goals conceded a little. These weights are judgement calls, not fitted, because we
have no history of who was missing when. Calls with less than 5 points between the
top two outcomes are labelled "too close to call".

`python -m plpred.predict --replay 2026-10-05` re-predicts a past week from
earlier results only (without team news), which is handy for checking the output.

## The site

`python -m plpred.site` builds the site into `site/` from the newest predictions:
a front page (match of the week, the rest of the round, charts, our record, last
week's verdict and the table) and a breakdown page for every match. It follows the
approved newspaper design, works on phones and has light and dark modes. Club crests
are saved from API-Football into `data/badges/`.

Each build also writes a posting kit for X and Instagram to `site/social/` (on the
live site at `/social/`, not linked from the front page). It has square images in the
site's dark look (the round, every call, the reasons chart for the bold call and the
scorecard) and the captions for Thursday, Friday and Monday, with X posts already
under 280 characters. The workflow renders the images to PNG with
`python -m plpred.social --png site/social`, which needs Playwright and Chromium.

The **Publish site** workflow builds and deploys it to GitHub Pages after every data
update and every push to `main`. GitHub Pages must be set to deploy from GitHub
Actions (Settings > Pages > Source).

## Running locally

```sh
pip install -r requirements.txt
python -m plpred.update            # all sources
python -m plpred.update history    # or just some: xg, history, fpl, fixtures
python -m plpred.backtest         # replay past weeks and score the model
python -m plpred.predict          # predict the coming round, with reasons
python -m plpred.site             # build the site into site/
python -m pytest
```

## Roadmap

1. Data pipeline
2. Team-strength model (Dixon-Coles) with a back-test
3. Ranked reasons for each call
4. The site on GitHub Pages
5. Weekly automation and a season scorecard
