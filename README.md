# Premier League Predictions

Weekly Premier League predictions with the reasons behind each call, published as a
newspaper-style "backpage" on GitHub Pages.

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

## Running locally

```sh
pip install -r requirements.txt
python -m plpred.update            # all sources
python -m plpred.update history    # or just some: xg, history, fpl, fixtures
python -m plpred.backtest         # replay past weeks and score the model
python -m pytest
```

## Roadmap

1. Data pipeline
2. Team-strength model (Dixon-Coles) with a back-test
3. Ranked reasons for each call
4. The backpage site on GitHub Pages
5. Weekly automation and a season scorecard
