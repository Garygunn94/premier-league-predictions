# Premier League Predictions

Weekly Premier League predictions with the reasons behind each call, published as a
newspaper-style "backpage" on GitHub Pages.

For every match the site gives a call (home win, draw or away win), how confident we
are, a predicted score, and the three biggest reasons for the call, each backed by
real numbers.

## Data

| Source | What we use | Folder |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/englandm.php) | Every result since 1993/94, shots, corners, cards, bookmaker odds | `data/raw/football-data-uk/`, combined in `data/matches.csv` |
| [Understat](https://understat.com/league/EPL) | Expected goals (xG) per match since 2014/15 | `data/xg/` |
| [Fantasy Premier League](https://fantasy.premierleague.com/) | Player availability, injury news, chance of playing | `data/upcoming/fpl_players.csv`, `key_absences.csv` |
| [API-Football](https://www.api-football.com/) | This round's fixtures, injuries and pre-match odds | `data/upcoming/fixtures.csv`, `injuries.csv`, `odds.csv` |

The **Update data** workflow refreshes everything on Thursday and Saturday mornings
and commits the results, so every week's inputs stay in the history. It can also be
run by hand from the Actions tab.

API-Football needs a free key saved as the repository secret `API_FOOTBALL_KEY`
(Settings > Secrets and variables > Actions). Without it, the other sources still update.

## Running locally

```sh
pip install -r requirements.txt
python -m plpred.update            # all sources
python -m plpred.update history    # or just some: xg, history, fpl, fixtures
python -m pytest
```

## Roadmap

1. Data pipeline (this step)
2. Team-strength model (Dixon-Coles) with a back-test on the last 3 seasons
3. Ranked reasons for each call
4. The backpage site on GitHub Pages
5. Weekly automation and a season scorecard
