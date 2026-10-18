# Inkwell & Algorithm

Data-driven Premier League predictions, with the reasons behind each call, published as
a newspaper-style site on GitHub Pages: https://garygunn94.github.io/premier-league-predictions/

Follow [@InkwellAlgo on X](https://x.com/InkwellAlgo) and [@inkwellandalgorithm on Instagram](https://www.instagram.com/inkwellandalgorithm/).

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

`python -m plpred.backtest` replays every past gameweek using only what was known before
the round's first kickoff, tunes the settings on last season, and checks them on this
season. Ratings for teams with little history (usually promoted sides) are pulled firmly
towards average, so one big result can't make them look like contenders. Results are
saved in `data/backtest/`. As of 18 Oct 2026:

| Season | Model accuracy | Bookmakers' favourite | Always home | Exact score |
|---|---|---|---|---|
| 2025/26 (tuning) | 49.7% | 49.5% | 42.6% | 9.7% |
| 2026/27 so far (unseen, 60 games) | 48.3% | 45.0% | 38.3% | 6.7% |

These samples are small: over 60 games a hit rate can swing about 13 points either way by
chance, so treat the model as level with the bookmakers until several seasons say otherwise.

## Reasons

`python -m plpred.predict` turns this round's fixtures into
`data/predictions/<date>.json`: for every match, the call, its chances against the
bookmakers', likely scores, and why. The "why" has two parts:

- **How we got there.** Each match is split into factors: each side's attack and
  defence, and each side's team news. Starting from an average home side against an
  average visitor, every factor's exact share of the final percentage is worked out
  (Shapley values), so the steps always add up to our number.
- **Reasons a reader can check:** recent xG home and away, form, results against
  performance, key absences and head to head, plus a headline and what would change
  our mind. "Results vs performance" (`plpred/luck.py`) compares each side's points over
  its last 10 league games with the points its chances were worth (expected points, from
  xG); a gap of 3 or more gets a reason, and it takes the second slot on the front page.
  Each call also stores its `luck_gap` for the pre-registered test in
  [preregistered/](preregistered/luck-gap-2026-27.md).

Team news comes from the FPL feed. A missing regular reduces his team's attack by
half his share of its goal involvements, and a missing defender or keeper raises
goals conceded a little. These weights are judgement calls, not fitted, because we
have no history of who was missing when. Calls with less than 5 points between the
top two outcomes are labelled "too close to call".

`python -m plpred.predict --replay 2026-10-05` re-predicts a past week from
earlier results only (without team news), which is handy for checking the output.

## The site

`python -m plpred.site` builds the site into `site/` from every predictions file:
a home page (what the site is, how it works, the data sources, our record and a list
of every matchweek), then for each matchweek a page at `matchweek/<n>/` (match of the
week, the rest of the round, charts, our record, last week's verdict, the table and the
table the chances deserved) with a breakdown page for every match beside it, and a
record page at `record/` showing how often our percentages came true, our labels'
hit rates, our score against the bookmakers and the running tally of the
pre-registered luck-gap test. It follows the
approved newspaper design, works on phones and has light and dark modes. Club crests
are saved from API-Football into `data/badges/`.

Alongside those:

- **Season projection** (`table/`, `plpred/simulate.py`): the rest of the season played
  out 10,000 times from the team ratings, giving each club's title, top-four and
  relegation chances, expected points and likely finishing range, with the change since
  a week earlier. The remaining fixtures are the unplayed home-and-away pairings, so it
  needs no fixture list.
- **FPL corner** (`fpl/`, `plpred/fantasy.py`): each club's clean-sheet, two-plus-goals
  and win chances this gameweek, and a six-gameweek fixture strip from our ratings. The
  strip uses the FPL fixture list (`data/upcoming/fpl_fixtures.csv`, saved by the `fpl`
  update step); without it the page shows this week only.
- **Club pages** (`club/<name>/`): position, deserved points, season projection, xG made
  and allowed in every game, our record on their games and the fixtures ahead.
- **V the pundits** (`pundits/`, `plpred/pundits.py`): pundits' public predictions,
  typed into `data/pundits.csv`, scored beside ours and the bookmakers' favourite on the
  same games. Only add predictions published before kick-off, with a link to the source.
- **What we got wrong**: each matchweek page picks last round's most confident wrong
  call and says whether the chances (xG) backed us or the result.
- **Sharing**: every page has share-preview tags (the posting-kit card for the newest
  week's pages, a brand card otherwise), plus `sitemap.xml`, an Atom feed of the
  matchweeks (`feed.xml`) and `robots.txt`.

Two optional extras switch on from repository variables (Settings > Secrets and
variables > Actions > Variables), which the Publish site workflow passes to the build:

| Variable | What it does |
|---|---|
| `GOATCOUNTER` | Your [GoatCounter](https://www.goatcounter.com/) site code (the `xyz` in `xyz.goatcounter.com`). Adds its visitor counter: no cookies, so no cookie banner. |
| `NEWSLETTER_URL` | The newsletter's sign-up page (for example a Buttondown or Substack address). Adds a sign-up link to every footer. |

Each build also writes a posting kit for X and Instagram to `site/social/` (on the
live site at `/social/`, not linked from the front page). It has square images in the
site's dark look (the round, every call, the reasons chart for the bold call and the
scorecard, plus the FPL, projection and what-we-got-wrong cards) and the captions for
the week, with X posts already under 280 characters. It also writes the week's
newsletter as Markdown (`social/newsletter.md`) to paste into the newsletter service. The workflow renders the images to PNG with
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
