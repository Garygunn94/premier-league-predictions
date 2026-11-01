"""Fetch everything the model needs and save it under data/.

    python -m plpred.update              # all sources
    python -m plpred.update history xg   # just some of them
    python -m plpred.update --league la-liga   # La Liga, into data/laliga/ (no FPL step)

Each source runs on its own: if one fails (a site is down, no API key yet), the
others still update and the run reports what was skipped.
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

from plpred import leagues, seasons
from plpred.sources import api_football, fixture_list, football_data_uk, fpl, understat

log = logging.getLogger("plpred.update")

DATA = Path(__file__).resolve().parent.parent / "data"


def update_history(data: Path) -> None:
    div = leagues.of(data).fd_code
    football_data_uk.download(data, div=div)
    matches = football_data_uk.build_matches(data, div=div)
    xg_files = [f for f in (data / "xg" / f"understat_{s}.csv" for s in seasons.window()) if f.exists()]
    if xg_files:
        xg = pd.concat([pd.read_csv(f) for f in xg_files], ignore_index=True)
        matches = matches.merge(xg[["season", "home_team", "away_team", "home_xg", "away_xg"]],
                                on=["season", "home_team", "away_team"], how="left")
        missing = matches[matches.home_xg.isna()]
        if len(missing):
            log.warning("%d matches have no xG (team name mismatch?): %s",
                        len(missing), missing[["home_team", "away_team"]].head().values.tolist())
    for col in ("home_xg", "away_xg"):  # no xG yet: the model and pages fall back to goals alone
        if col not in matches:
            matches[col] = float("nan")
    matches.to_csv(data / "matches.csv", index=False)
    log.info("matches.csv: %d matches, %s to %s", len(matches),
             matches.date.min().date(), matches.date.max().date())


def update_xg(data: Path) -> None:
    wanted = seasons.window()
    current = wanted[-1]
    out = data / "xg"
    out.mkdir(parents=True, exist_ok=True)
    for season in wanted:
        path = out / f"understat_{season}.csv"
        if path.exists() and season < current:
            continue
        frame = understat.to_frame(understat.fetch_raw(season, leagues.of(data).understat), season)
        frame.to_csv(path, index=False)
        log.info("xG %s: %d matches", seasons.label(season), len(frame))


def update_fpl(data: Path) -> None:
    out = data / "upcoming"
    out.mkdir(parents=True, exist_ok=True)
    raw = fpl.fetch_raw()
    players = fpl.availability(raw)
    players.to_csv(out / "fpl_players.csv", index=False)
    fpl.fixtures(raw, fpl.fetch_fixtures_raw()).to_csv(out / "fpl_fixtures.csv", index=False)
    fpl.key_absences(players).to_csv(out / "key_absences.csv", index=False)
    log.info("FPL: %d players, %d key absences", len(players), len(fpl.key_absences(players)))


def update_fixtures(data: Path) -> None:
    out = data / "upcoming"
    out.mkdir(parents=True, exist_ok=True)
    if not leagues.of(data).fpl:
        return update_fixtures_listed(data)
    try:
        client = api_football.Client()
        fixtures = api_football.upcoming_round(client, seasons.current_season())
    except (api_football.MissingKey, RuntimeError, requests.RequestException) as exc:
        # API-Football's free plan doesn't cover the current season, so this is the usual path.
        log.warning("API-Football unavailable (%s), so using the FPL fixture list and football-data.co.uk odds", exc)
        return update_fixtures_free(data)
    fixtures.to_csv(out / "fixtures.csv", index=False)
    badges = api_football.save_badges(fixtures, data / "badges")
    # Odds are fetched once per match: later runs reuse what we already have.
    odds_path = out / "odds.csv"
    known = known_odds(odds_path, fixtures)
    new_ids = [i for i in fixtures.fixture_id if i not in set(known.fixture_id)]
    odds = pd.concat([known, api_football.odds(client, new_ids)], ignore_index=True)
    odds.to_csv(odds_path, index=False)
    log.info("API-Football: %d fixtures (%s), odds fetched for %d new, %d new badges, %s requests left today",
             len(fixtures), fixtures["round"].iloc[0] if len(fixtures) else "no current round",
             len(new_ids), badges, client.remaining)


def known_odds(path: Path, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Odds already saved for this round. Each match keeps the first prices we saw for it."""
    known = pd.read_csv(path) if path.exists() else api_football.odds_frame([])
    return known[known.fixture_id.isin(fixtures.fixture_id)]


def update_fixtures_free(data: Path) -> None:
    out = data / "upcoming"
    fixtures = fpl.round_fixtures(fpl.fetch_raw(), fpl.fetch_fixtures_raw())
    fixtures.to_csv(out / "fixtures.csv", index=False)
    badges = api_football.save_badges(fixtures, data / "badges")
    odds_path = out / "odds.csv"
    known = known_odds(odds_path, fixtures)
    new = fixtures[~fixtures.fixture_id.isin(known.fixture_id)]
    try:
        priced = new.merge(football_data_uk.upcoming_odds(), on=["home_team", "away_team"])
    except (requests.RequestException, ValueError, KeyError) as exc:
        # Without odds the calls still go out, just without the bookies' comparison.
        log.warning("No odds from football-data.co.uk: %s", exc)
        priced = new.iloc[:0]
    odds = pd.concat([known, priced.reindex(columns=["fixture_id", "odds_home", "odds_draw", "odds_away"])], ignore_index=True)
    odds.to_csv(odds_path, index=False)
    log.info("FPL: %d fixtures (%s), odds for %d new, %d new badges", len(fixtures),
             fixtures["round"].iloc[0] if len(fixtures) else "no current round", len(priced), badges)


def update_fixtures_listed(data: Path) -> None:
    """A league with no FPL feed (La Liga): the season's fixtures from Understat and openfootball,
    the current round from them, and football-data.co.uk's odds."""
    league = leagues.of(data)
    out = data / "upcoming"
    season = seasons.current_season()
    sources = {}
    for name, fetch, frame in (
            ("Understat", lambda: understat.fetch_raw(season, league.understat), fixture_list.understat_frame),
            ("openfootball", lambda: fixture_list.fetch_openfootball(season, league.openfootball),
             fixture_list.openfootball_frame)):
        try:
            sources[name] = frame(fetch())
        except (requests.RequestException, ValueError, KeyError) as exc:
            # One source alone is enough to go on with; the health check notices if both stop.
            log.warning("No fixtures from %s: %s", name, exc)
    if not sources:
        raise RuntimeError("Neither Understat nor openfootball gave a fixture list")
    games = fixture_list.combine(sources.get("Understat"), sources.get("openfootball"))
    games.to_csv(out / "season_fixtures.csv", index=False)
    fixtures = fixture_list.round_fixtures(games, pd.Timestamp.now(tz="UTC"))
    fixtures.to_csv(out / "fixtures.csv", index=False)
    odds_path = out / "odds.csv"
    known = known_odds(odds_path, fixtures)
    new = fixtures[~fixtures.fixture_id.isin(known.fixture_id)]
    try:
        priced = new.merge(football_data_uk.upcoming_odds(league.fd_code), on=["home_team", "away_team"])
    except (requests.RequestException, ValueError, KeyError) as exc:
        log.warning("No odds from football-data.co.uk: %s", exc)
        priced = new.iloc[:0]
    odds = pd.concat([known, priced.reindex(columns=["fixture_id", "odds_home", "odds_draw", "odds_away"])], ignore_index=True)
    odds.to_csv(odds_path, index=False)
    log.info("%s: %d fixtures this season from %s; %s, %d games, odds for %d new", league.name, len(games),
             " and ".join(sources), fixtures["round"].iloc[0] if len(fixtures) else "no current round",
             len(fixtures), len(priced))


STEPS = {"xg": update_xg, "history": update_history, "fpl": update_fpl, "fixtures": update_fixtures}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("steps", nargs="*", metavar="step", help=f"any of: {', '.join(STEPS)}")
    parser.add_argument("--data", type=Path, default=None,
                        help="the data folder (default: data/, or data/laliga/ for La Liga)")
    parser.add_argument("--league", choices=list(leagues.LEAGUES), default=None,
                        help="which league (default: the one the data folder holds, else the Premier League)")
    args = parser.parse_args(argv)
    league = leagues.get(args.league) if args.league else None
    if args.data is None:
        args.data = DATA / leagues.FOLDERS[(league or leagues.PREMIER_LEAGUE).key]
    if league:
        leagues.mark(args.data, league)
    unknown = set(args.steps) - set(STEPS)
    if unknown:
        parser.error(f"unknown step(s): {', '.join(sorted(unknown))}")
    # FPL is the Premier League's only: asked for La Liga, the step is skipped.
    steps = [s for s in (args.steps or list(STEPS)) if s != "fpl" or leagues.of(args.data).fpl]

    failed = []
    for name in STEPS:  # fixed order: xG before history so it can be merged in
        if name not in steps:
            continue
        try:
            STEPS[name](args.data)
        except api_football.MissingKey as exc:
            log.warning("Skipped %s: %s", name, exc)
        except Exception:
            log.exception("Step %s failed", name)
            failed.append(name)
    if failed:
        log.error("Failed steps: %s", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
