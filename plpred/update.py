"""Fetch everything the model needs and save it under data/.

    python -m plpred.update              # all sources
    python -m plpred.update history xg   # just some of them

Each source runs on its own: if one fails (a site is down, no API key yet), the
others still update and the run reports what was skipped.
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from plpred import seasons
from plpred.sources import api_football, football_data_uk, fpl, understat

log = logging.getLogger("plpred.update")

DATA = Path(__file__).resolve().parent.parent / "data"


def update_history(data: Path) -> None:
    football_data_uk.download(data)
    matches = football_data_uk.build_matches(data)
    xg_files = [f for f in (data / "xg" / f"understat_{s}.csv" for s in seasons.window()) if f.exists()]
    if xg_files:
        xg = pd.concat([pd.read_csv(f) for f in xg_files], ignore_index=True)
        matches = matches.merge(xg[["season", "home_team", "away_team", "home_xg", "away_xg"]],
                                on=["season", "home_team", "away_team"], how="left")
        missing = matches[matches.home_xg.isna()]
        if len(missing):
            log.warning("%d matches have no xG (team name mismatch?): %s",
                        len(missing), missing[["home_team", "away_team"]].head().values.tolist())
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
        frame = understat.to_frame(understat.fetch_raw(season), season)
        frame.to_csv(path, index=False)
        log.info("xG %s: %d matches", seasons.label(season), len(frame))


def update_fpl(data: Path) -> None:
    out = data / "upcoming"
    out.mkdir(parents=True, exist_ok=True)
    players = fpl.availability(fpl.fetch_raw())
    players.to_csv(out / "fpl_players.csv", index=False)
    fpl.key_absences(players).to_csv(out / "key_absences.csv", index=False)
    log.info("FPL: %d players, %d key absences", len(players), len(fpl.key_absences(players)))


def update_fixtures(data: Path) -> None:
    out = data / "upcoming"
    out.mkdir(parents=True, exist_ok=True)
    client = api_football.Client()
    fixtures = api_football.upcoming_round(client, seasons.current_season())
    fixtures.to_csv(out / "fixtures.csv", index=False)
    badges = api_football.save_badges(fixtures, data / "badges")
    # Odds are fetched once per match: later runs reuse what we already have.
    odds_path = out / "odds.csv"
    known = pd.read_csv(odds_path) if odds_path.exists() else api_football.odds_frame([])
    known = known[known.fixture_id.isin(fixtures.fixture_id)]
    new_ids = [i for i in fixtures.fixture_id if i not in set(known.fixture_id)]
    odds = pd.concat([known, api_football.odds(client, new_ids)], ignore_index=True)
    odds.to_csv(odds_path, index=False)
    log.info("API-Football: %d fixtures (%s), odds fetched for %d new, %d new badges, %s requests left today",
             len(fixtures), fixtures["round"].iloc[0] if len(fixtures) else "no current round",
             len(new_ids), badges, client.remaining)


STEPS = {"xg": update_xg, "history": update_history, "fpl": update_fpl, "fixtures": update_fixtures}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("steps", nargs="*", metavar="step", help=f"any of: {', '.join(STEPS)}")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args(argv)
    unknown = set(args.steps) - set(STEPS)
    if unknown:
        parser.error(f"unknown step(s): {', '.join(sorted(unknown))}")
    steps = args.steps or list(STEPS)

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
