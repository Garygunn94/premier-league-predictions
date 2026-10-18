"""Check the data is still arriving and the week's calls were made, so a quiet failure gets noticed.

    python -m plpred.health                    # exit 1 and list the problems, if there are any
    python -m plpred.health --calls            # also require calls for the next round, if it starts within 8 days (Thursday)
    python -m plpred.health --report health.md # and write the problems to a file, for the alert issue

The Update data workflow runs this after every refresh. A run that crashes is caught by the
workflow itself; this catches the runs that succeed but bring in nothing new, which is what
happens when a source changes its format or stops updating.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger("plpred.health")
DATA = Path(__file__).resolve().parent.parent / "data"

# FPL marks a game finished within a couple of hours. football-data.co.uk updates about twice a
# week (Sunday and Wednesday nights), so it gets a wider margin before we call it stuck.
FPL_LAG = pd.Timedelta(hours=36)
RESULTS_LAG = pd.Timedelta(days=6)
CALLS_AHEAD = pd.Timedelta(days=8)


def fixture_list(data: Path) -> pd.DataFrame:
    path = data / "upcoming" / "fpl_fixtures.csv"
    if not path.exists():
        return pd.DataFrame(columns=["kickoff", "home_team", "away_team", "finished"])
    f = pd.read_csv(path)
    f["kickoff"] = pd.to_datetime(f.kickoff, utc=True, format="ISO8601")
    return f


def called_games(data: Path) -> set[tuple[str, str, str]]:
    """(home, away, kickoff date) for every game in a real calls file."""
    games = set()
    for p in (data / "predictions").glob("*.json"):
        if p.name.startswith("replay-"):
            continue
        for g in json.loads(p.read_text())["games"]:
            games.add((g["home_team"], g["away_team"], str(pd.Timestamp(g["kickoff"]).date())))
    return games


def problems(data: Path, now: pd.Timestamp, calls: bool = False) -> list[str]:
    out = []
    fixtures = fixture_list(data)
    if fixtures.empty:
        return ["There is no fixture list (data/upcoming/fpl_fixtures.csv), so the FPL fetch isn't working."]

    stuck = fixtures[(fixtures.kickoff < now - FPL_LAG) & ~fixtures.finished.astype(bool)]
    if len(stuck):
        out.append(f"FPL still hasn't marked {len(stuck)} games as finished, the oldest "
                   f"{stuck.iloc[0].home_team} v {stuck.iloc[0].away_team} on {stuck.kickoff.min():%d %B}. "
                   "The FPL fetch may be failing or its format may have changed.")

    matches = pd.read_csv(data / "matches.csv", parse_dates=["date"])
    have = set(zip(matches.home_team, matches.away_team, matches.date.dt.date))
    due = fixtures[(fixtures.kickoff < now - RESULTS_LAG) & fixtures.finished.astype(bool)]
    missing = [r for r in due.itertuples() if (r.home_team, r.away_team, r.kickoff.date()) not in have]
    if missing:
        out.append(f"football-data.co.uk is {len(missing)} results behind, the oldest "
                   f"{missing[0].home_team} v {missing[0].away_team} on {missing[0].kickoff:%d %B}. "
                   "Results, expected goals and our record will be out of date until it catches up.")

    if calls:
        # The next round only: a midweek round after it is called by hand, nearer the time.
        ahead = fixtures[(fixtures.kickoff > now) & (fixtures.kickoff < now + CALLS_AHEAD)]
        ahead = ahead[ahead.gameweek == ahead.gameweek.min()]
        done = called_games(data)
        uncalled = [r for r in ahead.itertuples() if (r.home_team, r.away_team, str(r.kickoff.date())) not in done]
        if uncalled:
            out.append(f"{len(uncalled)} of the {len(ahead)} games in the next round have no call, starting with "
                       f"{uncalled[0].home_team} v {uncalled[0].away_team} on {uncalled[0].kickoff:%d %B}. "
                       "Run Update data by hand with 'predict' ticked once the cause is fixed.")
    return out


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--calls", action="store_true", help="also require calls for the next round, if it starts within 8 days")
    parser.add_argument("--report", type=Path, help="write the problems here as a Markdown list")
    args = parser.parse_args(argv)

    found = problems(args.data, pd.Timestamp.now(tz="UTC"), args.calls)
    if args.report:
        args.report.write_text("".join(f"- {p}\n" for p in found))
    if not found:
        log.info("Data looks healthy")
        return
    for p in found:
        log.error(p)
    sys.exit(1)


if __name__ == "__main__":
    main()
