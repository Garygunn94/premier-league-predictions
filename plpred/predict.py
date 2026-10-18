"""Predict the coming round, with the reasons behind each call.

    python -m plpred.predict                       # uses data/upcoming/fixtures.csv
    python -m plpred.predict --replay 2026-10-04   # re-predict a past week, for checking

Writes data/predictions/<date of first game>.json, one entry per match.
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from plpred import explain, model, seasons

log = logging.getLogger("plpred.predict")
DATA = Path(__file__).resolve().parent.parent / "data"


def load_settings(data: Path) -> model.Settings:
    path = data / "backtest" / "settings.json"
    return model.Settings(**json.loads(path.read_text())) if path.exists() else model.Settings()


def upcoming(data: Path) -> pd.DataFrame:
    path = data / "upcoming" / "fixtures.csv"
    if not path.exists():
        return pd.DataFrame()
    fixtures = pd.read_csv(path, parse_dates=["kickoff"])
    fixtures = fixtures[fixtures.status.isin(["NS", "TBD", "PST"])]
    odds_path = data / "upcoming" / "odds.csv"
    if odds_path.exists():
        fixtures = fixtures.merge(pd.read_csv(odds_path), on="fixture_id", how="left")
    return fixtures


def replay(matches: pd.DataFrame, week_start: str) -> pd.DataFrame:
    start = pd.Timestamp(week_start)
    week = matches[(matches.date >= start) & (matches.date < start + pd.Timedelta(days=7))]
    return week.rename(columns={"date": "kickoff"})[
        ["kickoff", "home_team", "away_team", "odds_home", "odds_draw", "odds_away"]]


def run(data: Path, fixtures: pd.DataFrame, matches: pd.DataFrame, players: pd.DataFrame | None) -> dict:
    as_of = pd.Timestamp(fixtures.kickoff.min()).tz_localize(None).normalize()
    known = matches[matches.date < as_of]
    ratings = model.fit(known, as_of=as_of, settings=load_settings(data))
    season = seasons.current_season(as_of.date())
    games = []
    for f in fixtures.sort_values("kickoff").itertuples():
        odds = tuple(getattr(f, c, None) for c in ("odds_home", "odds_draw", "odds_away"))
        g = explain.explain(ratings, known, f.home_team, f.away_team, players, odds, season)
        g["kickoff"] = pd.Timestamp(f.kickoff).isoformat()
        g["venue"] = getattr(f, "venue", None) if isinstance(getattr(f, "venue", None), str) else None
        games.append(g)
    return {
        "as_of": str(as_of.date()),
        "season": seasons.label(season),
        "round": str(fixtures["round"].iloc[0]) if "round" in fixtures else None,
        "settings": vars(ratings.settings),
        "games": games,
    }


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--replay", metavar="YYYY-MM-DD",
                        help="predict the 7 days from this date using only earlier results (no team news)")
    args = parser.parse_args(argv)

    matches = pd.read_csv(args.data / "matches.csv", parse_dates=["date"])
    if args.replay:
        fixtures, players = replay(matches, args.replay), None
    else:
        fixtures = upcoming(args.data)
        players_path = args.data / "upcoming" / "fpl_players.csv"
        players = pd.read_csv(players_path) if players_path.exists() else None
    if fixtures.empty:
        log.warning("No fixtures to predict")
        return

    result = run(args.data, fixtures, matches, players)
    out = args.data / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{'replay-' if args.replay else ''}{result['as_of']}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    for g in result["games"]:
        p = g["probabilities"]
        log.info("%-15s v %-15s %s %d-%d  (H %.0f%% D %.0f%% A %.0f%%, %s)", g["home_team"], g["away_team"],
                 g["call"], *g["score"], p["H"] * 100, p["D"] * 100, p["A"] * 100, g["confidence"])
    log.info("Wrote %s", path)


if __name__ == "__main__":
    main()
