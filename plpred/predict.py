"""Predict the coming round, with the reasons behind each call.

    python -m plpred.predict                       # uses data/upcoming/fixtures.csv
    python -m plpred.predict --replay 2026-10-04   # re-predict a past week, for checking
    python -m plpred.predict --after-kickoff       # this round's games that have already started
    python -m plpred.predict --if-due              # only if a round starts tomorrow and has no calls yet

Writes data/predictions/<date of first game>.json, one entry per match. --after-kickoff is for a
round first run late: it calls the games that have already kicked off from results before the
round began (no team news, which would be from after them) and writes
<date>-after-kickoff.json, with every game marked so the site never shows them as predictions.

The Thursday run calls the coming round. --if-due is for the other mornings: a round that starts
before the next Thursday (a midweek round, or one after a failed Thursday run) is called the
morning before its first game, in time for its predictions kit. A round with any calls already
is left alone, so frozen calls are never redone.
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from plpred import explain, health, model, seasons

log = logging.getLogger("plpred.predict")
DATA = Path(__file__).resolve().parent.parent / "data"


def load_settings(data: Path) -> model.Settings:
    path = data / "backtest" / "settings.json"
    return model.Settings(**json.loads(path.read_text())) if path.exists() else model.Settings()


NOT_STARTED = ["NS", "TBD", "PST"]


def upcoming(data: Path, started: bool = False) -> pd.DataFrame:
    """This round's games still to kick off, or with started=True the ones already under way or played."""
    path = data / "upcoming" / "fixtures.csv"
    if not path.exists():
        return pd.DataFrame()
    fixtures = pd.read_csv(path, parse_dates=["kickoff"])
    fixtures = fixtures[~fixtures.status.isin(NOT_STARTED + ["CANC", "ABD"]) if started
                        else fixtures.status.isin(NOT_STARTED)]
    odds_path = data / "upcoming" / "odds.csv"
    if odds_path.exists():
        fixtures = fixtures.merge(pd.read_csv(odds_path), on="fixture_id", how="left")
    return fixtures


def due(data: Path, today) -> bool:
    """Whether this round starts tomorrow (UK time) with no calls yet, and is the one in fixtures.csv."""
    games = health.round_starting(health.fixture_list(data), today + pd.Timedelta(days=1))
    if games.empty:
        log.info("No round starts tomorrow")
        return False
    gameweek = int(games.gameweek.iloc[0])
    if len(health.uncalled(games, data)) < len(games):
        log.info("Matchweek %d already has calls, so they stay as they are", gameweek)
        return False
    path = data / "upcoming" / "fixtures.csv"
    current = pd.read_csv(path)["round"] if path.exists() else pd.Series(dtype=str)
    if not (current == f"Regular Season - {gameweek}").any():
        log.warning("Matchweek %d starts tomorrow, but the fixtures fetched are for another round", gameweek)
        return False
    return True


def replay(matches: pd.DataFrame, week_start: str) -> pd.DataFrame:
    start = pd.Timestamp(week_start)
    week = matches[(matches.date >= start) & (matches.date < start + pd.Timedelta(days=7))]
    return week.rename(columns={"date": "kickoff"})[
        ["kickoff", "home_team", "away_team", "odds_home", "odds_draw", "odds_away"]]


def run(data: Path, fixtures: pd.DataFrame, matches: pd.DataFrame, players: pd.DataFrame | None,
        after_kickoff: bool = False) -> dict:
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
        if after_kickoff:
            g["after_kickoff"] = True
            if getattr(f, "status", None) in ("FT", "AET", "PEN"):
                g["result"] = [int(f.home_goals), int(f.away_goals)]
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
    parser.add_argument("--after-kickoff", action="store_true",
                        help="call this round's games that have already started, from results before the round")
    parser.add_argument("--if-due", action="store_true",
                        help="only if a round starts tomorrow and none of its games has a call yet")
    args = parser.parse_args(argv)
    if args.if_due and not due(args.data, health.uk_date(pd.Timestamp.now(tz="UTC"))):
        return

    matches = pd.read_csv(args.data / "matches.csv", parse_dates=["date"])
    if args.replay:
        fixtures, players = replay(matches, args.replay), None
    elif args.after_kickoff:
        fixtures, players = upcoming(args.data, started=True), None
    else:
        fixtures = upcoming(args.data)
        players_path = args.data / "upcoming" / "fpl_players.csv"
        players = pd.read_csv(players_path) if players_path.exists() else None
    if fixtures.empty:
        log.warning("No fixtures to predict")
        return

    result = run(args.data, fixtures, matches, players, after_kickoff=args.after_kickoff)
    out = args.data / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{'replay-' if args.replay else ''}{result['as_of']}{'-after-kickoff' if args.after_kickoff else ''}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    for g in result["games"]:
        p = g["probabilities"]
        log.info("%-15s v %-15s %s %d-%d  (H %.0f%% D %.0f%% A %.0f%%, %s)", g["home_team"], g["away_team"],
                 g["call"], *g["score"], p["H"] * 100, p["D"] * 100, p["A"] * 100, g["confidence"])
    log.info("Wrote %s", path)


if __name__ == "__main__":
    main()
