"""The season's fixtures for a league without an FPL feed (La Liga), from two free sources.

Understat has every game of the season, played or not, with its kick-off (UTC) and, once
played, the score. It doesn't number the rounds, so the matchday comes from openfootball's
football.json (public domain, on GitHub), which lists each game under its matchday but is
updated by hand and can lag on scores and kick-off times. Either source alone still gives a
usable list: without openfootball the rounds are numbered from the dates, and without
Understat the kick-offs and scores come from openfootball.

A game moved well away from its matchday (a postponement, a cup clash) is counted in the round
being played when it kicks off, as FPL does, so every round is a block of days.
"""
import logging
import zlib

import pandas as pd

from plpred import teams
from plpred.http import session

log = logging.getLogger(__name__)

OPENFOOTBALL_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{start}-{end:02d}/{code}.json"
COLUMNS = ["fixture_id", "gameweek", "kickoff", "home_team", "away_team", "finished", "home_goals", "away_goals"]
# A game more than this far from its matchday's other games was moved, so it joins the round being
# played when it kicks off.
MOVED = pd.Timedelta(days=4)
# A game still not finished this long after its kick-off was postponed; it stops holding up its round.
POSTPONED = pd.Timedelta(days=2)


def fetch_openfootball(season: int, code: str) -> dict:
    resp = session().get(OPENFOOTBALL_URL.format(start=season, end=(season + 1) % 100, code=code), timeout=30)
    resp.raise_for_status()
    return resp.json()


def openfootball_frame(raw: dict) -> pd.DataFrame:
    rows = []
    for m in raw.get("matches", []):
        day = pd.to_numeric(str(m.get("round", "")).split()[-1], errors="coerce")
        when = pd.Timestamp(f"{m['date']} {m.get('time') or '00:00'}")
        # Times are Spanish local time; a game with no time yet is pinned to midnight.
        kickoff = when.tz_localize("Europe/Madrid", ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
        score = m.get("score")
        # A score is {"ht": [...], "ft": [...]}, or just the full-time [home, away] on some games.
        ft = score.get("ft") if isinstance(score, dict) else score if isinstance(score, list) and len(score) == 2 else None
        rows.append({"matchday": day, "kickoff": kickoff, "home_team": teams.canonical(m["team1"]),
                     "away_team": teams.canonical(m["team2"]), "finished": bool(ft),
                     "home_goals": ft[0] if ft else None, "away_goals": ft[1] if ft else None})
    return pd.DataFrame(rows, columns=["matchday", "kickoff", "home_team", "away_team", "finished",
                                       "home_goals", "away_goals"])


def understat_frame(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for m in raw:
        done = bool(m.get("isResult"))
        goals = m.get("goals") or {}
        rows.append({"understat_id": int(m["id"]), "kickoff": pd.Timestamp(m["datetime"], tz="UTC"),
                     "home_team": teams.canonical(m["h"]["title"]), "away_team": teams.canonical(m["a"]["title"]),
                     "finished": done,
                     "home_goals": int(goals["h"]) if done and goals.get("h") is not None else None,
                     "away_goals": int(goals["a"]) if done and goals.get("a") is not None else None})
    return pd.DataFrame(rows, columns=["understat_id", "kickoff", "home_team", "away_team", "finished",
                                       "home_goals", "away_goals"])


def combine(us: pd.DataFrame | None, of: pd.DataFrame | None) -> pd.DataFrame:
    """One row per game: Understat's kick-off and score where it has them, openfootball's matchday."""
    us = us if us is not None and len(us) else None
    of = of if of is not None and len(of) else None
    if us is None and of is None:
        return pd.DataFrame(columns=COLUMNS)
    if us is None:
        games = of.assign(understat_id=pd.NA, scheduled=of.kickoff)
    elif of is None:
        games = us.assign(matchday=float("nan"))
    else:
        games = us.merge(of, on=["home_team", "away_team"], how="outer", suffixes=("", "_of"))
        games["scheduled"] = games.kickoff_of  # openfootball's date, before any move
        missing = games.kickoff.isna()
        for col in ("kickoff", "home_goals", "away_goals"):
            games[col] = games[col].where(games[col].notna(), games[f"{col}_of"])
        games["finished"] = games.finished.where(~missing, games.finished_of).fillna(False).astype(bool)
        unmatched = games[games.understat_id.isna() | games.matchday.isna()]
        if len(unmatched):
            log.warning("%d games are in only one of Understat and openfootball (a name to add to teams.py?): %s",
                        len(unmatched), unmatched[["home_team", "away_team"]].head().values.tolist())
    games = games.dropna(subset=["kickoff"]).sort_values("kickoff").reset_index(drop=True)
    games["gameweek"] = rounds(games)
    games["fixture_id"] = [int(i) if pd.notna(i) else zlib.crc32(f"{h}|{a}".encode())
                           for i, h, a in zip(games.understat_id, games.home_team, games.away_team)]
    return games[COLUMNS]


def rounds(games: pd.DataFrame) -> pd.Series:
    """Each game's round: its matchday, unless it was moved well away from the matchday's other games,
    when it joins the round whose games are nearest in time. Without matchdays, numbered from the dates."""
    if games.matchday.isna().all():
        from plpred.backtest import gameweeks
        return gameweeks(games.kickoff.dt.tz_convert("Europe/Madrid").dt.tz_localize(None)).astype(int)
    # Each matchday's centre is from openfootball's dates, which a moved game doesn't drag away.
    when = games.scheduled.where(games.scheduled.notna(), games.kickoff) if "scheduled" in games else games.kickoff
    centre = games.assign(when=when).dropna(subset=["matchday"]).groupby("matchday").when.median()
    out = []
    for g in games.itertuples():
        if pd.notna(g.matchday) and abs(g.kickoff - centre[g.matchday]) <= MOVED:
            out.append(int(g.matchday))
        else:
            out.append(int((centre - g.kickoff).abs().idxmin()))
    return pd.Series(out, index=games.index)


def current_round(games: pd.DataFrame, now: pd.Timestamp) -> int | None:
    """The round being played, until all its games are done; then the next one."""
    open_ = games[~games.finished.astype(bool) & (games.kickoff > now - POSTPONED)]
    return int(open_.gameweek.min()) if len(open_) else None


def round_fixtures(games: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """The current round's games, in the same shape as the Premier League's fixtures.csv."""
    columns = ["fixture_id", "kickoff", "round", "status", "venue", "home_team", "away_team",
               "home_goals", "away_goals", "home_logo", "away_logo"]
    gw = current_round(games, now)
    if gw is None:
        return pd.DataFrame(columns=columns)
    r = games[games.gameweek == gw].copy()
    r["round"] = f"Regular Season - {gw}"
    r["status"] = ["FT" if done else "NS" if ko > now else "LIVE" for done, ko in zip(r.finished, r.kickoff)]
    r["venue"] = r["home_logo"] = r["away_logo"] = None
    return r[columns].reset_index(drop=True)
