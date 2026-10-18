"""Player availability from the official Fantasy Premier League feed.

Gives each player's status (a = available, d = doubtful, i = injured, s = suspended,
u = unavailable, n = not in squad), injury news and chance of playing next round.
"""
import pandas as pd

from plpred import teams
from plpred.http import session

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def fetch_raw() -> dict:
    resp = session().get(URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def availability(raw: dict) -> pd.DataFrame:
    team_names = {t["id"]: teams.canonical(t["name"]) for t in raw["teams"]}
    rows = []
    for p in raw["elements"]:
        rows.append({
            "team": team_names[p["team"]],
            "player": p["web_name"],
            "position": POSITIONS.get(p["element_type"], "?"),
            "status": p["status"],
            "chance_next_round": p.get("chance_of_playing_next_round"),
            "news": p.get("news") or "",
            "minutes": p.get("minutes", 0),
            "starts": p.get("starts", 0),
            "goals": p.get("goals_scored", 0),
            "assists": p.get("assists", 0),
            "expected_goal_involvements": float(p.get("expected_goal_involvements") or 0),
        })
    return pd.DataFrame(rows)


def key_absences(players: pd.DataFrame, min_minutes_share: float = 0.5) -> pd.DataFrame:
    """Regular players who are out or doubtful, most important first."""
    team_max = players.groupby("team")["minutes"].transform("max").replace(0, 1)
    regular = players["minutes"] >= team_max * min_minutes_share
    out = players["status"].isin(["i", "s", "u", "n", "d"])
    return players[regular & out].sort_values(
        ["team", "expected_goal_involvements"], ascending=[True, False])
