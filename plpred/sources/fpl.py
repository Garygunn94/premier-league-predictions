"""Player availability from the official Fantasy Premier League feed.

Gives each player's status (a = available, d = doubtful, i = injured, s = suspended,
u = unavailable, n = not in squad), injury news and chance of playing next round,
and the season's fixture list by FPL gameweek (for the fixture strip on the FPL page).
It is also the free fallback for this round's fixtures and club crests when API-Football
can't serve the current season (its free plan only covers past seasons).
"""
import pandas as pd

from plpred import teams
from plpred.http import session

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
BADGE_URL = "https://resources.premierleague.com/premierleague/badges/100/t{code}.png"
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def fetch_raw() -> dict:
    resp = session().get(URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_fixtures_raw() -> list:
    resp = session().get(FIXTURES_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fixtures(raw: dict, raw_fixtures: list) -> pd.DataFrame:
    """Every fixture this season: FPL gameweek (blank if not scheduled yet), kick-off and the two clubs."""
    team_names = {t["id"]: teams.canonical(t["name"]) for t in raw["teams"]}
    rows = [{"gameweek": f.get("event"), "kickoff": f.get("kickoff_time"),
             "home_team": team_names[f["team_h"]], "away_team": team_names[f["team_a"]],
             "finished": bool(f.get("finished"))} for f in raw_fixtures]
    return pd.DataFrame(rows, columns=["gameweek", "kickoff", "home_team", "away_team", "finished"])


def current_gameweek(raw: dict) -> int | None:
    """The gameweek being played, until all its games are done; then the next one."""
    events = raw.get("events", [])
    event = next((e for e in events if e.get("is_current")), None)
    if event is None or event.get("finished"):
        event = next((e for e in events if e.get("is_next")), event)
    return event["id"] if event else None


def round_fixtures(raw: dict, raw_fixtures: list) -> pd.DataFrame:
    """This round's games in the same shape as api_football.fixtures_frame."""
    gameweek = current_gameweek(raw)
    clubs = {t["id"]: t for t in raw["teams"]}
    rows = []
    for f in raw_fixtures:
        if f.get("event") != gameweek or gameweek is None:
            continue
        home, away = clubs[f["team_h"]], clubs[f["team_a"]]
        done = f.get("finished") or f.get("finished_provisional")
        rows.append({
            "fixture_id": f["id"], "kickoff": pd.Timestamp(f["kickoff_time"]),
            "round": f"Regular Season - {gameweek}",
            "status": "FT" if done else "LIVE" if f.get("started") else "NS", "venue": None,
            "home_team": teams.canonical(home["name"]), "away_team": teams.canonical(away["name"]),
            "home_goals": f.get("team_h_score"), "away_goals": f.get("team_a_score"),
            "home_logo": BADGE_URL.format(code=home["code"]) if home.get("code") else None,
            "away_logo": BADGE_URL.format(code=away["code"]) if away.get("code") else None,
        })
    columns = ["fixture_id", "kickoff", "round", "status", "venue", "home_team", "away_team",
               "home_goals", "away_goals", "home_logo", "away_logo"]
    return pd.DataFrame(rows, columns=columns)


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
