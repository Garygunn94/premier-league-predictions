"""Upcoming fixtures and pre-match odds from API-Football (api-sports.io).

Needs an API key in the API_FOOTBALL_KEY environment variable. Calls are kept to a
minimum: 2 per run for the round's fixtures, plus 1 per match for odds the first
time we see it (so about 12 a week). Injuries come from the free FPL feed instead.
"""
import logging
import os
import re
from pathlib import Path

import pandas as pd

from plpred import teams
from plpred.http import session

log = logging.getLogger(__name__)

BASE = "https://v3.football.api-sports.io"
PREMIER_LEAGUE = 39


class MissingKey(RuntimeError):
    pass


class Client:
    def __init__(self, key: str | None = None):
        key = key or os.environ.get("API_FOOTBALL_KEY")
        if not key:
            raise MissingKey("Set API_FOOTBALL_KEY to fetch fixtures and odds")
        self.s = session()
        self.s.headers["x-apisports-key"] = key
        self.remaining = None

    def get(self, path: str, **params) -> list[dict]:
        results, page = [], 1
        while True:
            resp = self.s.get(f"{BASE}/{path}", params={**params, **({"page": page} if page > 1 else {})}, timeout=30)
            resp.raise_for_status()
            self.remaining = resp.headers.get("x-ratelimit-requests-remaining")
            body = resp.json()
            if body.get("errors"):
                raise RuntimeError(f"API-Football {path}: {body['errors']}")
            results += body["response"]
            paging = body.get("paging") or {}
            if paging.get("current", 1) >= paging.get("total", 1):
                return results
            page += 1


def fixtures_frame(raw: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "fixture_id": f["fixture"]["id"],
        "kickoff": pd.Timestamp(f["fixture"]["date"]),
        "round": f["league"]["round"],
        "status": f["fixture"]["status"]["short"],
        "venue": (f["fixture"].get("venue") or {}).get("name"),
        "home_team": teams.canonical(f["teams"]["home"]["name"]),
        "away_team": teams.canonical(f["teams"]["away"]["name"]),
        "home_goals": f["goals"]["home"],
        "away_goals": f["goals"]["away"],
        "home_logo": f["teams"]["home"].get("logo"),
        "away_logo": f["teams"]["away"].get("logo"),
    } for f in raw])


def save_badges(fixtures: pd.DataFrame, folder: Path) -> int:
    """Download each club's crest once. Images come from API-Football's media server,
    which does not count towards the daily request limit."""
    folder.mkdir(parents=True, exist_ok=True)
    s = session()
    saved = 0
    for side in ("home", "away"):
        for team, url in zip(fixtures[f"{side}_team"], fixtures.get(f"{side}_logo", [])):
            path = folder / f"{badge_name(team)}.png"
            if path.exists() or not isinstance(url, str) or not url:
                continue
            resp = s.get(url, timeout=30)
            if resp.ok and resp.headers.get("content-type", "").startswith("image/"):
                path.write_bytes(resp.content)
                saved += 1
    return saved


def badge_name(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", team.lower()).strip("-")


def odds_frame(raw: list[dict]) -> pd.DataFrame:
    """Average 1X2 price across bookmakers for each fixture."""
    rows = []
    for item in raw:
        for book in item["bookmakers"]:
            for bet in book["bets"]:
                if bet["name"] != "Match Winner":
                    continue
                prices = {v["value"]: float(v["odd"]) for v in bet["values"]}
                rows.append({"fixture_id": item["fixture"]["id"], "bookmaker": book["name"],
                             "odds_home": prices.get("Home"), "odds_draw": prices.get("Draw"),
                             "odds_away": prices.get("Away")})
    df = pd.DataFrame(rows, columns=["fixture_id", "bookmaker", "odds_home", "odds_draw", "odds_away"])
    return df.groupby("fixture_id", as_index=False)[["odds_home", "odds_draw", "odds_away"]].mean()


def upcoming_round(client: Client, season: int) -> pd.DataFrame:
    current = client.get("fixtures/rounds", league=PREMIER_LEAGUE, season=season, current="true")
    if not current:
        return fixtures_frame([])
    return fixtures_frame(client.get("fixtures", league=PREMIER_LEAGUE, season=season, round=current[0]))


def odds(client: Client, fixture_ids: list[int]) -> pd.DataFrame:
    raw = []
    for fid in fixture_ids:
        raw += client.get("odds", fixture=fid)
    return odds_frame(raw)
