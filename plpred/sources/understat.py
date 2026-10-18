"""Per-match expected goals (xG) from Understat.

Understat has no official API. Its pages load match data from a JSON endpoint,
which answers once the session has the site's cookies.
"""
import logging

import pandas as pd

from plpred import teams
from plpred.http import session

log = logging.getLogger(__name__)

HOME_URL = "https://understat.com"
JSON_URL = "https://understat.com/getLeagueData/EPL/{season}"


def fetch_raw(season: int) -> list[dict]:
    s = session()
    s.get(HOME_URL, timeout=30)  # picks up the cookies the data endpoint expects
    resp = s.get(JSON_URL.format(season=season), timeout=30,
                 headers={"X-Requested-With": "XMLHttpRequest"})
    resp.raise_for_status()
    try:
        return resp.json()["dates"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Unexpected Understat response ({resp.headers.get('content-type')}): "
                         f"{resp.text[:200]!r}") from exc


def to_frame(raw: list[dict], season: int) -> pd.DataFrame:
    rows = []
    for m in raw:
        if not m.get("isResult"):
            continue
        rows.append({
            "season": season,
            "date": pd.Timestamp(m["datetime"]).normalize(),
            "home_team": teams.canonical(m["h"]["title"]),
            "away_team": teams.canonical(m["a"]["title"]),
            "home_xg": float(m["xG"]["h"]),
            "away_xg": float(m["xG"]["a"]),
            "understat_id": int(m["id"]),
        })
    return pd.DataFrame(rows)
