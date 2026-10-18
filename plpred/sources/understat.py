"""Per-match expected goals (xG) from Understat, Premier League from 2014/15.

Understat has no official API. Its league page loads a JSON list of matches; we try
the JSON endpoint first and fall back to the data embedded in the page's HTML.
"""
import codecs
import json
import logging
import re

import pandas as pd

from plpred import teams
from plpred.http import session

log = logging.getLogger(__name__)

FIRST_SEASON = 2014
JSON_URL = "https://understat.com/getLeagueData/EPL/{season}"
PAGE_URL = "https://understat.com/league/EPL/{season}"
EMBEDDED = re.compile(r"datesData\s*=\s*JSON\.parse\('(?P<data>.*?)'\)")


def fetch_raw(season: int) -> list[dict]:
    s = session()
    try:
        resp = s.get(JSON_URL.format(season=season), timeout=30,
                     headers={"X-Requested-With": "XMLHttpRequest"})
        if resp.ok and resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()["dates"]
    except (ValueError, KeyError) as exc:
        log.info("Understat JSON endpoint unusable (%s), falling back to page", exc)
    resp = s.get(PAGE_URL.format(season=season), timeout=30)
    resp.raise_for_status()
    return parse_page(resp.text)


def parse_page(html: str) -> list[dict]:
    m = EMBEDDED.search(html)
    if not m:
        raise ValueError("Understat page layout changed: no datesData found")
    return json.loads(codecs.decode(m.group("data"), "unicode_escape"))


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
