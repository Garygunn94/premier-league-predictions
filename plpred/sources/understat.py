"""Per-match expected goals (xG) from Understat.

Understat has no official API. Its pages load match data from a JSON endpoint,
which answers once the session has the site's cookies. When that endpoint answers with a
web page instead (as it did in November 2026), the data is read from the league page, which
has at times carried it inline as `datesData = JSON.parse('...')`.
"""
import codecs
import json
import logging
import re

import pandas as pd

from plpred import teams
from plpred.http import session

log = logging.getLogger(__name__)

HOME_URL = "https://understat.com"
JSON_URL = "https://understat.com/getLeagueData/{league}/{season}"
PAGE_URL = "https://understat.com/league/{league}/{season}"
INLINE = re.compile(r"datesData\s*=\s*JSON\.parse\('(.*?)'\)", re.S)


def from_page(html: str) -> list[dict] | None:
    """The games embedded in a league page, or None if the page doesn't carry them."""
    m = INLINE.search(html)
    if not m:
        return None
    return json.loads(codecs.decode(m.group(1), "unicode_escape"))


def fetch_raw(season: int, league: str = "EPL") -> list[dict]:
    """Every game of the season, played or not (La Liga is 'La_liga')."""
    s = session()
    s.get(HOME_URL, timeout=30)  # picks up the cookies the data endpoint expects
    page_url = PAGE_URL.format(league=league, season=season)
    resp = s.get(JSON_URL.format(league=league, season=season), timeout=30,
                 headers={"X-Requested-With": "XMLHttpRequest", "Referer": page_url,
                          "Accept": "application/json, text/javascript, */*; q=0.01"})
    resp.raise_for_status()
    try:
        return resp.json()["dates"]
    except (ValueError, KeyError, TypeError) as exc:
        page = s.get(page_url, timeout=30)
        games = from_page(page.text) if page.ok else None
        if games is not None:
            log.warning("Understat's data endpoint sent a web page, so read the league page instead")
            return games
        # What the page does load, so the next fix doesn't start blind.
        hints = sorted(set(re.findall(r"""(?:src=|fetch\(|url:\s*|get\()["']([^"']+)["']""", page.text)))[:15]
        raise ValueError(f"Unexpected Understat response ({resp.status_code}, {resp.headers.get('content-type')}, "
                         f"{len(resp.text)} chars, title {_title(resp.text)!r}); league page {page.status_code}, "
                         f"{len(page.text)} chars, loads {hints}") from exc


def _title(html: str) -> str:
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    return m.group(1).strip() if m else ""


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
