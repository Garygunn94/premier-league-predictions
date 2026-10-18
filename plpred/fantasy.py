"""The FPL corner: clean-sheet and scoring chances per club, and a fixture-difficulty strip.

This week's numbers come from the same expected goals as the published calls (team news
included). The strip for the weeks after uses the plain team ratings, because team news
that far ahead isn't known. It needs data/upcoming/fpl_fixtures.csv, saved by
`python -m plpred.update fpl`; without it the page shows this week only.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from plpred import model

STRIP_WEEKS = 6
# Three-letter codes as the FPL app uses them; any other club gets its first three letters.
CODES = {"Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE", "Brighton": "BHA",
         "Burnley": "BUR", "Chelsea": "CHE", "Coventry": "COV", "Crystal Palace": "CRY", "Everton": "EVE",
         "Fulham": "FUL", "Hull": "HUL", "Ipswich": "IPS", "Leeds": "LEE", "Leicester": "LEI",
         "Liverpool": "LIV", "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW",
         "Nott'm Forest": "NFO", "Southampton": "SOU", "Sunderland": "SUN", "Tottenham": "TOT",
         "West Ham": "WHU", "Wolves": "WOL"}


def code(team: str) -> str:
    return CODES.get(team, team[:3].upper())


def chances(lam: float, mu: float, rho: float) -> dict:
    """Clean-sheet, two-plus-goals and win chances for both sides, from expected goals."""
    m = model.score_matrix(lam, mu, rho)
    return {
        "home": {"cs": float(m[:, 0].sum()), "two": float(m[2:, :].sum()), "win": float(np.tril(m, -1).sum())},
        "away": {"cs": float(m[0, :].sum()), "two": float(m[:, 2:].sum()), "win": float(np.triu(m, 1).sum())},
    }


def this_week(views: list[dict], rho: float) -> list[dict]:
    rows = []
    for v in views:
        lam, mu = v["expected_goals"]["home"], v["expected_goals"]["away"]
        c = chances(lam, mu, rho)
        for side, team, opp, xf, xa in (("home", v["home_team"], v["away_team"], lam, mu),
                                         ("away", v["away_team"], v["home_team"], mu, lam)):
            rows.append({"team": team, "opponent": opp, "venue": "H" if side == "home" else "A",
                         "xg_for": round(xf, 2), "xg_against": round(xa, 2),
                         "cs": round(c[side]["cs"] * 100), "two": round(c[side]["two"] * 100),
                         "win": round(c[side]["win"] * 100), "path": v["path"]})
    return rows


def load_fixtures(data: Path) -> pd.DataFrame | None:
    path = data / "upcoming" / "fpl_fixtures.csv"
    if not path.exists():
        return None
    f = pd.read_csv(path)
    return f.dropna(subset=["gameweek"]).astype({"gameweek": int})


def strip(fixtures: pd.DataFrame | None, ratings: model.Ratings, views: list[dict],
          weeks: int = STRIP_WEEKS) -> dict | None:
    """Each club's next `weeks` FPL gameweeks with its win and clean-sheet chances.
    Starts at the gameweek holding this week's games."""
    if fixtures is None or fixtures.empty or not views:
        return None
    ours = {(v["home_team"], v["away_team"]) for v in views}
    unplayed = fixtures[~fixtures.finished]
    now = unplayed[[(h, a) in ours for h, a in zip(unplayed.home_team, unplayed.away_team)]]
    if now.empty:
        return None
    first = int(now.gameweek.mode().iloc[0])
    gws = list(range(first, first + weeks))
    upcoming = unplayed[unplayed.gameweek.isin(gws)]
    teams = sorted(set(fixtures.home_team) | set(fixtures.away_team))
    cells = {t: {gw: [] for gw in gws} for t in teams}
    for f in upcoming.itertuples():
        c = chances(*ratings.expected_goals(f.home_team, f.away_team), ratings.settings.rho)
        for side, team, opp in (("home", f.home_team, f.away_team), ("away", f.away_team, f.home_team)):
            win = round(c[side]["win"] * 100)
            cells[team][f.gameweek].append({"opponent": opp, "code": code(opp), "venue": "H" if side == "home" else "A",
                                            "win": win, "cs": round(c[side]["cs"] * 100),
                                            "band": band(win)})
    rows = []
    for t in teams:
        games = [g for gw in gws for g in cells[t][gw]]
        rows.append({"team": t, "weeks": [cells[t][gw] for gw in gws],
                     "avg_win": round(np.mean([g["win"] for g in games])) if games else 0,
                     "avg_cs": round(np.mean([g["cs"] for g in games])) if games else 0,
                     "games": len(games)})
    rows.sort(key=lambda r: -r["avg_win"])
    return {"gameweeks": gws, "rows": rows}


def band(win: int) -> str:
    """Five shades, from a tough fixture (1) to a kind one (5)."""
    return str(1 + sum(win >= cut for cut in (25, 38, 50, 62)))
