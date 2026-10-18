"""Results versus performance: how far each team's results have run ahead of, or behind, its chances.

Expected points (xPts) are the points a match's chances were worth. Each side's xG is
treated as the average of its goals, and the chance of every scoreline gives the chance
of a win, draw or loss. Summed over a run of games, points minus xPts says whose results
flatter them and who has been unlucky. It describes the past; it is not a forecast.

The luck gap is the measure behind our pre-registered test (preregistered/luck-gap-2026-27.md):
each team's goals minus xG, at both ends, per game over its last 20 league games, home
side minus away side. Back-tests hinted that our model does best against the bookmakers
when that gap is large.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from plpred import explain

LUCK_GAP_GAMES = 20
LUCK_GAP_MIN_GAMES = 8
LUCK_GAP_THRESHOLD = 0.45   # goals per game; the top third of 2024/25 to 2026/27 back-test games
TEST_START = "2026-10-19"   # only live calls published after the test was written down count
RECENT_GAMES = 10
NOTABLE_POINTS = 3.0        # a gap this big over 10 games gets a reason on the match page

_GOALS = np.arange(11)


def match_xpts(home_xg: float, away_xg: float) -> tuple[float, float]:
    """(home, away) expected points from the two sides' xG."""
    probs = np.outer(poisson.pmf(_GOALS, home_xg), poisson.pmf(_GOALS, away_xg))
    p_home, p_draw, p_away = np.tril(probs, -1).sum(), np.trace(probs), np.triu(probs, 1).sum()
    total = p_home + p_draw + p_away
    return float((3 * p_home + p_draw) / total), float((3 * p_away + p_draw) / total)


def team_games(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match with xG, oldest first: points, xPts, goals and xG both ways."""
    m = matches.dropna(subset=["home_xg", "away_xg"])
    xp = [match_xpts(h, a) for h, a in zip(m.home_xg, m.away_xg)]
    sides = []
    for side, other, k in (("home", "away", 0), ("away", "home", 1)):
        gf, ga = m[f"{side}_goals"], m[f"{other}_goals"]
        sides.append(pd.DataFrame({
            "date": pd.to_datetime(m.date), "season": m.season, "team": m[f"{side}_team"],
            "points": np.select([gf > ga, gf == ga], [3, 1], 0), "xpts": [x[k] for x in xp],
            "gf": gf, "ga": ga, "xf": m[f"{side}_xg"], "xa": m[f"{other}_xg"]}))
    return pd.concat(sides).sort_values("date", kind="stable").reset_index(drop=True)


def recent(games: pd.DataFrame, team: str, n: int = RECENT_GAMES) -> dict | None:
    """Points against xPts over a team's last n games (at least half of them needed)."""
    g = games[games.team == team].tail(n)
    if len(g) < n // 2:
        return None
    points, xpts = int(g.points.sum()), float(g.xpts.sum())
    return {"games": len(g), "points": points, "xpts": round(xpts, 1), "difference": round(points - xpts, 1)}


def luck_gap(games: pd.DataFrame, home: str, away: str) -> float | None:
    """Home side's goals-minus-xG (scored minus conceded) per game, minus the away side's."""
    def per_game(team):
        g = games[games.team == team].tail(LUCK_GAP_GAMES)
        if len(g) < LUCK_GAP_MIN_GAMES:
            return None
        return float(((g.gf - g.xf) - (g.ga - g.xa)).mean())
    h, a = per_game(home), per_game(away)
    return None if h is None or a is None else round(h - a, 3)


def deserved_table(matches: pd.DataFrame, season: int) -> list[dict]:
    """This season's table re-ranked by expected points, with each side's actual place beside it."""
    g = team_games(matches[matches.season == season])
    if g.empty:
        return []
    t = g.groupby("team").agg(played=("points", "size"), points=("points", "sum"), xpts=("xpts", "sum"),
                              gd=("gf", "sum"), ga=("ga", "sum")).reset_index()
    t["gd"] = t.gd - t.ga
    t = t.sort_values(["points", "gd"], ascending=False)
    t["actual"] = np.arange(1, len(t) + 1)
    t = t.sort_values("xpts", ascending=False).reset_index(drop=True)
    t["position"] = t.index + 1
    t["difference"] = t.points - t.xpts
    return [{"position": int(r.position), "team": r.team, "played": int(r.played), "points": int(r.points),
             "xpts": round(float(r.xpts), 1), "difference": round(float(r.difference), 1),
             "actual": int(r.actual), "moved": int(r.actual) - int(r.position)}
            for r in t.itertuples()]


# ---------------------------------------------------------------- calibration

def _probs(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f"{prefix}_home", f"{prefix}_draw", f"{prefix}_away"]].to_numpy(dtype=float)


def _rps(probs: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    actual = np.eye(3)[outcome]
    return (((probs.cumsum(1) - actual.cumsum(1))[:, :2]) ** 2).sum(1) / 2


def power_margin(odds: np.ndarray) -> np.ndarray:
    """Bookmaker odds to probabilities by the power method: find k with sum((1/odds)^k) = 1.
    Plain scaling takes the margin evenly off every outcome, which overstates outsiders and
    makes the bookmakers look worse calibrated than they are."""
    inv = 1 / odds
    lo, hi = np.full(len(odds), 1.0), np.full(len(odds), 3.0)
    for _ in range(60):
        k = (lo + hi) / 2
        too_big = (inv ** k[:, None]).sum(1) > 1
        lo, hi = np.where(too_big, k, lo), np.where(too_big, hi, k)
    p = inv ** ((lo + hi) / 2)[:, None]
    return p / p.sum(1, keepdims=True)


def calibration(data: Path, matches: pd.DataFrame) -> dict | None:
    """How often our percentages came true, from every back-tested game on file."""
    files = sorted((data / "backtest").glob("predictions_*.csv"))
    if not files:
        return None
    p = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
    p = p.drop(columns=["odds_home", "odds_draw", "odds_away"], errors="ignore").merge(matches[["date", "home_team", "away_team", "odds_home", "odds_draw", "odds_away"]]
                .assign(date=lambda d: pd.to_datetime(d.date)), on=["date", "home_team", "away_team"], how="left")
    outcome = p.result.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    ours = _probs(p, "p")

    flat_p, flat_hit = ours.ravel(), (np.eye(3)[outcome] == 1).ravel()
    edges = [0, .1, .2, .3, .4, .5, .6, .7, 1.01]
    bands = []
    for lo, hi in zip(edges, edges[1:]):
        inside = (flat_p >= lo) & (flat_p < hi)
        if inside.sum() < 10:
            continue
        said, came = flat_p[inside].mean(), flat_hit[inside].mean()
        bands.append({"label": f"{lo:.0%} to {min(hi, 1):.0%}" if hi <= 1 else f"{lo:.0%} or more",
                      "n": int(inside.sum()), "said": round(said * 100), "came": round(came * 100),
                      "hits": int(flat_hit[inside].sum())})

    fav = ours.argmax(1)
    fav_p = ours.max(1)
    labels = np.array([explain.confidence(dict(zip("HDA", row))) for row in ours])
    fav_bands = []
    strong = f"{explain.STRONG:.0%}"
    for key, label in (("too close to call", "Too close to call"), ("solid", f"Solid (under {strong})"),
                       ("strong", f"Strong ({strong} or more)")):
        inside = labels == key
        if inside.any():
            fav_bands.append({"label": label, "n": int(inside.sum()), "said": round(fav_p[inside].mean() * 100),
                              "won": int((fav == outcome)[inside].sum()),
                              "came": round((fav == outcome)[inside].mean() * 100)})

    odds = p[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=float)
    has_odds = ~np.isnan(odds).any(1)
    books = power_margin(odds[has_odds])
    scores = {"n": int(has_odds.sum()),
              "us_rps": round(float(_rps(ours[has_odds], outcome[has_odds]).mean()), 4),
              "books_rps": round(float(_rps(books, outcome[has_odds]).mean()), 4),
              "us_rate": round(float((fav == outcome)[has_odds].mean() * 100)),
              "books_rate": round(float((books.argmax(1) == outcome[has_odds]).mean() * 100))}
    seasons = sorted(p.season.unique())
    return {"n": len(p), "bands": bands, "favourites": fav_bands, "scores": scores,
            "seasons": [f"{s}/{str(s + 1)[2:]}" for s in seasons]}


# ---------------------------------------------------------------- the pre-registered test

def preregistered(data: Path, matches: pd.DataFrame) -> dict:
    """Score the luck-gap test on live calls only, exactly as written in preregistered/."""
    calls = {}
    results = matches.assign(date=pd.to_datetime(matches.date))
    for f in sorted((data / "predictions").glob("*.json")):
        if f.name.startswith("replay-") or f.stem < TEST_START:
            continue
        preds = json.loads(f.read_text())
        for g in preds["games"]:
            gap = g.get("luck_gap")
            if gap is None:
                continue
            day = pd.Timestamp(g["kickoff"])
            if day.tzinfo is not None:
                day = day.tz_convert("Europe/London").tz_localize(None)
            day = day.normalize()
            r = results[(results.home_team == g["home_team"]) & (results.away_team == g["away_team"]) &
                        ((results.date - day).abs() <= pd.Timedelta(days=1))]
            if r.empty or r[["odds_home", "odds_draw", "odds_away"]].isna().any(axis=None):
                continue
            r = r.iloc[0]
            pr = g["probabilities"]
            # Files are read oldest first, so a later re-run of the same game replaces the earlier call.
            calls[(r.home_team, r.away_team, r.date)] = {
                "gap": abs(gap), "result": r.result, "p_home": pr["H"], "p_draw": pr["D"], "p_away": pr["A"],
                "odds": [r.odds_home, r.odds_draw, r.odds_away]}
    rows = list(calls.values())
    test = [r for r in rows if r["gap"] >= LUCK_GAP_THRESHOLD]
    out = {"start": pd.Timestamp(TEST_START).strftime("%d %B %Y"), "threshold": LUCK_GAP_THRESHOLD,
           "scored": len(rows), "n": len(test)}
    if test:
        df = pd.DataFrame(test)
        outcome = df.result.map({"H": 0, "D": 1, "A": 2}).to_numpy()
        ours = _probs(df, "p")
        odds = np.array(df.odds.tolist(), dtype=float)
        books = (1 / odds) / (1 / odds).sum(1, keepdims=True)   # plain scaling, as in the back-test
        out.update(us_rps=round(float(_rps(ours, outcome).mean()), 4),
                   books_rps=round(float(_rps(books, outcome).mean()), 4),
                   us_right=int((ours.argmax(1) == outcome).sum()),
                   books_right=int((books.argmax(1) == outcome).sum()))
    return out
