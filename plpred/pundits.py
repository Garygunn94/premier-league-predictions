"""Us against the pundits: well-known pundits' public predictions, scored beside ours.

Pundit picks are typed into data/pundits.csv, one row per prediction:

    season,matchweek,pundit,source,home_team,away_team,home_goals,away_goals
    2026,9,Chris Sutton,https://www.bbc.co.uk/sport/football/...,Arsenal,Chelsea,2,1

Only quote predictions that were published before kick-off, and link the source.
Each pick is scored once the result is in, against our call and the bookmakers'
favourite for the same game, so every comparison is on identical matches.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from plpred import teams

COLUMNS = ["season", "matchweek", "pundit", "source", "home_team", "away_team", "home_goals", "away_goals"]


def _result(h, a) -> str:
    return "H" if h > a else "A" if a > h else "D"


def load(data: Path) -> pd.DataFrame:
    path = data / "pundits.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    p = pd.read_csv(path).dropna(subset=["pundit", "home_team", "away_team"])
    for side in ("home_team", "away_team"):
        p[side] = p[side].map(teams.canonical)
    return p


def our_calls(data: Path, live: list[dict]) -> pd.DataFrame:
    """Our call for every game we have one for: the back-test replay, replaced by the
    published call wherever there is one."""
    frames = [pd.read_csv(f) for f in sorted((data / "backtest").glob("predictions_*.csv"))]
    calls = {}
    for f in frames:
        for r in f.itertuples():
            mk = np.array([r.mkt_home, r.mkt_draw, r.mkt_away], dtype=float)
            calls[(int(r.season), r.home_team, r.away_team)] = {
                "call": r.call, "score": r.score, "fav": None if np.isnan(mk).any() else "HDA"[int(mk.argmax())]}
    for preds in live:
        season = int(preds["season"][:4])
        for g in preds["games"]:
            key = (season, g["home_team"], g["away_team"])
            mkt = g.get("market")
            calls[key] = {"call": g["call"], "score": f"{g['score'][0]}-{g['score'][1]}",
                          "fav": max(mkt, key=mkt.get) if mkt else calls.get(key, {}).get("fav")}
    return calls


def scoreboard(data: Path, matches: pd.DataFrame, live: list[dict]) -> dict:
    picks = load(data)
    if picks.empty:
        return {"pundits": [], "latest": None}
    results = {(int(m.season), m.home_team, m.away_team): (int(m.home_goals), int(m.away_goals))
               for m in matches.itertuples()}
    calls = our_calls(data, live)
    rows = []
    for p in picks.itertuples():
        key = (int(p.season), p.home_team, p.away_team)
        ours = calls.get(key, {})
        res = results.get(key)
        pick = _result(p.home_goals, p.away_goals)
        rows.append({"season": key[0], "matchweek": int(p.matchweek), "pundit": p.pundit, "source": p.source,
                     "match": f"{p.home_team} v {p.away_team}",
                     "pick": f"{int(p.home_goals)}–{int(p.away_goals)}",
                     "ours": (ours.get("score") or "").replace("-", "–") or None,
                     "result": f"{res[0]}–{res[1]}" if res else None,
                     "pundit_right": res is not None and pick == _result(*res),
                     "pundit_exact": res is not None and (int(p.home_goals), int(p.away_goals)) == res,
                     "us_right": res is not None and ours.get("call") == _result(*res),
                     "us_exact": res is not None and ours.get("score") == f"{res[0]}-{res[1]}",
                     "fav_right": res is not None and ours.get("fav") == _result(*res),
                     "scored": res is not None and bool(ours)})
    df = pd.DataFrame(rows)
    season = int(df.season.max())
    board = []
    for name, g in df[(df.season == season) & df.scored].groupby("pundit"):
        board.append({"pundit": name, "n": len(g),
                      "right": int(g.pundit_right.sum()), "exact": int(g.pundit_exact.sum()),
                      "us_right": int(g.us_right.sum()), "us_exact": int(g.us_exact.sum()),
                      "fav_right": int(g.fav_right.sum())})
    board.sort(key=lambda b: -b["right"] / max(b["n"], 1))
    mw = int(df[df.season == season].matchweek.max())
    latest = df[(df.season == season) & (df.matchweek == mw)].to_dict("records")
    return {"pundits": board, "latest": {"matchweek": mw, "rows": latest}}
