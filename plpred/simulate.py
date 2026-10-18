"""Play out the rest of the season thousands of times to project the final table.

Every remaining fixture is drawn from the same score chances the weekly calls use, so
each run gives a full final table. Counting over the runs gives each club's chance of
the title, the top four and relegation, its expected points and the range of places
it finishes in.

The remaining fixtures are every home-and-away pairing of this season's clubs that
hasn't been played yet, so no fixture list (and no API call) is needed.
"""
import numpy as np
import pandas as pd

from plpred import explain, model

RUNS = 10_000
SEED = 2026  # fixed, so a rebuild with the same data gives the same numbers
TOP = 4
RELEGATED = 3


def remaining_fixtures(matches: pd.DataFrame, season: int, teams: list[str]) -> list[tuple[str, str]]:
    m = matches[matches.season == season]
    played = set(zip(m.home_team, m.away_team))
    return [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]


def run(matches: pd.DataFrame, season: int, as_of, settings: model.Settings | None = None,
        runs: int = RUNS, seed: int = SEED) -> dict | None:
    """Projection for `season` using only matches before `as_of`."""
    as_of = pd.Timestamp(as_of)
    known = matches[matches.date < as_of]
    table = explain.table(known, season)
    if table.empty:
        return None
    teams = sorted(table.team)
    idx = {t: i for i, t in enumerate(teams)}
    fixtures = remaining_fixtures(known, season, teams)
    ratings = model.fit(known, as_of=as_of, settings=settings)
    rng = np.random.default_rng(seed)

    n = len(teams)
    current = table.set_index("team")
    points = np.tile(current.loc[teams, "points"].to_numpy(float), (runs, 1))
    gd = np.tile(current.loc[teams, "gd"].to_numpy(float), (runs, 1))
    gf = np.tile(current.loc[teams, "gf"].to_numpy(float), (runs, 1))
    size = model.MAX_GOALS + 1
    for home, away in fixtures:
        probs = model.score_matrix(*ratings.expected_goals(home, away), ratings.settings.rho).ravel()
        drawn = rng.choice(size * size, runs, p=probs / probs.sum())
        hg, ag = drawn // size, drawn % size
        h, a = idx[home], idx[away]
        points[:, h] += np.select([hg > ag, hg == ag], [3, 1], 0)
        points[:, a] += np.select([ag > hg, hg == ag], [3, 1], 0)
        gd[:, h] += hg - ag
        gd[:, a] += ag - hg
        gf[:, h] += hg
        gf[:, a] += ag

    # Points, then goal difference, then goals scored; a coin toss settles anything left.
    key = points * 1e6 + (gd + 500) * 1e3 + gf + rng.random((runs, n)) * 0.5
    order = np.argsort(-key, axis=1)
    place = np.empty_like(order)
    place[np.arange(runs)[:, None], order] = np.arange(1, n + 1)

    rows = []
    for t in teams:
        i = idx[t]
        p = place[:, i]
        rows.append({
            "team": t,
            "played": int(current.loc[t, "played"]), "points": int(current.loc[t, "points"]),
            "position": int(current.loc[t, "position"]),
            "xpts": round(float(points[:, i].mean()), 1),
            "title": float((p == 1).mean()), "top": float((p <= TOP).mean()),
            "relegation": float((p > n - RELEGATED).mean()),
            "low": int(np.percentile(p, 10)), "high": int(np.percentile(p, 90)),
            "likely": int(np.bincount(p).argmax()),
        })
    rows.sort(key=lambda r: (-r["xpts"], r["position"]))
    for k, r in enumerate(rows, 1):
        r["projected"] = k
    return {"teams": rows, "runs": runs, "fixtures_left": len(fixtures), "as_of": str(as_of.date())}


def with_changes(now: dict | None, before: dict | None) -> dict | None:
    """Add how much each club's chances moved since the previous projection."""
    if not now:
        return now
    old = {r["team"]: r for r in (before or {}).get("teams", [])}
    for r in now["teams"]:
        o = old.get(r["team"])
        for k in ("title", "top", "relegation"):
            r[f"{k}_change"] = None if o is None else round((r[k] - o[k]) * 100)
    return now


def headline_moves(sim: dict, n: int = 3) -> list[dict]:
    """The biggest swings in title, top-four or relegation chances since last week."""
    moves = []
    for r in sim["teams"]:
        for k, what in (("title", "title"), ("top", "top-four"), ("relegation", "relegation")):
            change = r.get(f"{k}_change")
            if change:
                moves.append({"team": r["team"], "what": what, "change": change, "now": round(r[k] * 100)})
    moves.sort(key=lambda m: -abs(m["change"]))
    return [m for m in moves if abs(m["change"]) >= 5][:n]
