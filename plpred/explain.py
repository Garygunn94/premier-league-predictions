"""Why the model makes each call.

Every prediction is broken into named factors (each side's attack and defence
ratings, and team news), and each factor's share of the final probability is
worked out exactly, so the steps add up from a neutral starting point to our number.
Alongside that sit the plain facts a reader wants: recent xG, form, head to head,
key absences and how we compare with the bookmakers.
"""
import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from plpred import luck, model

OUTCOME_NAMES = {"H": "home win", "D": "draw", "A": "away win"}

# Team news: how much of a team's attack or defence a missing regular takes with them.
# These are judgement calls, not fitted: we have no history of who was injured when.
ATTACK_DAMPING = 0.5      # losing 20% of a team's goal involvements cuts its expected goals by 10%
DEFENCE_DAMPING = 0.15    # losing 20% of its defensive minutes adds 3% to goals conceded
MAX_NEWS_EFFECT = 0.25
OUT_STATUSES = {"i": "Injured", "s": "Suspended", "u": "Unavailable", "n": "Not in squad", "d": "Doubtful"}


@dataclass
class Factor:
    label: str
    detail: str
    team: str = ""
    log_home: float = 0.0   # added to log(home expected goals)
    log_away: float = 0.0   # added to log(away expected goals)
    kind: str = ""          # "attack", "defence" or "news", for wording


# ---------------------------------------------------------------- team news

def team_news(players: pd.DataFrame, team: str) -> tuple[float, float, list[dict]]:
    """(attack multiplier, conceded multiplier, rows for the table) for one team."""
    squad = players[players.team == team]
    if squad.empty:
        return 1.0, 1.0, []
    xgi_total = squad.expected_goal_involvements.sum() or 1.0
    defenders = squad[squad.position.isin(["GK", "DEF"])]
    def_minutes = defenders.minutes.sum() or 1.0
    regular = squad.minutes >= squad.minutes.max() * 0.4
    missing = squad[regular & squad.status.isin(OUT_STATUSES)]

    attack_lost = defence_lost = 0.0
    rows = []
    for p in missing.itertuples():
        absent = 1.0 if p.status != "d" else 1 - _chance(p.chance_next_round)
        a = absent * p.expected_goal_involvements / xgi_total
        d = absent * p.minutes / def_minutes if p.position in ("GK", "DEF") else 0.0
        attack_lost += a
        defence_lost += d
        share = max(a, d)
        rows.append({
            "team": team, "player": p.player, "position": p.position,
            "status": OUT_STATUSES[p.status] + (f", {int(_chance(p.chance_next_round) * 100)}% to play"
                                                if p.status == "d" else ""),
            "news": p.news,
            "impact": "Big" if share >= 0.15 else "Medium" if share >= 0.07 else "Small",
            "share": round(float(share), 3),
        })
    rows.sort(key=lambda r: -r["share"])
    attack = 1 - min(MAX_NEWS_EFFECT, ATTACK_DAMPING * attack_lost)
    conceded = 1 + min(MAX_NEWS_EFFECT, DEFENCE_DAMPING * defence_lost)
    return attack, conceded, rows


def _chance(value) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.5
    return float(value) / 100


# ---------------------------------------------------------------- factors

def factors(ratings: model.Ratings, home: str, away: str,
            players: pd.DataFrame | None) -> tuple[list[Factor], list[dict]]:
    h_att, h_def = ratings.rating(home)
    a_att, a_def = ratings.rating(away)
    out = [
        Factor(f"{_pos(home)} attack", _strength_text(home, h_att, "attack"), home, log_home=h_att, kind="attack"),
        Factor(f"{_pos(away)} defence", _strength_text(away, a_def, "defence"), away, log_home=a_def, kind="defence"),
        Factor(f"{_pos(away)} attack", _strength_text(away, a_att, "attack"), away, log_away=a_att, kind="attack"),
        Factor(f"{_pos(home)} defence", _strength_text(home, h_def, "defence"), home, log_away=h_def, kind="defence"),
    ]
    news_rows = []
    if players is not None:
        for team, side in ((home, "home"), (away, "away")):
            attack, conceded, rows = team_news(players, team)
            news_rows += rows
            if attack == 1 and conceded == 1:
                continue
            names = ", ".join(r["player"] for r in rows[:3])
            f = Factor(f"{team} team news", f"Missing or doubtful: {names}", team, kind="news")
            if side == "home":
                f.log_home, f.log_away = math.log(attack), math.log(conceded)
            else:
                f.log_away, f.log_home = math.log(attack), math.log(conceded)
            out.append(f)
    return out, news_rows


def _strength_text(team: str, value: float, what: str) -> str:
    pct = round((math.exp(value) - 1) * 100)
    if what == "attack":
        return f"{team} create and score {abs(pct)}% {'more' if pct >= 0 else 'less'} than an average side"
    return f"{team} concede {abs(pct)}% {'more' if pct >= 0 else 'less'} than an average side"


def probabilities(lam: float, mu: float, rho: float) -> dict:
    probs = model.score_matrix(lam, mu, rho)
    return {"H": float(np.tril(probs, -1).sum()), "D": float(np.trace(probs)), "A": float(np.triu(probs, 1).sum())}


def contributions(base_lam: float, base_mu: float, rho: float, fs: list[Factor], outcome: str) -> list[float]:
    """Each factor's exact share of P(outcome), averaged over every order the factors
    could be added in (Shapley values), so the shares always add up to the total."""
    n = len(fs)
    cache = {}

    def value(subset):
        if subset not in cache:
            lh = sum(fs[i].log_home for i in subset)
            la = sum(fs[i].log_away for i in subset)
            cache[subset] = probabilities(base_lam * math.exp(lh), base_mu * math.exp(la), rho)[outcome]
        return cache[subset]

    shares = [0.0] * n
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k in range(n):
            weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for combo in itertools.combinations(others, k):
                with_i = tuple(sorted(combo + (i,)))
                shares[i] += weight * (value(with_i) - value(tuple(combo)))
    return shares


# ---------------------------------------------------------------- facts

def last_games(matches: pd.DataFrame, team: str, n: int = 5, venue: str | None = None) -> pd.DataFrame:
    m = matches
    if venue == "home":
        m = m[m.home_team == team]
    elif venue == "away":
        m = m[m.away_team == team]
    else:
        m = m[(m.home_team == team) | (m.away_team == team)]
    return m.sort_values("date").tail(n)


def form(matches: pd.DataFrame, team: str, n: int = 5) -> list[dict]:
    out = []
    for g in last_games(matches, team, n).itertuples():
        home = g.home_team == team
        gf, ga = (g.home_goals, g.away_goals) if home else (g.away_goals, g.home_goals)
        out.append({"result": "W" if gf > ga else "L" if gf < ga else "D",
                    "line": f"{g.home_team} {g.home_goals}-{g.away_goals} {g.away_team}",
                    "date": str(pd.Timestamp(g.date).date())})
    return out


def xg_split(matches: pd.DataFrame, team: str, venue: str, n: int = 5) -> dict | None:
    games = last_games(matches, team, n, venue).dropna(subset=["home_xg"])
    if games.empty:
        return None
    mine, theirs = ("home_xg", "away_xg") if venue == "home" else ("away_xg", "home_xg")
    goals, conceded = ("home_goals", "away_goals") if venue == "home" else ("away_goals", "home_goals")
    return {"games": len(games), "venue": venue,
            "xg_for": round(games[mine].mean(), 2), "xg_against": round(games[theirs].mean(), 2),
            "goals_for": round(games[goals].mean(), 2), "goals_against": round(games[conceded].mean(), 2)}


def finishing(matches: pd.DataFrame, team: str, season: int) -> dict | None:
    """Goals scored minus xG this season: positive means clinical (or lucky) finishing."""
    m = matches[(matches.season == season) & ((matches.home_team == team) | (matches.away_team == team))]
    m = m.dropna(subset=["home_xg"])
    if m.empty:
        return None
    home = m.home_team == team
    goals = np.where(home, m.home_goals, m.away_goals).sum()
    xg = np.where(home, m.home_xg, m.away_xg).sum()
    return {"games": len(m), "goals": int(goals), "xg": round(float(xg), 1), "difference": round(float(goals - xg), 1)}


def head_to_head(matches: pd.DataFrame, home: str, away: str, n: int = 5) -> list[dict]:
    m = matches[((matches.home_team == home) & (matches.away_team == away)) |
                ((matches.home_team == away) & (matches.away_team == home))]
    return [{"date": str(pd.Timestamp(g.date).date()),
             "line": f"{g.home_team} {g.home_goals}-{g.away_goals} {g.away_team}",
             "margin": abs(int(g.home_goals) - int(g.away_goals))}
            for g in m.sort_values("date", ascending=False).head(n).itertuples()]


def table(matches: pd.DataFrame, season: int) -> pd.DataFrame:
    m = matches[matches.season == season]
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        gf, ga = m[f"{side}_goals"], m[f"{other}_goals"]
        rows.append(pd.DataFrame({"team": m[f"{side}_team"], "played": 1, "gf": gf, "ga": ga,
                                  "points": np.select([gf > ga, gf == ga], [3, 1], 0)}))
    t = pd.concat(rows).groupby("team", as_index=False).sum()
    t["gd"] = t.gf - t.ga
    t = t.sort_values(["points", "gd", "gf"], ascending=False).reset_index(drop=True)
    t["position"] = t.index + 1
    return t


def ranks(ratings: model.Ratings, clubs: set[str]) -> dict:
    """This season's clubs in the model's order, best first: by attack, by defence (fewest conceded)
    and overall. The videos say a reason as a place in these ("the fifth-best attack")."""
    r = {t: ratings.rating(t) for t in sorted(clubs)}
    return {"attack": sorted(r, key=lambda t: -r[t][0]), "defence": sorted(r, key=lambda t: r[t][1]),
            "overall": sorted(r, key=lambda t: r[t][1] - r[t][0])}


def market(odds: tuple | None) -> dict | None:
    if odds is None or any(o is None or not o or (isinstance(o, float) and math.isnan(o)) for o in odds):
        return None
    inv = np.array([1 / o for o in odds])
    p = inv / inv.sum()
    return {"H": float(p[0]), "D": float(p[1]), "A": float(p[2])}


# Back-tests over 2025/26 and 2026/27 (440 games): calls at 55% or more won 64% of the time,
# against 47% for the rest. These are the "strong" calls the scorecard tracks separately.
STRONG = 0.55


def confidence(probs: dict) -> str:
    # Back-tests showed favourites under 45% won no more often than near coin-flips,
    # so both count as too close to call.
    top, second = sorted(probs.values(), reverse=True)[:2]
    if top - second < 0.05 or top < 0.45:
        return "too close to call"
    if top >= STRONG:
        return "strong"
    return "solid"


# ---------------------------------------------------------------- putting it together

def explain(ratings: model.Ratings, matches: pd.DataFrame, home: str, away: str,
            players: pd.DataFrame | None = None, odds: tuple | None = None,
            season: int | None = None) -> dict:
    """Everything the match page needs, as plain data."""
    rho = ratings.settings.rho
    base_lam = math.exp(ratings.home_adv)
    base_mu = 1.0
    fs, news_rows = factors(ratings, home, away, players)
    lam = base_lam * math.exp(sum(f.log_home for f in fs))
    mu = base_mu * math.exp(sum(f.log_away for f in fs))
    probs = probabilities(lam, mu, rho)
    call = max(probs, key=probs.get)
    start = probabilities(base_lam, base_mu, rho)[call]
    shares = contributions(base_lam, base_mu, rho, fs, call)

    steps = sorted(({"label": f.label, "detail": f.detail, "kind": f.kind, "team": f.team,
                     "change": round(s * 100, 1)} for f, s in zip(fs, shares)),
                   key=lambda s: -abs(s["change"]))
    for s in steps:
        s["helps"] = _helps(call, s["change"], home, away)

    score_probs = model.score_matrix(lam, mu, rho)
    top_scores = sorted(((score_probs[i, j], i, j) for i in range(6) for j in range(6)), reverse=True)[:5]
    i, j = np.indices(score_probs.shape)
    agrees = {"H": i > j, "D": i == j, "A": i < j}[call]
    best = np.unravel_index(np.argmax(np.where(agrees, score_probs, -1)), score_probs.shape)

    season = season if season is not None else int(matches.season.max())
    standings = table(matches, season)
    pos = {r.team: {"position": int(r.position), "points": int(r.points), "played": int(r.played)}
           for r in standings.itertuples()}
    mkt = market(odds)

    result = {
        "home_team": home, "away_team": away,
        "call": call, "call_name": OUTCOME_NAMES[call],
        "confidence": confidence(probs),
        "probabilities": {k: round(v, 3) for k, v in probs.items()},
        "market": {k: round(v, 3) for k, v in mkt.items()} if mkt else None,
        "expected_goals": {"home": round(lam, 2), "away": round(mu, 2)},
        "score": [int(best[0]), int(best[1])],
        "top_scores": [{"score": [int(a), int(b)], "probability": round(float(p), 3)} for p, a, b in top_scores],
        "waterfall": {"start": round(start * 100, 1), "start_label": "Average home side v average visitor",
                      "steps": steps, "total": round(probs[call] * 100, 1)},
        "standings": {"home": pos.get(home), "away": pos.get(away)},
        "xg": {"home": xg_split(matches, home, "home"), "away": xg_split(matches, away, "away")},
        "finishing": {"home": finishing(matches, home, season), "away": finishing(matches, away, season)},
        "form": {"home": form(matches, home), "away": form(matches, away)},
        "head_to_head": head_to_head(matches, home, away),
        "team_news": news_rows,
        "ranks": ranks(ratings, set(pos) | {home, away}),
    }
    games = luck.team_games(matches)
    result["luck"] = {"home": luck.recent(games, home), "away": luck.recent(games, away)}
    # Written into every published call before kick-off, so the pre-registered test can't be back-fitted.
    result["luck_gap"] = luck.luck_gap(games, home, away)
    result["reasons"] = reasons(result)
    result["headline"] = headline(result)
    result["change_our_mind"] = change_our_mind(ratings, fs, base_lam, base_mu, call, home, away, steps)
    return result


def _a(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def _pos(team: str) -> str:
    """Possessive: Arsenal's, Spurs'."""
    return f"{team}'" if team.endswith("s") else f"{team}'s"


def _helps(call: str, change: float, home: str, away: str) -> str:
    """Which side a step pushes towards, for colouring the chart."""
    if call == "D":
        return "draw" if change >= 0 else "against"
    favoured, other = (home, away) if call == "H" else (away, home)
    return favoured if change >= 0 else other


# ---------------------------------------------------------------- words

def headline(r: dict) -> str:
    home, away = r["home_team"], r["away_team"]
    if r["confidence"] == "too close to call":
        return f"Nothing to split {home} and {away}"
    winner, loser = (home, away) if r["call"] == "H" else (away, home)
    if r["call"] == "D":
        return f"{home} and {away} look evenly matched"
    top = next((s for s in r["waterfall"]["steps"] if s["change"] > 0), None)
    if top is None:
        return f"{winner} edge it against {loser}"
    if top["kind"] == "news":
        team = top["team"]
        return f"Missing players leave {team} exposed" if team == loser else f"{winner} hit by absences but still favoured"
    team = top["team"]
    if top["kind"] == "attack":
        return f"{_pos(winner)} attack should be too much for {loser}"
    if team == loser:
        return f"{_pos(loser)} leaky defence hands {winner} the edge"
    return f"{_pos(winner)} defence gives them the edge over {loser}"


def reasons(r: dict) -> list[dict]:
    """Short reasons a reader can check, strongest first."""
    home, away = r["home_team"], r["away_team"]
    out = []

    hx, ax = r["xg"]["home"], r["xg"]["away"]
    if hx and ax:
        out.append({
            "topic": "Chances created and allowed",
            "title": _xg_title(home, away, hx, ax),
            "text": (f"Expected goals (xG) measures the quality of chances. Over their last {hx['games']} home games "
                     f"{home} created {hx['xg_for']:.1f} and allowed {hx['xg_against']:.1f} a game. "
                     f"Over their last {ax['games']} away games {away} created {ax['xg_for']:.1f} "
                     f"and allowed {ax['xg_against']:.1f}."),
        })

    news = r["team_news"]
    if news:
        worst = news[0]
        out.append({
            "topic": "Team news",
            "title": f"{worst['team']} without {worst['player']}" if worst["impact"] != "Small"
            else "Only minor absences",
            "text": " ".join(
                f"{team}: " + ", ".join(f"{n['player']} ({n['status'].lower()})" for n in news if n["team"] == team) + "."
                for team in dict.fromkeys(n["team"] for n in news)),
        })

    fh, fa = r["form"]["home"], r["form"]["away"]
    if fh and fa:
        pts = lambda f: sum(3 if g["result"] == "W" else 1 if g["result"] == "D" else 0 for g in f)
        ph, pa = pts(fh), pts(fa)
        better = home if ph > pa else away if pa > ph else None
        out.append({
            "topic": "Form",
            "title": f"{better} arrive in better form" if better else "Similar recent form",
            "text": f"Last five league games: {home} took {ph} points, {away} took {pa}.",
        })

    luck_reason = results_vs_performance(r)
    if luck_reason:
        out.append(luck_reason)

    fin = [(t, r["finishing"][s]) for t, s in ((home, "home"), (away, "away")) if r["finishing"][s]]
    lucky = [(t, f) for t, f in fin if abs(f["difference"]) >= 3]
    if lucky and not luck_reason:
        t, f = max(lucky, key=lambda x: abs(x[1]["difference"]))
        more = f["difference"] > 0
        out.append({
            "topic": "Finishing",
            "title": f"{t} are {'scoring above' if more else 'scoring below'} their chances",
            "text": (f"{t} have {f['goals']} goals from {f['xg']} xG this season. Finishing like that "
                     f"tends to {'cool off' if more else 'pick up'}, so we lean on the chances, not the goals."),
        })

    h2h = r["head_to_head"]
    if h2h:
        close = sum(1 for g in h2h if g["margin"] <= 1)
        n = len(h2h)
        if n < 3:
            title = "Not much history to go on"
        elif close >= n / 2:
            title = "Tight when they meet"
        else:
            title = "Usually one-sided when they meet"
        count = "None" if close == 0 else "All" if close == n else f"{close}"
        out.append({
            "topic": "Head to head",
            "title": title,
            "text": f"{count} of the last {n} {'meeting' if n == 1 else 'meetings'} "
                    f"{'was' if n == 1 or close == 1 else 'were'} drawn or won by one goal. "
                    "Past meetings count for little in the model.",
        })
    return out


def results_vs_performance(r: dict) -> dict | None:
    """Whose results over the last 10 games have run furthest from what their chances deserved."""
    sides = [(t, r.get("luck", {}).get(s)) for t, s in ((r["home_team"], "home"), (r["away_team"], "away"))]
    notable = [(t, l) for t, l in sides if l and abs(l["difference"]) >= luck.NOTABLE_POINTS]
    if not notable:
        return None
    notable.sort(key=lambda x: -abs(x[1]["difference"]))
    t, l = notable[0]
    up = l["difference"] > 0
    gap = abs(round(l["difference"]))
    text = (f"{t} have {l['points']} points from their last {l['games']} league games, but their chances "
            f"were worth {l['xpts']:.0f}.")
    if len(notable) > 1:
        t2, l2 = notable[1]
        text += f" {t2} have {l2['points']}, from chances worth {l2['xpts']:.0f}."
    text += f" We rate teams on their chances, so {t} get {'less' if up else 'more'} credit than their results."
    return {
        "topic": "Results vs performance",
        "title": f"{_pos(t)} results {'flatter them' if up else 'undersell them'} by {gap} "
                 f"{'point' if gap == 1 else 'points'}",
        "text": text,
    }


def _xg_title(home: str, away: str, hx: dict, ax: dict) -> str:
    home_edge = (hx["xg_for"] - hx["xg_against"]) - (ax["xg_for"] - ax["xg_against"])
    if ax["xg_against"] >= 1.6:
        return f"{away} are leaking chances away from home"
    if hx["xg_against"] >= 1.6:
        return f"{home} are leaking chances at home"
    if home_edge > 0.5:
        return f"{home} are creating far more at home"
    if home_edge < -0.5:
        return f"{away} are creating far more on the road"
    return "Little between them on chances"


def change_our_mind(ratings, fs, base_lam, base_mu, call, home, away, steps) -> str:
    """What single factor, if it went away, would flip or soften the call."""
    rho = ratings.settings.rho
    for step in steps:
        if step["change"] <= 0:
            continue
        rest = [f for f in fs if f.label != step["label"]]
        lam = base_lam * math.exp(sum(f.log_home for f in rest))
        mu = base_mu * math.exp(sum(f.log_away for f in rest))
        probs = probabilities(lam, mu, rho)
        new_call = max(probs, key=probs.get)
        if step["kind"] == "news":
            what = f"if {step['team']}'s missing players are fit"
            biggest = f"{step['team']}'s team news"
        else:
            what = f"if {step['label']} were only average"
            biggest = step["label"]
        what = what[0].upper() + what[1:]
        if new_call != call:
            return f"{what}, this would swing to {_a(OUTCOME_NAMES[new_call])} ({probs[new_call]:.0%})."
        return (f"The biggest single factor is {biggest}. "
                f"{what}, the {OUTCOME_NAMES[call]} chance would drop to {probs[call]:.0%}.")
    return "No single factor is driving this call."
