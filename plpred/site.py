"""Build the Inkwell & Algorithm site: a static site from this week's predictions and our record.

    python -m plpred.site            # writes site/

Pages: the front page (this week's calls, our record, charts, last week's verdict,
the table), one page per match with the full breakdown, and the posting kit for
social media in social/ (see plpred.social).
"""
import argparse
import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from plpred import explain, seasons, social
from plpred.sources import api_football

log = logging.getLogger("plpred.site")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TEMPLATES = Path(__file__).resolve().parent / "templates"
CALL_NAMES = {"H": "Home win", "D": "Draw", "A": "Away win"}
SHORT = {"Man United": "Man Utd", "Nott'm Forest": "Forest", "Crystal Palace": "Palace",
         "Bournemouth": "B'mouth", "Aston Villa": "Villa", "Tottenham": "Spurs", "Brighton": "Brighton"}


# ---------------------------------------------------------------- loading

def latest_predictions(data: Path) -> tuple[dict, bool]:
    """The newest predictions file, preferring real ones over replays."""
    files = sorted((data / "predictions").glob("*.json"))
    real = [f for f in files if not f.name.startswith("replay-")]
    chosen = (real or files)[-1]
    return json.loads(chosen.read_text()), not real


def slug(home: str, away: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", f"{home}-v-{away}".lower()).strip("-")


def pct(p: float) -> int:
    return int(round(p * 100))


# ---------------------------------------------------------------- one game

def game_view(g: dict) -> dict:
    home, away = g["home_team"], g["away_team"]
    p = g["probabilities"]
    mkt = g.get("market")
    call_team = {"H": home, "A": away, "D": None}[g["call"]]
    ko = pd.Timestamp(g["kickoff"])
    if ko.tzinfo is not None:
        ko = ko.tz_convert("Europe/London")
    view = {
        **g,
        "slug": slug(home, away),
        "home_short": SHORT.get(home, home), "away_short": SHORT.get(away, away),
        "score_text": f"{g['score'][0]}–{g['score'][1]}",
        "call_label": CALL_NAMES[g["call"]],
        "call_team": call_team,
        "call_long": f"{call_team} win" if call_team else "Draw",
        "h": pct(p["H"]), "d": pct(p["D"]), "a": pct(p["A"]),
        "kickoff_text": ko.strftime("%a %H:%M") if (ko.hour or ko.minute) else ko.strftime("%a %d %b"),
        "kickoff_long": ko.strftime("%A %d %B, %H:%M") if (ko.hour or ko.minute) else ko.strftime("%A %d %B"),
        "our_p": pct(p[g["call"]]),
        "mkt": {k: pct(v) for k, v in mkt.items()} if mkt else None,
        "mkt_p": pct(mkt[g["call"]]) if mkt else None,
        "top_scores": [{"score": f"{s['score'][0]}–{s['score'][1]}", "p": round(s["probability"] * 100),
                        "side": "H" if s["score"][0] > s["score"][1] else "A" if s["score"][0] < s["score"][1] else "D"}
                       for s in g["top_scores"]],
    }
    for m in view["head_to_head"]:
        m["date_text"] = pd.Timestamp(m["date"]).strftime("%b %Y")
    view["front_reasons"] = front_reasons(view)
    view["waterfall_rows"] = waterfall_rows(g)
    view["xg_rows"] = xg_rows(g)
    view["form_rows"] = [{"team": t, "games": g["form"][side]} for t, side in ((home, "home"), (away, "away"))]
    return view


def front_reasons(v: dict) -> list[dict]:
    """Three short reasons for the front page: the two strongest factors, then the market."""
    out = []
    for s in [s for s in v["waterfall"]["steps"] if s["change"] > 0][:2]:
        out.append({"title": s["label"].capitalize() if s["label"][0].islower() else s["label"],
                    "text": s["detail"] + f". Worth {s['change']:+.0f} points to our call."})
    if v["mkt"]:
        fav = max(v["mkt"], key=v["mkt"].get)
        fav_name = {"H": v["home_team"], "A": v["away_team"], "D": "a draw"}[fav]
        if fav == v["call"]:
            title = "The market agrees" + (", just" if abs(v["our_p"] - v["mkt_p"]) <= 4 else "")
            text = (f"Bookmakers make {fav_name} {v['mkt'][fav]}% favourites. We have it at {v['our_p']}%.")
        else:
            title = "We part company with the market"
            text = (f"Bookmakers favour {fav_name} at {v['mkt'][fav]}%. We have {v['call_long'].lower()} "
                    f"at {v['our_p']}%.")
        out.append({"title": title, "text": text})
    elif v["reasons"]:
        r = v["reasons"][0]
        out.append({"title": r["title"], "text": r["text"]})
    return out[:3]


def waterfall_rows(g: dict) -> dict:
    w = g["waterfall"]
    running, peak = w["start"], w["start"]
    for s in w["steps"]:
        running += s["change"]
        peak = max(peak, running)
    scale = max(60, int(np.ceil(peak / 10) * 10))
    to_pct = lambda x: round(x / scale * 100, 1)
    rows = [{"label": w["start_label"], "tip": "Our starting point before anything about these two teams",
             "val": f"{w['start']:.0f}%", "left": 0, "width": to_pct(w["start"]), "fill": "start", "bold": False}]
    run = w["start"]
    for s in w["steps"]:
        lo, hi = sorted((run, run + s["change"]))
        run += s["change"]
        rows.append({"label": s["label"], "tip": s["detail"],
                     "val": f"{'+' if s['change'] >= 0 else '−'}{abs(s['change']):.0f}",
                     "left": to_pct(lo), "width": max(to_pct(hi - lo), 0.4),
                     "fill": "up" if s["change"] >= 0 else "down", "bold": False})
    rows.append({"label": f"Our call: {g['call_name']}" if g["call"] == "D"
                 else f"Our call: {(g['home_team'] if g['call'] == 'H' else g['away_team'])} win",
                 "tip": "Final chance", "val": f"{w['total']:.0f}%", "left": 0,
                 "width": to_pct(w["total"]), "fill": "total", "bold": True})
    return {"rows": rows, "scale": scale, "start_line": to_pct(w["start"])}


def xg_rows(g: dict) -> list[dict]:
    out = []
    for team, side, where in ((g["home_team"], "home", "home"), (g["away_team"], "away", "away")):
        x = g["xg"][side]
        if not x:
            continue
        out.append({"label": f"{team} created ({where})", "v": x["xg_for"], "side": side, "kind": "for"})
        out.append({"label": f"{team} allowed ({where})", "v": x["xg_against"], "side": side, "kind": "against"})
    top = max([r["v"] for r in out] + [2.5])
    for r in out:
        r["w"] = round(r["v"] / top * 100)
    return out


# ---------------------------------------------------------------- the week

def pick_features(views: list[dict]) -> dict:
    """Match of the week, the banker and an upset to watch."""
    def standing(t, side):
        s = t["standings"][side]
        return s["position"] if s else 20
    biggest = min(views, key=lambda v: standing(v, "home") + standing(v, "away"))
    banker = max(views, key=lambda v: v["our_p"])
    upsets = [v for v in views if v["mkt"] and max(v["mkt"], key=v["mkt"].get) != v["call"]]
    upset = max(upsets, key=lambda v: v["our_p"] - (v["mkt_p"] or 0)) if upsets else None
    return {"lead": biggest, "banker": banker if banker is not biggest else None, "upset": upset}


def gaps(views: list[dict]) -> list[dict]:
    rows = [v for v in views if v["mkt"]]
    rows.sort(key=lambda v: -(v["our_p"] - v["mkt_p"]))
    lo = max(0, min([min(v["our_p"], v["mkt_p"]) for v in rows] + [20]) // 5 * 5)
    hi = min(100, -(-max([max(v["our_p"], v["mkt_p"]) for v in rows] + [70]) // 5) * 5)
    span = hi - lo or 1
    pos = lambda p: round((p - lo) / span * 100, 1)
    out = [{"label": f"{v['home_short']} v {v['away_short']}",
            "tip": f"{v['home_team']} v {v['away_team']}: {v['call_long']}. Us {v['our_p']}%, bookmakers {v['mkt_p']}%",
            "us": pos(v["our_p"]), "mkt": pos(v["mkt_p"]),
            "line_left": pos(min(v["our_p"], v["mkt_p"])), "line_w": round(abs(v["our_p"] - v["mkt_p"]) / span * 100, 1),
            "gap": f"{v['our_p'] - v['mkt_p']:+d}"} for v in rows]
    return {"rows": out, "lo": lo, "mid": (lo + hi) // 2, "hi": hi}


def attack_defence(matches: pd.DataFrame, season: int, highlight: set[str]) -> dict:
    m = matches[(matches.season == season)].dropna(subset=["home_xg"])
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({"team": m[f"{side}_team"], "for": m[f"{side}_xg"], "against": m[f"{other}_xg"]}))
    t = pd.concat(rows).groupby("team").mean().reset_index()
    if t.empty:
        return {"teams": []}
    x0, x1 = np.floor(t["for"].min() * 5) / 5, np.ceil(t["for"].max() * 5) / 5
    y0, y1 = np.floor(t["against"].min() * 5) / 5, np.ceil(t["against"].max() * 5) / 5
    teams = []
    for r in t.itertuples():
        left = (r._2 - x0) / ((x1 - x0) or 1) * 100
        top = (r.against - y0) / ((y1 - y0) or 1) * 100  # low xG allowed at the top
        on = r.team in highlight
        teams.append({"name": r.team, "left": round(left, 1), "top": round(top, 1), "on": on,
                      "label_left": left > 70,
                      "tip": f"{r.team}: {r._2:.1f} xG created, {r.against:.1f} allowed per game"})
    return {"teams": teams, "x0": f"{x0:.1f}", "x1": f"{x1:.1f}",
            "xmid": round((t["for"].mean() - x0) / ((x1 - x0) or 1) * 100, 1),
            "ymid": round((t["against"].mean() - y0) / ((y1 - y0) or 1) * 100, 1)}


def record(data: Path, season: int) -> dict | None:
    path = data / "backtest" / f"predictions_{season}.csv"
    if not path.exists():
        return None
    p = pd.read_csv(path, parse_dates=["date"])
    if p.empty:
        return None
    p["right"] = p.call == p.result
    p["exact"] = p.score == p.home_goals.astype(str) + "-" + p.away_goals.astype(str)
    mk = p[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy()
    p["fav_right"] = np.where(np.isnan(mk).any(axis=1), np.nan,
                              (np.array(["H", "D", "A"])[np.nan_to_num(mk).argmax(axis=1)] == p.result))
    p["week"] = p.date.dt.to_period("W-MON").dt.start_time
    weeks = []
    for i, (wk, g) in enumerate(p.groupby("week"), start=1):
        us, fav, n = int(g.right.sum()), int(np.nansum(g.fav_right)), len(g)
        weeks.append({"label": wk.strftime("%d %b"), "us": us, "fav": fav, "n": n,
                      "us_h": round(us / max(n, 1) * 100), "fav_h": round(fav / max(n, 1) * 100),
                      "tip": f"Week of {wk.strftime('%d %b')}: us {us} of {n}, bookies' favourite {fav} of {n}"})
    weeks = weeks[-8:]
    better = sum(w["us"] > w["fav"] for w in weeks)
    worse = sum(w["us"] < w["fav"] for w in weeks)
    last = p[p.week == p.week.max()]
    verdict_rows = []
    for r in last.itertuples():
        verdict = "Exact score" if r.exact else "Right result" if r.right else "Wrong"
        verdict_rows.append({"match": f"{r.home_team} v {r.away_team}",
                             "call": f"{CALL_NAMES[r.call]} {r.score.replace('-', '–')}",
                             "result": f"{r.home_goals}–{r.away_goals}", "verdict": verdict,
                             "wrong": verdict == "Wrong"})
    fav_rate = np.nanmean(p.fav_right)
    return {
        "n": len(p), "right": int(p.right.sum()), "exact": int(p.exact.sum()),
        "rate": round(p.right.mean() * 100), "fav_rate": round(fav_rate * 100) if not np.isnan(fav_rate) else None,
        "weeks": weeks, "better": better, "worse": worse,
        "last": {"rows": verdict_rows, "right": int(last.right.sum()), "n": len(last),
                 "exact": int(last.exact.sum()), "label": last.date.min().strftime("%d %B")},
    }


def league_table(matches: pd.DataFrame, season: int) -> list[dict]:
    t = explain.table(matches, season)
    return t.to_dict("records")


# ---------------------------------------------------------------- build

def build(data: Path, out: Path) -> Path:
    preds, is_replay = latest_predictions(data)
    matches = pd.read_csv(data / "matches.csv", parse_dates=["date"])
    season = int(preds["season"][:4])
    views = [game_view(g) for g in preds["games"]]
    features = pick_features(views)
    lead = features["lead"]
    rest = [v for v in views if v is not lead]
    rnd = preds.get("round") or ""
    mw = re.search(r"(\d+)$", rnd)
    matchweek = int(mw.group(1)) if mw else int(explain.table(matches[matches.date < preds["as_of"]], season)
                                                 .played.max() or 0) + 1
    ctx = {
        "matchweek": matchweek,
        "season": seasons.label(season),
        "date_text": pd.Timestamp(preds["as_of"]).strftime("%A %d %B %Y"),
        "replay": is_replay,
        "lead": lead, "rest": rest, "features": features,
        "gaps": gaps(views),
        "scatter": attack_defence(matches[matches.date < preds["as_of"]], season,
                                  {lead["home_team"], lead["away_team"]} |
                                  ({features["upset"]["home_team"], features["upset"]["away_team"]}
                                   if features["upset"] else set())),
        "record": record(data, season),
        "table": league_table(matches[matches.date < preds["as_of"]], season),
        "built": datetime.now().strftime("%d %B %Y"),
        "badges": {},
    }
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]),
                      trim_blocks=True, lstrip_blocks=True)
    if out.exists():
        shutil.rmtree(out)
    (out / "match").mkdir(parents=True)
    for asset in ("style.css", "favicon.svg"):
        shutil.copy(TEMPLATES / asset, out / asset)
    badge_dir = data / "badges"
    if badge_dir.exists():
        shutil.copytree(badge_dir, out / "badges")
        ctx["badges"] = {t: f"{api_football.badge_name(t)}.png" for t in
                         {v["home_team"] for v in views} | {v["away_team"] for v in views} | set(matches.home_team)
                         if (badge_dir / f"{api_football.badge_name(t)}.png").exists()}
    (out / "index.html").write_text(env.get_template("index.html").render(**ctx, root="."))
    for v in views:
        (out / "match" / f"{v['slug']}.html").write_text(
            env.get_template("match.html").render(**ctx, g=v, root=".."))
    social.build_kit(ctx, views, env, out / "social")
    (out / ".nojekyll").write_text("")
    log.info("Built %d pages into %s (%s)", len(views) + 1, out, "replay" if is_replay else "live")
    return out


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "site")
    args = parser.parse_args(argv)
    build(args.data, args.out)


if __name__ == "__main__":
    main()
