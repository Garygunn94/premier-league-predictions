"""Build the Inkwell & Algorithm site: a static site from each week's predictions and our record.

    python -m plpred.site            # writes site/

Pages: the newest matchweek as the front page, an about/ page (what the site is, how it
works, the data sources, every matchweek), one page per matchweek in <season>/matchweek/<n>/ (the calls, our record, charts, last week's
verdict, the table and the table the chances deserved) with a page per match beside it,
a record/ page on how often our percentages came true (with the pre-registered luck-gap
test), the projected final table in table/ (see plpred.simulate), the FPL corner in fpl/
(see plpred.fantasy), a page per club in club/<name>/, us against the pundits in pundits/
(see plpred.pundits), and the posting kit for social media and the weekly newsletter in
social/ for the newest week (see plpred.social). It also writes sitemap.xml, an Atom feed
of the matchweeks (feed.xml), robots.txt and a 404 page. Every page has a search title, a
description, a canonical address and schema.org structured data (see page_meta).

Optional settings, read from the environment (the Publish site workflow passes them from
the repository's Actions variables):
    SITE_URL        the site's public address, for share previews, the sitemap and the feed
    GOATCOUNTER     a GoatCounter site code, to count visits (no cookies, no banner)
    NEWSLETTER_URL  the newsletter's sign-up page, for the sign-up link in every footer
"""
import argparse
import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from plpred import explain, fantasy, luck, model, pundits, seasons, simulate, social
from plpred.predict import load_settings
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


def all_predictions(data: Path) -> tuple[list[dict], bool]:
    """Every real predictions file, oldest first. Until one exists, the newest replay alone."""
    files = sorted((data / "predictions").glob("*.json"))
    real = [f for f in files if not f.name.startswith("replay-")]
    chosen = real or files[-1:]
    return [json.loads(f.read_text()) for f in chosen], not real


def merge_weeks(all_preds: list[dict], matches: pd.DataFrame) -> list[dict]:
    """One predictions dict per matchweek. Files are read oldest first, so a later run for a game
    replaces an earlier one, and games only in the earlier run (say, called after kick-off) stay."""
    weeks = {}
    for preds in all_preds:
        season = int(preds["season"][:4])
        key = (season, matchweek_of(preds, matches, season))
        if key not in weeks:
            weeks[key] = {**preds, "games": list(preds["games"])}
            continue
        w = weeks[key]
        fresh = {(g["home_team"], g["away_team"]) for g in preds["games"]}
        w["games"] = sorted([g for g in w["games"] if (g["home_team"], g["away_team"]) not in fresh] + preds["games"],
                            key=lambda g: g["kickoff"])
        w["as_of"] = min(w["as_of"], preds["as_of"])
        w["round"] = w.get("round") or preds.get("round")
    return list(weeks.values())


def matchweek_of(preds: dict, matches: pd.DataFrame, season: int) -> int:
    mw = re.search(r"(\d+)$", preds.get("round") or "")
    if mw:
        return int(mw.group(1))
    played = explain.table(matches[matches.date < preds["as_of"]], season).played.max()
    return int(played or 0) + 1


def slug(home: str, away: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", f"{home}-v-{away}".lower()).strip("-")


def club_slug(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", team.lower().replace("'", "")).strip("-")


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
        "result_text": f"{g['result'][0]}–{g['result'][1]}" if g.get("result") else None,
        "result_call": ("H" if g["result"][0] > g["result"][1] else "A" if g["result"][0] < g["result"][1] else "D")
                       if g.get("result") else None,
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
    """Three short reasons for the front page: the two strongest factors, then the market. When a
    side's results have run well away from its chances, that takes the second factor's place."""
    out = []
    luck_reason = next((r for r in v["reasons"] if r["topic"] == "Results vs performance"), None)
    for s in [s for s in v["waterfall"]["steps"] if s["change"] > 0][:1 if luck_reason else 2]:
        out.append({"title": s["label"].capitalize() if s["label"][0].islower() else s["label"],
                    "text": s["detail"] + f". Worth {s['change']:+.0f} points to our call."})
    if luck_reason:
        out.append({"title": luck_reason["title"], "text": luck_reason["text"]})
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


def record(data: Path, season: int, before=None) -> dict | None:
    """Our record this season, from games played before `before` (default: all of them)."""
    path = data / "backtest" / f"predictions_{season}.csv"
    if not path.exists():
        return None
    p = pd.read_csv(path, parse_dates=["date"])
    if before is not None:
        p = p[p.date < pd.Timestamp(before)]
    if p.empty:
        return None
    p["right"] = p.call == p.result
    p["exact"] = p.score == p.home_goals.astype(str) + "-" + p.away_goals.astype(str)
    mk = p[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy()
    p["fav_right"] = np.where(np.isnan(mk).any(axis=1), np.nan,
                              (np.array(["H", "D", "A"])[np.nan_to_num(mk).argmax(axis=1)] == p.result))
    p["fav_call"] = np.where(np.isnan(mk).any(axis=1), None, np.array(["H", "D", "A"])[np.nan_to_num(mk).argmax(axis=1)])
    # Back-tests number the rounds; older files without it fall back to calendar weeks.
    p["week"] = p.gameweek if "gameweek" in p else p.date.dt.to_period("W-MON").dt.start_time
    weeks = []
    for _, g in p.groupby("week"):
        us, fav, n = int(g.right.sum()), int(np.nansum(g.fav_right)), len(g)
        wk = g.date.min()
        weeks.append({"label": wk.strftime("%d %b"), "us": us, "fav": fav, "n": n,
                      "us_h": round(us / max(n, 1) * 100), "fav_h": round(fav / max(n, 1) * 100),
                      "tip": f"Round starting {wk.strftime('%d %b')}: us {us} of {n}, bookies' favourite {fav} of {n}"})
    weeks = weeks[-8:]
    better = sum(w["us"] > w["fav"] for w in weeks)
    worse = sum(w["us"] < w["fav"] for w in weeks)
    last = p[p.week == p.week.iloc[p.date.argmax()]]
    verdict_rows = []
    for r in last.itertuples():
        verdict = "Exact score" if r.exact else "Right result" if r.right else "Wrong"
        verdict_rows.append({"match": f"{r.home_team} v {r.away_team}",
                             "home_team": r.home_team, "away_team": r.away_team, "call_code": r.call,
                             "result_code": r.result, "p": pct(max(r.p_home, r.p_draw, r.p_away)),
                             "season": int(r.season),
                             "call": f"{CALL_NAMES[r.call]} {r.score.replace('-', '–')}",
                             "result": f"{r.home_goals}–{r.away_goals}", "verdict": verdict,
                             "wrong": verdict == "Wrong"})
    fav_rate = np.nanmean(p.fav_right)
    split = p[p.fav_call.notna() & (p.call != p.fav_call)]
    strong = p[p[["p_home", "p_draw", "p_away"]].max(axis=1) >= explain.STRONG]
    strong_fav = np.nanmean(strong.fav_right) if len(strong) else np.nan
    return {
        "strong": {"n": len(strong), "right": int(strong.right.sum()),
                   "rate": round(strong.right.mean() * 100) if len(strong) else None,
                   "fav_rate": None if np.isnan(strong_fav) else round(strong_fav * 100)},
        "disagree": {"n": len(split), "us": int(split.right.sum()), "fav": int(np.nansum(split.fav_right))},
        "n": len(p), "right": int(p.right.sum()), "exact": int(p.exact.sum()),
        "rate": round(p.right.mean() * 100), "fav_rate": round(fav_rate * 100) if not np.isnan(fav_rate) else None,
        "weeks": weeks, "better": better, "worse": worse,
        "last": {"rows": verdict_rows, "right": int(last.right.sum()), "n": len(last),
                 "exact": int(last.exact.sum()), "label": last.date.min().strftime("%d %B")},
    }


def biggest_miss(rec: dict | None, matches: pd.DataFrame) -> dict | None:
    """Last round's most confident wrong call, and whether the chances (xG) backed us or the result."""
    wrong = [r for r in (rec or {}).get("last", {}).get("rows", []) if r["wrong"]]
    if not wrong:
        return None
    r = max(wrong, key=lambda r: r["p"])
    m = matches[(matches.season == r["season"]) & (matches.home_team == r["home_team"])
                & (matches.away_team == r["away_team"])]
    called = {"H": f"{r['home_team']} to win", "A": f"{r['away_team']} to win", "D": "a draw"}[r["call_code"]]
    out = {**r, "called": called}
    hx, ax = (m.iloc[0].home_xg, m.iloc[0].away_xg) if len(m) else (np.nan, np.nan)
    if np.isnan(hx) or np.isnan(ax):
        out.update(kind="No xG yet", text="We don't have the expected goals for this one yet, so the jury is out.")
        return out
    margin = hx - ax
    xg_side = "H" if margin > 0.35 else "A" if margin < -0.35 else "D"
    xg = f"{r['home_team']} {hx:.1f}, {r['away_team']} {ax:.1f}"
    if xg_side == r["call_code"]:
        out.update(kind="Unlucky", text=f"The chances backed our call (xG: {xg}). On another day this lands, "
                                        "so it doesn't change how we rate either side much.")
    elif xg_side == r["result_code"]:
        out.update(kind="Our mistake", text=f"The chances went the way of the result too (xG: {xg}). "
                                            "We misread this one, and the ratings will move to reflect it.")
    else:
        out.update(kind="Somewhere in between", text=f"The chances pointed to neither our call nor the result "
                                                     f"(xG: {xg}). A closer game than either the score or we suggested.")
    return out


def league_table(matches: pd.DataFrame, season: int) -> list[dict]:
    t = explain.table(matches, season)
    return t.to_dict("records")


# ---------------------------------------------------------------- build

def week_context(preds: dict, is_replay: bool, data: Path, matches: pd.DataFrame) -> dict:
    """Everything one matchweek's pages need."""
    season = int(preds["season"][:4])
    matchweek = matchweek_of(preds, matches, season)
    views = [game_view(g) for g in preds["games"]]
    week_dir = f"{seasons.label(season).replace('/', '-')}/matchweek/{matchweek}/"
    for v in views:
        v["path"] = f"{week_dir}{v['slug']}.html"
    late = [v for v in views if v.get("after_kickoff")]
    # The features (match of the week, banker, upset) are predictions, so they come from live calls.
    features = pick_features([v for v in views if not v.get("after_kickoff")] or views)
    lead = features["lead"]
    before = matches[matches.date < preds["as_of"]]
    return {
        "matchweek": matchweek,
        "week_dir": week_dir,
        "season": seasons.label(season),
        "as_of": pd.Timestamp(preds["as_of"]),
        "date_text": pd.Timestamp(preds["as_of"]).strftime("%A %d %B %Y"),
        "replay": is_replay,
        "late": len(late),
        "views": views,
        "lead": lead, "rest": [v for v in views if v is not lead], "features": features,
        "gaps": gaps(views),
        "scatter": attack_defence(before, season,
                                  {lead["home_team"], lead["away_team"]} |
                                  ({features["upset"]["home_team"], features["upset"]["away_team"]}
                                   if features["upset"] else set())),
        "record": (rec := record(data, season, preds["as_of"])),
        "miss": biggest_miss(rec, matches),
        "table": league_table(before, season),
        "deserved": luck.deserved_table(before, season),
    }


def club_view(team: str, w: dict, sim: dict | None, strip: dict | None, matches: pd.DataFrame,
              data: Path, season: int) -> dict:
    """Everything a club's page needs, as of the newest matchweek."""
    # Every result we have, so the page keeps up with the daily data refresh.
    before = matches[(matches.season == season)
                     & ((matches.home_team == team) | (matches.away_team == team))].sort_values("date")
    games = []
    for g in before.itertuples():
        home = g.home_team == team
        gf, ga = (g.home_goals, g.away_goals) if home else (g.away_goals, g.home_goals)
        xf, xa = (g.home_xg, g.away_xg) if home else (g.away_xg, g.home_xg)
        games.append({"date": pd.Timestamp(g.date).strftime("%d %b"), "opponent": g.away_team if home else g.home_team,
                      "venue": "H" if home else "A", "score": f"{int(gf)}–{int(ga)}",
                      "result": "W" if gf > ga else "L" if gf < ga else "D",
                      "xf": None if pd.isna(xf) else round(float(xf), 1), "xa": None if pd.isna(xa) else round(float(xa), 1)})
    top = max([x for g in games for x in (g["xf"], g["xa"]) if x is not None] + [3.0])
    for g in games:
        g["xf_h"] = round((g["xf"] or 0) / top * 100)
        g["xa_h"] = round((g["xa"] or 0) / top * 100)
    calls = None
    path = data / "backtest" / f"predictions_{season}.csv"
    if path.exists():
        p = pd.read_csv(path)
        p = p[(p.home_team == team) | (p.away_team == team)]
        if len(p):
            calls = {"n": len(p), "right": int((p.call == p.result).sum())}
            calls["rate"] = round(calls["right"] / calls["n"] * 100)
    return {
        "team": team, "slug": club_slug(team),
        "row": next((t for t in w["table"] if t["team"] == team), None),
        "deserved": next((t for t in w["deserved"] if t["team"] == team), None),
        "sim": next((t for t in (sim or {}).get("teams", []) if t["team"] == team), None),
        "games": games[::-1],
        "calls": calls,
        "now": next((v for v in w["views"] if team in (v["home_team"], v["away_team"])), None),
        "strip": next((r for r in (strip or {}).get("rows", []) if r["team"] == team), None),
        "gameweeks": (strip or {}).get("gameweeks", []),
    }


BRAND = "Inkwell & Algorithm"
SAME_AS = ["https://x.com/InkwellAlgo", "https://www.instagram.com/inkwellandalgorithm/"]


def clean_path(path: str) -> str:
    """The address we want search engines to use: 2026-27/matchweek/7/ rather than .../index.html."""
    return path[:-len("index.html")] if path.endswith("index.html") else path


def short_description(text: str, limit: int = 160) -> str:
    """Cut a description at a word so search results show it whole."""
    if len(text) <= limit:
        return text
    return text[:limit - 1].rsplit(" ", 1)[0].rstrip(",;:.") + "…"


def page_meta(path: str, title: str, description: str, image: str | None = None, *,
              search_title: str | None = None, canonical: str | None = None,
              crumbs: list[tuple[str, str]] | None = None, ld: list[dict] | None = None) -> dict:
    """A page's share preview and search details.

    title is the share headline; search_title is the <title> (keyword-led, the brand after it);
    canonical is the path search engines should index this page under (defaults to its own);
    crumbs are (name, path) pairs after the front page, for breadcrumbs in search results;
    ld is any extra structured data (schema.org) for the page.
    """
    return {"path": path, "url": clean_path(path if canonical is None else canonical),
            "title": title, "search_title": f"{search_title or title} | {BRAND}",
            "description": short_description(description), "image": image or "social/png/brand.png",
            "crumbs": crumbs, "ld": ld or []}


def structured_data(meta: dict, site_url: str) -> list[dict]:
    """schema.org JSON-LD: who we are on the front page, breadcrumbs elsewhere, plus the page's own."""
    if not site_url:
        return []
    out = []
    if meta["path"] == "":
        out.append({"@context": "https://schema.org", "@graph": [
            {"@type": "Organization", "@id": f"{site_url}#org", "name": BRAND, "url": site_url,
             "logo": f"{site_url}social/png/brand.png", "sameAs": SAME_AS},
            {"@type": "WebSite", "@id": f"{site_url}#site", "name": BRAND, "url": site_url,
             "description": "Data-driven Premier League predictions, with the reasons.",
             "publisher": {"@id": f"{site_url}#org"}, "inLanguage": "en-GB"}]})
    if meta.get("crumbs"):
        items = [("Home", "")] + meta["crumbs"]
        out.append({"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": [
            {"@type": "ListItem", "position": i, "name": name, "item": site_url + clean_path(path)}
            for i, (name, path) in enumerate(items, 1)]})
    return out + [{"@context": "https://schema.org", **d} for d in meta["ld"]]


def match_event(v: dict, site_url: str) -> dict:
    """The match itself, so search engines know which game a breakdown is about."""
    ko = pd.Timestamp(v["kickoff"])
    event = {"@type": "SportsEvent", "name": f"{v['home_team']} v {v['away_team']}", "sport": "Association football",
             "startDate": ko.isoformat() if (ko.hour or ko.minute) else ko.strftime("%Y-%m-%d"),
             "homeTeam": {"@type": "SportsTeam", "name": v["home_team"]},
             "awayTeam": {"@type": "SportsTeam", "name": v["away_team"]},
             "superEvent": {"@type": "SportsEvent", "name": "Premier League"}}
    if v.get("venue"):
        event["location"] = {"@type": "Place", "name": v["venue"]}
    if site_url:
        event["url"] = site_url + v["path"]
    return event


def write_extras(out: Path, weeks: list[dict], pages: list[str], site_url: str, built: datetime) -> None:
    """sitemap.xml (pages is the indexable paths), feed.xml (one entry per matchweek, newest first) and robots.txt."""
    lastmod = built.strftime("%Y-%m-%d")
    urls = "".join(f"<url><loc>{escape(site_url + p)}</loc><lastmod>{lastmod}</lastmod></url>" for p in pages)
    (out / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n'
                                     f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>\n')
    entries = []
    for w in weeks[::-1]:
        link = f"{site_url}{w['week_dir']}"
        when = w["as_of"].strftime("%Y-%m-%dT00:00:00Z")
        summary = "; ".join(f"{v['home_team']} v {v['away_team']}: {v['call_long'].lower() if v['call'] != 'D' else 'draw'} "
                            f"{v['score_text']} ({v['our_p']}%)" for v in w["views"])
        title = f"Matchweek {w['matchweek']}: {w['lead']['headline']}"
        entries.append(f"<entry><title>{escape(title)}</title>"
                       f"<link href=\"{escape(link)}\"/><id>{escape(link)}</id><updated>{when}</updated>"
                       f"<summary>{escape(summary)}</summary></entry>")
    (out / "feed.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<feed xmlns="http://www.w3.org/2005/Atom">'
        f"<title>Inkwell &amp; Algorithm</title><subtitle>Data-driven Premier League predictions, with the reasons."
        f"</subtitle><link href=\"{escape(site_url)}\"/><link rel=\"self\" href=\"{escape(site_url)}feed.xml\"/>"
        f"<id>{escape(site_url)}</id><author><name>Inkwell &amp; Algorithm</name></author><updated>{built.strftime('%Y-%m-%dT%H:%M:%SZ')}</updated>"
        + "".join(entries) + "</feed>\n")
    (out / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {site_url}sitemap.xml\n")


# Matchweeks of this season were first published at matchweek/<n>/, without the season.
OLD_ADDRESS_SEASON = "2026/27"
REDIRECT = ('<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><title>Moved</title>'
            '<link rel="canonical" href="{url}"><meta http-equiv="refresh" content="0; url={url}">'
            '<script>location.replace({url_js} + location.hash)</script></head>'
            '<body><p>This page has moved to <a href="{url}">{url}</a>.</p></body></html>\n')


def write_old_addresses(out: Path, weeks: list[dict], site_url: str) -> None:
    """Redirects from the first, season-less addresses, so links already posted keep working."""
    for w in weeks:
        if w["season"] != OLD_ADDRESS_SEASON:
            continue
        old_dir = out / "matchweek" / str(w["matchweek"])
        old_dir.mkdir(parents=True, exist_ok=True)
        targets = {"index.html": w["week_dir"]} | {f"{v['slug']}.html": v["path"] for v in w["views"]}
        for name, new in targets.items():
            # Relative when there is no site address, so a local build still works.
            url = site_url + new if site_url else "../../" + new
            (old_dir / name).write_text(REDIRECT.format(url=escape(url, {'"': "&quot;"}), url_js=json.dumps(url)))


def build(data: Path, out: Path) -> Path:
    all_preds, is_replay = all_predictions(data)
    matches = pd.read_csv(data / "matches.csv", parse_dates=["date"])
    weeks = [week_context(preds, is_replay, data, matches) for preds in merge_weeks(all_preds, matches)]
    weeks.sort(key=lambda w: w["as_of"])
    for i, w in enumerate(weeks):
        w["prev"] = weeks[i - 1] if i else None
        w["next"] = weeks[i + 1] if i + 1 < len(weeks) else None
        w["first_kickoff"] = min(pd.Timestamp(v["kickoff"]) for v in w["views"]).strftime("%d %B")
    latest = weeks[-1]
    season = int(latest["season"][:4])
    settings = load_settings(data)
    built = datetime.now()

    # Season projection from every result we have (so a midweek rebuild includes the weekend),
    # and as it stood a week earlier, for the week-on-week moves.
    sim_as_of = max(latest["as_of"], matches.date.max() + pd.Timedelta(days=1))
    sim = simulate.run(matches, season, sim_as_of, settings)
    sim = simulate.with_changes(sim, simulate.run(matches, season, sim_as_of - pd.Timedelta(days=7), settings))
    latest["sim"] = sim
    # The newest week's tables follow the daily results; older weeks keep the table they went into.
    latest["table"] = league_table(matches, season)
    latest["deserved"] = luck.deserved_table(matches, season)
    latest["table_date"] = matches.date.max().strftime("%d %B")
    latest["sim_moves"] = simulate.headline_moves(sim) if sim else []
    known = matches[matches.date < latest["as_of"]]
    ratings = model.fit(known, as_of=latest["as_of"], settings=settings)
    fpl = {"this_week": fantasy.this_week(latest["views"], settings.rho),
           "strip": fantasy.strip(fantasy.load_fixtures(data), ratings, latest["views"])}
    fpl["clean_sheets"] = sorted(fpl["this_week"], key=lambda r: -r["cs"])
    fpl["scoring"] = sorted(fpl["this_week"], key=lambda r: -r["xg_for"])

    site_url = social.SITE_URL
    common = {
        "built": built.strftime("%d %B %Y"),
        "badges": {},
        "weeks": weeks[::-1],
        "latest": latest,
        "site_url": site_url,
        "goatcounter": os.environ.get("GOATCOUNTER", "").strip(),
        "newsletter_url": os.environ.get("NEWSLETTER_URL", "").strip(),
        "clubs": {t["team"]: club_slug(t["team"]) for t in latest["table"]},
    }
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]),
                      trim_blocks=True, lstrip_blocks=True)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for asset in ("style.css", "favicon.svg"):
        shutil.copy(TEMPLATES / asset, out / asset)
    badge_dir = data / "badges"
    if badge_dir.exists():
        shutil.copytree(badge_dir, out / "badges")
        teams = set(matches.home_team) | {t for w in weeks for v in w["views"] for t in (v["home_team"], v["away_team"])}
        common["badges"] = {t: f"{api_football.badge_name(t)}.png" for t in teams
                            if (badge_dir / f"{api_football.badge_name(t)}.png").exists()}
    written, indexed = [], []

    def write(path: str, template: str, meta: dict, root: str | None = None, **ctx) -> None:
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        root = root if root is not None else "/".join([".."] * path.count("/")) or "."
        meta = {**meta, "jsonld": structured_data(meta, site_url)}
        target.write_text(env.get_template(template).render(**common, **ctx, meta=meta, root=root))
        written.append(path)
        if not meta.get("noindex") and meta["url"] == clean_path(path) and meta["url"] not in indexed:
            indexed.append(meta["url"])

    season_label = latest["season"]
    tagline = "Data-driven Premier League predictions, with the reasons. New every Thursday."
    write("about/index.html", "about.html",
          page_meta("about/index.html", "About Inkwell & Algorithm", "How we predict every Premier League match: "
                    "attack and defence ratings from goals and expected goals, the reasons behind each call, "
                    "and our record against the bookmakers.",
                    search_title="About: how our Premier League predictions work", crumbs=[("About", "about/")]),
          record=latest["record"], replay=is_replay)
    write("record/index.html", "calibration.html",
          page_meta("record/index.html", "Are our percentages right?",
                    "Every chance we've given, checked against what happened, and our public test.",
                    search_title="Premier League prediction record: are our percentages right?",
                    crumbs=[("Our record", "record/")]),
          record=latest["record"], cal=luck.calibration(data, matches), prereg=luck.preregistered(data, matches))
    if sim:
        write("table/index.html", "projection.html",
              page_meta("table/index.html", f"How the {season_label} season ends: {sim['runs']:,} simulations",
                        f"Our predicted {season_label} Premier League table: title, top-four and relegation chances "
                        f"for every club from {sim['runs']:,} simulations, updated every week.",
                        search_title=f"Premier League {season_label} predicted table",
                        crumbs=[("Season projection", "table/")]),
              sim=sim, moves=latest["sim_moves"])
    write("fpl/index.html", "fpl.html",
          page_meta("fpl/index.html", f"FPL corner, Gameweek {latest['matchweek']}",
                    f"FPL Gameweek {latest['matchweek']}: clean-sheet and scoring chances for every Premier League "
                    "club, and how hard the fixtures ahead are.",
                    search_title=f"FPL Gameweek {latest['matchweek']}: clean sheet and goal chances",
                    crumbs=[("FPL corner", "fpl/")]),
          fpl=fpl)
    write("pundits/index.html", "pundits.html",
          page_meta("pundits/index.html", "The algorithm v the pundits",
                    "Well-known pundits' Premier League predictions, scored beside ours on the same games.",
                    search_title="Premier League predictions: the algorithm v the pundits",
                    crumbs=[("V the pundits", "pundits/")]),
          board=pundits.scoreboard(data, matches, all_preds if not is_replay else []))
    clubs = [club_view(t, latest, sim, fpl["strip"], matches, data, season) for t in sorted(common["clubs"])]
    write("club/index.html", "clubs.html",
          page_meta("club/index.html", "The clubs", "Every Premier League club's season in numbers: projected "
                    "finish, form, expected goals and fixtures.",
                    search_title=f"Premier League clubs {season_label}: the season in numbers",
                    crumbs=[("Clubs", "club/")]), club_list=clubs)
    for c in clubs:
        write(f"club/{c['slug']}/index.html", "club.html",
              page_meta(f"club/{c['slug']}/index.html", f"{c['team']}: the season in numbers",
                        f"{c['team']}'s {season_label} Premier League season in numbers: where they're predicted "
                        "to finish, form, expected goals and the fixtures ahead.",
                        search_title=f"{c['team']} {season_label}: predicted finish, form and fixtures",
                        crumbs=[("Clubs", "club/"), (c["team"], f"club/{c['slug']}/")]), c=c)
    for w in weeks:
        ctx = {**w}
        is_latest = w is latest
        week_path = f"{w['week_dir']}index.html"
        week_title = f"Matchweek {w['matchweek']}: {w['lead']['headline']}"
        week_search = f"Premier League predictions: Matchweek {w['matchweek']}, {w['season']}"
        week_desc = (f"Predictions for every Premier League match in Matchweek {w['matchweek']} "
                     f"({w['season']}, from {w['first_kickoff']}): our call, the likely score and the reasons behind it.")
        week_crumbs = [(f"Matchweek {w['matchweek']}", w["week_dir"])]
        # The newest week is also the front page, so search engines are pointed at the front page
        # until the next week replaces it.
        write(week_path, "week.html",
              page_meta(week_path, week_title, week_desc, "social/png/round.png" if is_latest else None,
                        search_title=week_search, canonical="" if is_latest else None, crumbs=week_crumbs), **ctx)
        if is_latest:
            # The front page is this week's calls, so a link from a social bio lands on them.
            write("index.html", "week.html",
                  page_meta("", week_title, week_desc, "social/png/round.png", search_title=week_search), **ctx)
        for v in w["views"]:
            call = "draw" if v["call"] == "D" else f"{v['call_team']} win"
            write(v["path"], "match.html",
                  page_meta(v["path"], f"{v['home_team']} v {v['away_team']}: {v['headline']}",
                            f"{v['home_team']} v {v['away_team']} prediction, {v['kickoff_long']}. "
                            f"Our call: {v['call_long'] if v['call'] != 'D' else 'draw'} {v['score_text']} ({v['our_p']}%). "
                            + (v["front_reasons"][0]["text"] if v["front_reasons"] else ""),
                            f"social/png/call-{v['slug']}.png" if is_latest else None,
                            search_title=f"{v['home_team']} v {v['away_team']} prediction: {call} {v['score_text']}",
                            crumbs=week_crumbs + [(f"{v['home_team']} v {v['away_team']}", v["path"])],
                            ld=[match_event(v, site_url)]), **ctx, g=v)
    # GitHub Pages serves 404.html at whatever address was missing, so its links can't be relative.
    write("404.html", "missing.html", {**page_meta("404.html", "Page not found", tagline), "noindex": True},
          root=site_url.rstrip("/"))
    write_old_addresses(out, weeks, site_url)
    social.build_kit({**latest, **common, "fpl": fpl}, latest["views"], env, out / "social")
    write_extras(out, weeks, indexed, site_url, built)
    (out / ".nojekyll").write_text("")
    log.info("Built %d pages for %d matchweeks into %s (%s)", len(written), len(weeks), out,
             "replay" if is_replay else "live")
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
