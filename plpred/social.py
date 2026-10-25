"""The weekly posting kit: images and captions for X and Instagram, and the newsletter.

    python -m plpred.social --png site/social   # render the cards built by plpred.site to PNG

`plpred.site` calls `build_kit` to write the kit into site/social/: one HTML card per
image (1080 x 1080, the site's dark look) and a page listing every card with its
caption, ready to copy, plus the week's newsletter as Markdown (newsletter.md) for
pasting into the newsletter service, and the scores for our Superbru entry (superbru.txt). Rendering the cards to PNG needs Playwright and Chromium, so
it is a separate step that the Publish site workflow runs after the build.

    python -m plpred.social --email site/social # email the rendered kit, one message per post

The kit comes in two editions, each emailed on its own day: predictions (the round's calls,
carousels, the bold call, FPL and the Superbru picks) the day before the round's first game, and
results (the results carousel and scorecard, the verdicts, what we got wrong and the projection)
the day after its last game. posts.json tags every post with its edition and gives each
edition's date, and --edition due sends whichever is due today (UK time).

After rendering, ready/ holds one numbered folder per post with its images in slide order and its
captions as text files. The email sends the same posts from and to the KIT_EMAIL Gmail account
(KIT_EMAIL_PASSWORD is its app password), so each post can be saved and posted from a phone.
"""
import argparse
import json
import logging
import os
import re
import shutil
import smtplib
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
from jinja2 import Environment

from plpred import video

log = logging.getLogger("plpred.social")
BRAND = "Inkwell & Algorithm"
HANDLE = "@InkwellAlgo"                 # X
IG_HANDLE = "@inkwellandalgorithm"      # Instagram
SITE_URL = os.environ.get("SITE_URL", "https://inkwellandalgorithm.com/")
# Instagram allows five hashtags a post and X rewards one or two; both treat them as labels, not reach.
HASHTAGS = "#PremierLeague #FootballAnalytics #xG"
ROUND_TAGS = "#PremierLeague #PLPredictions #FootballAnalytics #xG #EPL"
FPL_TAGS = "#FPL #FantasyPremierLeague #FPLCommunity #PremierLeague"
X_TAG = "#PremierLeague"
# The three-letter codes behind the league's matchday tags, e.g. #ARSLIV.
CODES = {"Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE", "Brighton": "BHA",
         "Burnley": "BUR", "Chelsea": "CHE", "Coventry": "COV", "Crystal Palace": "CRY", "Everton": "EVE",
         "Fulham": "FUL", "Hull": "HUL", "Ipswich": "IPS", "Leeds": "LEE", "Leicester": "LEI",
         "Liverpool": "LIV", "Luton": "LUT", "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW",
         "Nott'm Forest": "NFO", "Sheffield United": "SHU", "Southampton": "SOU", "Sunderland": "SUN",
         "Tottenham": "TOT", "West Ham": "WHU", "Wolves": "WOL"}
CLUB_TAGS = {"Arsenal": "#Arsenal", "Aston Villa": "#AVFC", "Bournemouth": "#AFCB", "Brentford": "#BrentfordFC",
             "Brighton": "#BHAFC", "Burnley": "#BurnleyFC", "Chelsea": "#ChelseaFC", "Coventry": "#PUSB",
             "Crystal Palace": "#CPFC", "Everton": "#EFC", "Fulham": "#FulhamFC", "Hull": "#HullCity",
             "Ipswich": "#ITFC", "Leeds": "#LUFC", "Leicester": "#LCFC", "Liverpool": "#LFC",
             "Man City": "#MCFC", "Man United": "#MUFC", "Newcastle": "#NUFC", "Nott'm Forest": "#NFFC",
             "Southampton": "#SaintsFC", "Sunderland": "#SAFC", "Tottenham": "#THFC", "West Ham": "#WHUFC",
             "Wolves": "#Wolves"}
SIZE = 1080
X_LIMIT = 280
X_URL_LENGTH = 23  # X counts every link as 23 characters
EDITIONS = ("predictions", "results")


# ---------------------------------------------------------------- captions

def call_phrase(v: dict) -> str:
    return f"a draw {v['score_text']}" if v["call"] == "D" else f"a {v['call_long']} {v['score_text']}"


def reason(v: dict) -> str:
    """The strongest reason as one sentence with its number, e.g. 'Arsenal concede 27% less...'."""
    if not v["front_reasons"]:
        return ""
    return v["front_reasons"][0]["text"].split(". Worth")[0].rstrip(".") + "."


def x_length(text: str) -> int:
    """Length as X counts it: each URL is 23 characters."""
    return len(text) - sum(len(u) - X_URL_LENGTH for u in re.findall(r"https?://\S+", text))


def fit_x(text: str, link: str = "") -> str:
    """Trim the end of the text until it and the link fit X's limit."""
    tail = f" {link}" if link else ""
    over = x_length(text + tail) - X_LIMIT
    if over > 0:
        text = text[: len(text) - over - 1].rstrip() + "…"
    return text + tail


def match_url(v: dict) -> str:
    return f"{SITE_URL}{v['path']}"


def club_tag(team: str) -> str:
    return CLUB_TAGS.get(team, "#" + re.sub(r"[^A-Za-z]", "", team))


def match_tag(v: dict) -> str:
    """The matchday tag X groups a game's posts under, e.g. #ARSLIV."""
    code = lambda t: CODES.get(t, re.sub(r"[^A-Za-z]", "", t)[:3].upper())
    return f"#{code(v['home_team'])}{code(v['away_team'])}"


def with_link(post: str, link: str, lead: str = "The full breakdown:") -> list[str]:
    """A post and a first reply carrying the link: link posts are the likeliest to lose reach on X."""
    return [post, f"{lead} {link}"]


def late_line(v: dict) -> str:
    """For a game called after kick-off: say so, and how it finished, so it never reads as a prediction."""
    outcome = (f" It finished {v['result_text']}, so the model was "
               f"{'right' if v['result_call'] == v['call'] else 'wrong'}.") if v.get("result_text") else ""
    return (f"Before the round, the model would have called {call_phrase(v).removeprefix('a ')} "
            f"({v['our_p']}%).{outcome} Run after kick-off, so not part of our record.")


def match_post(v: dict) -> str:
    close = ", too close to call" if v["confidence"] == "too close to call" else ""
    if v.get("after_kickoff"):
        return fit_x(f"{v['home_team']} v {v['away_team']}\n{late_line(v)}\nWhy: {reason(v)}", match_tag(v))
    return fit_x(f"{v['home_team']} v {v['away_team']}\n"
                 f"Our call: {v['call_long'] if v['call'] != 'D' else 'draw'} {v['score_text']} ({v['our_p']}%{close}).\n"
                 f"Why: {reason(v)}", match_tag(v))


def carousel_caption(v: dict) -> str:
    """The Instagram caption for one game's carousel: the call, the main reason, then five tags."""
    close = ", too close to call" if v["confidence"] == "too close to call" else f", {v['confidence']}"
    call = (late_line(v) if v.get("after_kickoff")
            else f"Our call: {call_phrase(v).removeprefix('a ')} ({v['our_p']}%{close}).")
    return (f"{v['home_team']} v {v['away_team']}, {v['kickoff_long']}.\n\n"
            f"{call}\n\n"
            f"Why: {reason(v)}\n\nSwipe for how we got there, the numbers behind it and the most likely "
            f"scores. The full breakdown is at the link in bio.\n\n"
            f"#PremierLeague {club_tag(v['home_team'])} {club_tag(v['away_team'])} #FootballAnalytics #xG")


def captions(ctx: dict, views: list[dict]) -> dict:
    mw, lead, upset, rec = ctx["matchweek"], ctx["lead"], ctx["features"]["upset"], ctx["record"]
    late = [v for v in views if v.get("after_kickoff")]
    intro = (f"Matchweek {mw}: every Premier League game called, with the data behind each call.\n"
             if not late else
             f"Matchweek {mw}, run late: {len(late)} games had already kicked off, so for those we show what the "
             f"model would have called from pre-round data, and how it went.\n")
    thread = [fit_x(intro + "The thread is below.", X_TAG)]
    thread += [match_post(v) for v in views]
    last = f"Every call, chart and reason: {SITE_URL}"
    if ctx.get("proof"):
        # The calls file's fingerprint, dated by X: proof the calls came before kick-off (plpred.proof).
        last += (f"\n\nThis week's calls have the fingerprint {ctx['proof']['short']}, the start of their "
                 f"SHA-256 hash. How to check it: {SITE_URL}record/#proof")
    thread.append(last)
    lines = "\n".join(f"{v['home_team']} v {v['away_team']}: {call_phrase(v).removeprefix('a ')} ({v['our_p']}%)"
                      + (f", finished {v['result_text']}" if v.get("result_text") else
                         ", called after kick-off" if v.get("after_kickoff") else "")
                      for v in views)
    if late:
        lines += (f"\n\nWe ran this round late, so {len(late)} of these games had already kicked off. Those calls "
                  f"use only results from before the round and aren't part of our record.")
    out = {
        "thursday_x": thread,
        "thursday_instagram": (f"Matchweek {mw}, called.\n\n{lines}\n\nEvery call comes with the reasons and the "
                               f"numbers behind it. Each game gets its own carousel, and the full breakdowns are at the "
                               f"link in bio.\n\n{ROUND_TAGS}"),
    }
    feature = upset or lead
    if upset:
        fav = max(upset["mkt"], key=upset["mkt"].get)
        fav_name = {"H": upset["home_team"], "A": upset["away_team"], "D": "a draw"}[fav]
        x = (f"The model disagrees with the market.\nBookmakers favour {fav_name} at {upset['mkt'][fav]}%. "
             f"We call {call_phrase(upset)} at {upset['our_p']}%.\nWhy: {reason(upset)}")
        ig = (f"The model v the market: {upset['home_team']} v {upset['away_team']}.\n\nBookmakers favour "
              f"{fav_name} at {upset['mkt'][fav]}%. We call {call_phrase(upset)} at {upset['our_p']}%. "
              f"Swipe for how we got there.\n\n{HASHTAGS}")
    else:
        x = (f"Match of the week: {lead['home_team']} v {lead['away_team']}.\nWe call {call_phrase(lead)} "
             f"at {lead['our_p']}%. Here is how we got there:")
        ig = (f"Match of the week: {lead['home_team']} v {lead['away_team']}.\n\nWe call {call_phrase(lead)} at "
              f"{lead['our_p']}%. The chart shows every factor that moved the number.\n\n{HASHTAGS}")
    out["friday_x"] = with_link(fit_x(x, match_tag(feature)), match_url(feature))
    out["friday_instagram"], out["friday_slug"] = ig, feature["slug"]
    if rec:
        last = rec["last"]
        season = f"Season so far: {rec['rate']}% of results right"
        if rec["fav_rate"] is not None:
            season += f", against {rec['fav_rate']}% for the bookmakers' favourite"
        body = (f"Last week: {last['right']} of {last['n']} results right, {last['exact']} exact "
                f"{'score' if last['exact'] == 1 else 'scores'}.\n{season}.")
        out["monday_x"] = with_link(fit_x(body, X_TAG), SITE_URL, "Every call, right or wrong:")
        out["monday_instagram"] = f"{body}\n\nWe publish every call, right or wrong.\n\n{HASHTAGS} #DataViz"
    return out


def results_captions(res: dict) -> dict:
    """The round's results post and one verdict post per game, scored on the calls we published."""
    mw = res["matchweek"]
    fav = f" The bookies' favourite got {res['fav_right']} of {res['fav_n']}." if res["fav_n"] else ""
    exact = f", {res['exact']} exact {'score' if res['exact'] == 1 else 'scores'}" if res["exact"] else ""
    late = (f"\n{res['late_n']} were called after kick-off from pre-round data ({res['late_right']} right), so they "
            f"aren't in our record." if res["late_n"] else "")
    body = f"Matchweek {mw}: {res['right']} of {res['n']} results right{exact}.{fav}{late}"
    lines = "\n".join(f"{r['g']['home_team']} {r['result_text']} {r['g']['away_team']}: we said "
                      f"{call_phrase(r['g']).removeprefix('a ')} ({r['g']['our_p']}%), "
                      f"{'right' if r['right'] else 'wrong'}" for r in res["rows"])
    out = {"results_x": with_link(fit_x(body, X_TAG), SITE_URL, "Every call, right or wrong:"),
           "results_instagram": f"{body}\n\n{lines}\n\nWe publish every call, right or wrong. Swipe for each game.\n\n"
                                f"{HASHTAGS} #DataViz", "verdicts": {}}
    for r in res["rows"]:
        v = r["g"]
        said = (f"Before the round, the model would have said {call_phrase(v).removeprefix('a ')} ({v['our_p']}%)"
                if v.get("after_kickoff") else f"We said {call_phrase(v).removeprefix('a ')} ({v['our_p']}%)")
        post = f"{v['home_team']} {r['result_text']} {v['away_team']}\n{said}. {r['verdict']}."
        if r["xg"]:
            post += f"\n{r['kind']}: {r['text']}"
        out["verdicts"][v["slug"]] = {
            "x": [fit_x(post, match_tag(v)), f"The full breakdown: {match_url(v)}"],
            "instagram": f"{post}\n\n#PremierLeague {club_tag(v['home_team'])} {club_tag(v['away_team'])} #xG"}
    return out


def extra_captions(ctx: dict) -> dict:
    """Captions for the projection, the FPL corner and the biggest miss, when the site has them."""
    out = {}
    sim, fpl, miss = ctx.get("sim"), ctx.get("fpl"), ctx.get("miss")
    if sim:
        top = sim["teams"][0]
        down = max(sim["teams"], key=lambda r: r["relegation"])
        moves = ctx.get("sim_moves") or []
        move = (f"\nBiggest mover: {moves[0]['team']}, {moves[0]['what']} chance "
                f"{moves[0]['change']:+d} points to {moves[0]['now']}%.") if moves else ""
        body = (f"We played out the rest of the season {sim['runs']:,} times.\n"
                f"{top['team']} win the title in {round(top['title'] * 100)}% of them. "
                f"{down['team']} go down in {round(down['relegation'] * 100)}%.{move}")
        out["projection_x"] = with_link(fit_x(body, X_TAG), f"{SITE_URL}table/", "Every club:")
        out["projection_instagram"] = f"{body}\n\nEvery club's chances at the link in bio.\n\n{HASHTAGS}"
    if fpl and fpl.get("clean_sheets"):
        cs = fpl["clean_sheets"][:3]
        lines = ", ".join(f"{r['team']} {r['cs']}%" for r in cs)
        goals = fpl["scoring"][0]
        body = (f"FPL Gameweek {ctx['matchweek']}: best clean-sheet chances, from our model: {lines}.\n"
                f"Most expected goals: {goals['team']} ({goals['xg_for']:.1f}).")
        out["fpl_x"] = with_link(fit_x(body, "#FPL"), f"{SITE_URL}fpl/",
                                 "Every club and the next six weeks:")
        out["fpl_instagram"] = f"{body}\n\nThe full fixture strip is at the link in bio.\n\n{FPL_TAGS}"
    if miss:
        body = (f"What we got wrong: {miss['match']}.\nWe said {miss['called']} at {miss['p']}%. "
                f"It finished {miss['result']}.\n{miss['kind']}: {miss['text']}")
        out["miss_x"] = [fit_x(body)]
        out["miss_instagram"] = f"{body}\n\nWe publish every call, right or wrong.\n\n{HASHTAGS}"
    return out


def newsletter(ctx: dict, views: list[dict]) -> str:
    """The week's email as Markdown, ready to paste into the newsletter service."""
    mw, lead, rec = ctx["matchweek"], ctx["lead"], ctx["record"]
    lines = [f"# Matchweek {mw}, called", "",
             f"**Match of the week: {lead['home_team']} v {lead['away_team']}.** {lead['headline']}. "
             f"We call {call_phrase(lead).removeprefix('a ')} ({lead['our_p']}%).", ""]
    for r in lead["front_reasons"]:
        lines.append(f"- **{r['title']}.** {r['text']}")
    lines += ["", f"[Read the full breakdown]({match_url(lead)})", "", "## Every call", ""]
    for v in views:
        lines.append(f"- [{v['home_team']} v {v['away_team']}]({match_url(v)}): "
                     f"{call_phrase(v).removeprefix('a ')} ({v['our_p']}%, {v['confidence']})"
                     + (f", called after kick-off, finished {v['result_text']}" if v.get("result_text") else
                        ", called after kick-off" if v.get("after_kickoff") else ""))
    strong = [v for v in views if v["confidence"] == "strong"]
    if strong:
        lines += ["", "**Our strongest calls:** " + ", ".join(
            f"{v['call_long'] if v['call'] != 'D' else 'draw'} in {v['home_team']} v {v['away_team']} ({v['our_p']}%)"
            for v in strong) + "."]
    upset = ctx["features"]["upset"]
    if upset:
        lines += ["", f"**Upset watch:** {upset['home_team']} v {upset['away_team']}. The bookmakers lean the other "
                      f"way, but we make it {upset['our_p']}% to their {upset['mkt_p']}%."]
    miss = ctx.get("miss")
    if rec:
        lines += ["", "## Last week", "",
                  f"{rec['last']['right']} of {rec['last']['n']} results right. This season: {rec['rate']}%"
                  + (f", against {rec['fav_rate']}% for the bookmakers' favourite." if rec["fav_rate"] is not None else ".")]
        if miss:
            lines += ["", f"**What we got wrong:** {miss['match']}. We said {miss['called']} ({miss['p']}%), "
                          f"it finished {miss['result']}. {miss['kind']}: {miss['text']}"]
    sim = ctx.get("sim")
    if sim:
        top = sim["teams"][0]
        lines += ["", f"**The season in {sim['runs']:,} simulations:** {top['team']} win the title in "
                      f"{round(top['title'] * 100)}% of them. [Every club's chances]({SITE_URL}table/)"]
    lines += ["", "---", "", f"Analysis, not betting advice. Follow {HANDLE} on X and {IG_HANDLE} on Instagram. "
                             f"Every call, chart and reason: {SITE_URL}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the kit

def superbru(ctx: dict, views: list[dict]) -> str:
    """The week's scores to type into our Superbru Premier League predictor entry, where the model
    plays against the public: an outside leaderboard that dates every pick before kick-off."""
    games = [v for v in views if not v.get("after_kickoff") and not v.get("result_text")]
    lines = [f"{v['home_team']} {v['score_text'].replace('–', '-')} {v['away_team']}   ({v['kickoff_text']})"
             for v in games]
    return (f"Matchweek {ctx['matchweek']}: enter these on Superbru (account InkwellAndAlgorithm) before the "
            "first kick-off. Enter them exactly, even when you disagree: the point is the model's own record.\n\n"
            + "\n".join(lines) + "\n")


def carousel(v: dict, day: str | None = None) -> list[dict]:
    """One game's slides: the call, how we got there, the numbers, the likely scores, us v the bookies."""
    teams = f"{v['home_team']} v {v['away_team']}"
    slides = [("call", teams), ("reasons", f"How we got to {v['our_p']}%: {teams}"),
              ("numbers", f"The numbers: {teams}"), ("scores", f"Most likely scores: {teams}")]
    if v["mkt"]:
        slides.append(("market", f"Us v the bookies: {teams}"))
    return [{"name": f"{kind}-{v['slug']}", "kind": kind, "g": v, "title": title,
             "day": day if kind == "call" else None} for kind, title in slides]


def uk_time(kickoff: str) -> pd.Timestamp:
    ko = pd.Timestamp(kickoff)
    return ko.tz_convert("Europe/London").tz_localize(None) if ko.tzinfo else ko


def build_kit(ctx: dict, views: list[dict], env: Environment, out: Path) -> Path:
    """Write the cards and the kit page into out (normally site/social)."""
    # The predictions edition goes out the day before the round's first game, the results edition
    # the day after its last.
    start = min(uk_time(v["kickoff"]) for v in views)
    eve = start.normalize() - pd.Timedelta(days=1)
    calls_day, match_day = eve.strftime("%A"), start.strftime("%A")
    cards_dir = out / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    card = env.get_template("social_card.html")
    base = {**ctx, "brand": BRAND, "handle": HANDLE, "ig_handle": IG_HANDLE, "size": SIZE, "root": "../.."}
    cap = {**captions(ctx, views), **extra_captions(ctx)}
    res = ctx.get("results")
    if res:
        cap.update(results_captions(res))
    feature = next(v for v in views if v["slug"] == cap["friday_slug"])
    cards = [{"name": "round", "kind": "round", "title": f"Matchweek {ctx['matchweek']}: all the calls", "day": calls_day}]
    carousels = {v["slug"]: carousel(v, calls_day) for v in views}
    for v in views:
        cards += carousels[v["slug"]]
    if ctx["record"]:
        cards.append({"name": "scorecard", "kind": "scorecard", "day": res["post_day"] if res else "Monday",
                      "title": "Last week's scorecard"})
    if res:
        cards += [{"name": "results", "kind": "results", "day": res["post_day"], "title": f"Matchweek {res['matchweek']}: the results"},
                  {"name": "results-list", "kind": "results-list", "day": res["post_day"], "title": "Every call against the result"}]
        cards += [{"name": f"verdict-{r['g']['slug']}", "kind": "verdict", "g": r["g"], "r": r, "day": res["post_day"],
                   "title": f"The verdict: {r['g']['home_team']} v {r['g']['away_team']}"} for r in res["rows"]]
    if "miss_x" in cap:
        cards.append({"name": "miss", "kind": "miss", "day": res["post_day"] if res else "Monday",
                      "title": "What we got wrong"})
    if "projection_x" in cap:
        cards.append({"name": "projection", "kind": "projection", "day": "Wednesday", "title": "How the season ends"})
    if "fpl_x" in cap:
        cards.append({"name": "fpl", "kind": "fpl", "day": calls_day, "title": "FPL: clean-sheet chances"})
    # Not posted: the default preview image for links to the site.
    cards.append({"name": "brand", "kind": "brand", "day": None, "title": BRAND})
    for c in cards:
        (cards_dir / f"{c['name']}.html").write_text(card.render(**base, card=c, g=c.get("g")))
    (out / "cards.json").write_text(json.dumps([{"name": c["name"]} for c in cards]))
    days = [
        {"day": calls_day, "when": "The day before the round", "slug": "round",
         "x": cap["thursday_x"], "instagram": cap["thursday_instagram"],
         "cards": [c for c in cards if c["kind"] in ("round", "call")],
         "how": "Post the X thread with the round card on the first post and each match card on its game's "
                "post. On Instagram, post the round card on its own; each game's carousel follows below."},
        {"day": match_day, "when": "The bold call, on the morning of the first game",
         "x": cap["friday_x"], "instagram": cap["friday_instagram"],
         "cards": [c for c in carousels[feature["slug"]] if c["kind"] in ("call", "reasons")],
         "how": "Post the reasons chart on X, then the link as the first reply. On Instagram, post the match "
                "card then the chart as a carousel."},
    ]
    days += [{"day": f"{v['home_team']} v {v['away_team']}", "slug": v["slug"],
              "when": f"Carousel · {v['kickoff_text']}" + (" · called after kick-off" if v.get("after_kickoff") else ""),
              "x": [match_post(v), f"The full breakdown: {match_url(v)}"], "instagram": carousel_caption(v),
              "cards": carousels[v["slug"]],
              "how": f"Post every slide, in order, as one Instagram carousel, any time from {calls_day} to "
                     "matchday. On X, attach the first four slides to the post and put the link in the "
                     "first reply."} for v in views]
    if "fpl_x" in cap:
        days.insert(1, {"day": f"{calls_day}, later", "when": "Before the FPL deadline", "x": cap["fpl_x"],
                        "instagram": cap["fpl_instagram"], "cards": [c for c in cards if c["kind"] == "fpl"],
                        "how": "For Fantasy Premier League players. Post it a few hours after the main thread."})
    if res:
        post_day = res["post_day"]
        wait = ("" if res["complete"] else
                f" The round isn't finished yet ({', '.join(v['home_team'] + ' v ' + v['away_team'] for v in res['pending'])}"
                f" still to play), so wait for {post_day}'s Publish site run and the carousel will cover every game.")
        days.append({"day": post_day, "when": f"Matchweek {res['matchweek']}: the results", "edition": "results", "x": cap["results_x"],
                     "instagram": cap["results_instagram"],
                     "cards": [c for c in cards if c["kind"] in ("results", "results-list", "scorecard")],
                     "how": "Post all three slides as one Instagram carousel, the day after the round's last game. "
                            "On X, attach the first two and put the link in the first reply. The game-by-game "
                            "verdicts follow below." + wait})
        days += [{"day": f"Result: {r['g']['home_team']} v {r['g']['away_team']}", "edition": "results",
                  "when": f"From {r['after_day']}, after the result",
                  "x": cap["verdicts"][r["g"]["slug"]]["x"], "instagram": cap["verdicts"][r["g"]["slug"]]["instagram"],
                  "cards": [c for c in cards if c["name"] == f"verdict-{r['g']['slug']}"],
                  "how": "One game's verdict. It's ready the morning after the game, so space these out over the "
                         "days after the round rather than posting them all at once."} for r in res["rows"]]
    elif "monday_x" in cap:
        days.append({"day": "Monday", "when": "The scorecard", "edition": "results", "x": cap["monday_x"],
                     "instagram": cap["monday_instagram"], "cards": [c for c in cards if c["kind"] == "scorecard"],
                     "how": "The site picks up results every morning, so use the kit from Monday's "
                            "Publish site run to get the weekend's results in the scorecard."})
    if "miss_x" in cap:
        days.append({"day": f"{res['post_day'] if res else 'Monday'}, later", "when": "What we got wrong",
                     "edition": "results", "x": cap["miss_x"],
                     "instagram": cap["miss_instagram"], "cards": [c for c in cards if c["kind"] == "miss"],
                     "how": "Our most confident wrong call of the round, and whether the chances backed us. "
                            "Owning the misses is what makes the hits believable."})
    if "projection_x" in cap:
        after = (pd.Timestamp(res["post_date"]) + pd.Timedelta(days=1)).strftime("%A") if res else "Wednesday"
        days.append({"day": after, "edition": "results", "when": "The season so far", "x": cap["projection_x"],
                     "instagram": cap["projection_instagram"],
                     "cards": [c for c in cards if c["kind"] == "projection"],
                     "how": "Use the kit from the newest Publish site run: the data updates every morning, "
                            "so the projection includes the latest results."})
    # A narrated video for the round and for each game still to come (rendered by plpred.video).
    videos = {v.get("slug", "round"): v for v in video.build_videos(ctx, views, env, out)}
    for d in days:
        vid = videos.get(d.get("slug"))
        if vid:
            d["video"], d["voice"] = vid["name"], video.VOICES[vid["voice"]]
    editions = {"predictions": {"matchweek": ctx["matchweek"], "date": str(eve.date())}}
    if res:
        editions["results"] = {"matchweek": res["matchweek"], "date": res["post_date"]}
    (out / "posts.json").write_text(json.dumps({"matchweek": ctx["matchweek"], "editions": editions, "posts": [
        {"day": d["day"], "when": d["when"], "how": d["how"], "x": d["x"], "instagram": d["instagram"],
         "edition": d.get("edition", "predictions"), "images": [c["name"] for c in d["cards"]],
         "video": d.get("video"), "voice": d.get("voice")} for d in days]}))
    (out / "superbru.txt").write_text(superbru(ctx, views))
    letter = newsletter(ctx, views)
    (out / "newsletter.md").write_text(letter)
    (out / "index.html").write_text(env.get_template("social_kit.html").render(
        **ctx, brand=BRAND, days=days, newsletter=letter, root="..", x_length=x_length, x_limit=X_LIMIT))
    log.info("Wrote %d social cards into %s", len(cards), out)
    return out


# ---------------------------------------------------------------- PNGs

def render_pngs(kit: Path) -> list[Path]:
    """Screenshot every card into kit/png/. Needs `pip install playwright`."""
    from playwright.sync_api import sync_playwright

    names = [c["name"] for c in json.loads((kit / "cards.json").read_text())]
    (kit / "png").mkdir(exist_ok=True)
    done = []
    with sync_playwright() as p:
        path = os.environ.get("CHROMIUM_PATH")
        browser = p.chromium.launch(executable_path=path) if path else p.chromium.launch()
        page = browser.new_page(viewport={"width": SIZE, "height": SIZE})
        for name in names:
            page.goto((kit / "cards" / f"{name}.html").resolve().as_uri())
            page.wait_for_load_state("networkidle")
            page.evaluate("document.fonts.ready")
            target = kit / "png" / f"{name}.png"
            page.screenshot(path=str(target))
            done.append(target)
        browser.close()
    log.info("Rendered %d PNGs into %s", len(done), kit / "png")
    return done


# ---------------------------------------------------------------- ready to post

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def x_text(posts: list[str]) -> str:
    """An X post, a post and its first reply, or a thread, labelled so each part is easy to copy."""
    if len(posts) == 1:
        return posts[0]
    names = ["Post", "First reply"] if len(posts) == 2 else [f"Post {i} of {len(posts)}" for i in range(1, len(posts) + 1)]
    return "\n\n".join(f"[{n}]\n{p}" for n, p in zip(names, posts))


def ready_to_post(kit: Path) -> list[Path]:
    """Copy the rendered kit into kit/ready/: one folder per post, images numbered in slide order."""
    data = json.loads((kit / "posts.json").read_text())
    ready = kit / "ready"
    shutil.rmtree(ready, ignore_errors=True)
    folders = []
    for i, post in enumerate(data["posts"], 1):
        folder = ready / f"{i:02d}-{slugify(post['day'])}"
        folder.mkdir(parents=True)
        for n, name in enumerate(post["images"], 1):
            shutil.copy(kit / "png" / f"{name}.png", folder / f"{n:02d}-{name}.png")
        (folder / "instagram.txt").write_text(post["instagram"] + "\n")
        (folder / "x.txt").write_text(x_text(post["x"]) + "\n")
        how = f"{post['when']}\n\n{post['how']}\n"
        clip = kit / "video" / f"{post.get('video')}.mp4"
        if post.get("video") and clip.exists():
            # Named after the match (or the round), with the voice, e.g. arsenal-v-everton-emma.mp4.
            what = post["video"].removeprefix("video-")
            if what == "round":
                what = f"matchweek-{data['matchweek']}-round"
            shutil.copy(clip, folder / f"{what}-{post['voice'].lower()}.mp4")
            how += (f"\nThere's also a narrated video, read by {post['voice']}. Post it as a Reel and on X "
                    f"(same captions), as well as the images. Note the voice, to see which one people prefer.\n")
        (folder / "how.txt").write_text(how)
        folders.append(folder)
    log.info("Laid out %d posts in %s", len(folders), ready)
    return folders


def due_editions(kit: Path, today) -> list[str]:
    """The editions whose day is today (a UK date)."""
    editions = json.loads((kit / "posts.json").read_text()).get("editions", {})
    return [e for e in EDITIONS if e in editions and editions[e]["date"] == str(today)]


def kit_emails(kit: Path, sender: str, to: str, editions=EDITIONS) -> list[EmailMessage]:
    """For each edition asked for, one email per post. The predictions edition starts with the Superbru picks."""
    data = json.loads((kit / "posts.json").read_text())
    folders = sorted(p for p in (kit / "ready").iterdir() if p.is_dir())
    out = []
    for edition in editions:
        mw = data.get("editions", {}).get(edition, {}).get("matchweek", data["matchweek"])
        picks = kit / "superbru.txt"
        if edition == "predictions" and picks.exists():
            msg = EmailMessage()
            msg["Subject"] = f"Matchweek {mw} · Superbru picks"
            msg["From"], msg["To"] = sender, to
            msg.set_content(picks.read_text())
            out.append(msg)
        posts = [(post, folder) for post, folder in zip(data["posts"], folders)
                 if post.get("edition", "predictions") == edition]
        out += [post_email(post, folder, f"Matchweek {mw} · {edition.title()} · {i} of {len(posts)}", sender, to)
                for i, (post, folder) in enumerate(posts, 1)]
    return out


def post_email(post: dict, folder: Path, subject: str, sender: str, to: str) -> EmailMessage:
    """One post's email: what it is, both captions to copy, and its images in slide order."""
    how = (folder / "how.txt").read_text().strip()
    msg = EmailMessage()
    msg["Subject"] = f"{subject} · {post['day']}"
    msg["From"], msg["To"] = sender, to
    msg.set_content(f"{how}\n\n"
                    f"---- INSTAGRAM ----\n\n{(folder / 'instagram.txt').read_text().strip()}\n\n"
                    f"---- X ----\n\n{(folder / 'x.txt').read_text().strip()}\n")
    for img in sorted(folder.glob("*.png")):
        msg.add_attachment(img.read_bytes(), maintype="image", subtype="png", filename=img.name)
    for clip in sorted(folder.glob("*.mp4")):
        msg.add_attachment(clip.read_bytes(), maintype="video", subtype="mp4", filename=clip.name)
    return msg


def email_kit(kit: Path, edition: str = "both") -> int:
    """Send the ready-to-post kit to KIT_EMAIL (or KIT_EMAIL_TO): one edition, both, or whichever is
    due today. Skips with a warning when the secrets are unset."""
    if edition == "due":
        editions = due_editions(kit, pd.Timestamp.now(tz="Europe/London").date())
        if not editions:
            log.info("No edition of the kit is due today, so nothing was emailed.")
            return 0
    else:
        editions = EDITIONS if edition == "both" else (edition,)
    sender, password = os.environ.get("KIT_EMAIL"), os.environ.get("KIT_EMAIL_PASSWORD")
    if not sender or not password:
        log.warning("KIT_EMAIL and KIT_EMAIL_PASSWORD are not both set, so the kit was not emailed.")
        return 0
    msgs = kit_emails(kit, sender, os.environ.get("KIT_EMAIL_TO") or sender, editions)
    with smtplib.SMTP_SSL(os.environ.get("KIT_SMTP_HOST", "smtp.gmail.com"), 465) as smtp:
        smtp.login(sender, password)
        for m in msgs:
            smtp.send_message(m)
    log.info("Emailed %d posts (%s)", len(msgs), ", ".join(editions))
    return len(msgs)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--png", type=Path, metavar="KIT",
                       help="render the kit folder written by plpred.site (normally site/social) and lay it out")
    group.add_argument("--email", type=Path, metavar="KIT", help="email a rendered kit, one message per post")
    parser.add_argument("--edition", choices=["both", "due", *EDITIONS], default="both",
                        help="with --email: which edition to send; 'due' sends the one due today, if any")
    args = parser.parse_args(argv)
    if args.png:
        render_pngs(args.png)
        ready_to_post(args.png)
    else:
        email_kit(args.email, args.edition)


if __name__ == "__main__":
    main()
