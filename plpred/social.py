"""The weekly posting kit: images and captions for X and Instagram, and the newsletter.

    python -m plpred.social --png site/social   # render the cards built by plpred.site to PNG

`plpred.site` calls `build_kit` to write the kit into site/social/: one HTML card per
image (1080 x 1080, the site's dark look) and a page listing every card with its
caption, ready to copy, plus the week's newsletter as Markdown (newsletter.md) for
pasting into the newsletter service. Rendering the cards to PNG needs Playwright and Chromium, so
it is a separate step that the Publish site workflow runs after the build.

    python -m plpred.social --email site/social # email the rendered kit, one message per post

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

from jinja2 import Environment

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
    thread.append(f"Every call, chart and reason: {SITE_URL}")
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

def carousel(v: dict) -> list[dict]:
    """One game's slides: the call, how we got there, the numbers, the likely scores, us v the bookies."""
    teams = f"{v['home_team']} v {v['away_team']}"
    slides = [("call", teams), ("reasons", f"How we got to {v['our_p']}%: {teams}"),
              ("numbers", f"The numbers: {teams}"), ("scores", f"Most likely scores: {teams}")]
    if v["mkt"]:
        slides.append(("market", f"Us v the bookies: {teams}"))
    return [{"name": f"{kind}-{v['slug']}", "kind": kind, "g": v, "title": title,
             "day": "Thursday" if kind == "call" else None} for kind, title in slides]


def build_kit(ctx: dict, views: list[dict], env: Environment, out: Path) -> Path:
    """Write the cards and the kit page into out (normally site/social)."""
    cards_dir = out / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    card = env.get_template("social_card.html")
    base = {**ctx, "brand": BRAND, "handle": HANDLE, "ig_handle": IG_HANDLE, "size": SIZE, "root": "../.."}
    cap = {**captions(ctx, views), **extra_captions(ctx)}
    feature = next(v for v in views if v["slug"] == cap["friday_slug"])
    cards = [{"name": "round", "kind": "round", "title": f"Matchweek {ctx['matchweek']}: all the calls", "day": "Thursday"}]
    carousels = {v["slug"]: carousel(v) for v in views}
    for v in views:
        cards += carousels[v["slug"]]
    if ctx["record"]:
        cards.append({"name": "scorecard", "kind": "scorecard", "day": "Monday", "title": "Last week's scorecard"})
    if "miss_x" in cap:
        cards.append({"name": "miss", "kind": "miss", "day": "Monday", "title": "What we got wrong"})
    if "projection_x" in cap:
        cards.append({"name": "projection", "kind": "projection", "day": "Wednesday", "title": "How the season ends"})
    if "fpl_x" in cap:
        cards.append({"name": "fpl", "kind": "fpl", "day": "Thursday", "title": "FPL: clean-sheet chances"})
    # Not posted: the default preview image for links to the site.
    cards.append({"name": "brand", "kind": "brand", "day": None, "title": BRAND})
    for c in cards:
        (cards_dir / f"{c['name']}.html").write_text(card.render(**base, card=c, g=c.get("g")))
    (out / "cards.json").write_text(json.dumps([{"name": c["name"]} for c in cards]))
    days = [
        {"day": "Thursday", "when": "When the calls go live",
         "x": cap["thursday_x"], "instagram": cap["thursday_instagram"],
         "cards": [c for c in cards if c["kind"] in ("round", "call")],
         "how": "Post the X thread with the round card on the first post and each match card on its game's "
                "post. On Instagram, post the round card on its own; each game's carousel follows below."},
        {"day": "Friday", "when": "The bold call",
         "x": cap["friday_x"], "instagram": cap["friday_instagram"],
         "cards": [c for c in carousels[feature["slug"]] if c["kind"] in ("call", "reasons")],
         "how": "Post the reasons chart on X, then the link as the first reply. On Instagram, post the match "
                "card then the chart as a carousel."},
    ]
    days += [{"day": f"{v['home_team']} v {v['away_team']}",
              "when": f"Carousel · {v['kickoff_text']}" + (" · called after kick-off" if v.get("after_kickoff") else ""),
              "x": [match_post(v), f"The full breakdown: {match_url(v)}"], "instagram": carousel_caption(v),
              "cards": carousels[v["slug"]],
              "how": "Post every slide, in order, as one Instagram carousel, any time from Thursday to "
                     "matchday. On X, attach the first four slides to the post and put the link in the "
                     "first reply."} for v in views]
    if "fpl_x" in cap:
        days.insert(1, {"day": "Thursday, later", "when": "Before the FPL deadline", "x": cap["fpl_x"],
                        "instagram": cap["fpl_instagram"], "cards": [c for c in cards if c["kind"] == "fpl"],
                        "how": "For Fantasy Premier League players. Post it a few hours after the main thread."})
    if "monday_x" in cap:
        days.append({"day": "Monday", "when": "The scorecard", "x": cap["monday_x"],
                     "instagram": cap["monday_instagram"], "cards": [c for c in cards if c["kind"] == "scorecard"],
                     "how": "The site picks up results every morning, so use the kit from Monday's "
                            "Publish site run to get the weekend's results in the scorecard."})
    if "miss_x" in cap:
        days.append({"day": "Monday, later", "when": "What we got wrong", "x": cap["miss_x"],
                     "instagram": cap["miss_instagram"], "cards": [c for c in cards if c["kind"] == "miss"],
                     "how": "Our most confident wrong call of the round, and whether the chances backed us. "
                            "Owning the misses is what makes the hits believable."})
    if "projection_x" in cap:
        days.append({"day": "Wednesday", "when": "The season so far", "x": cap["projection_x"],
                     "instagram": cap["projection_instagram"],
                     "cards": [c for c in cards if c["kind"] == "projection"],
                     "how": "Use the kit from the newest Publish site run: the data updates every morning, "
                            "so the projection includes the latest results."})
    (out / "posts.json").write_text(json.dumps({"matchweek": ctx["matchweek"], "posts": [
        {"day": d["day"], "when": d["when"], "how": d["how"], "x": d["x"], "instagram": d["instagram"],
         "images": [c["name"] for c in d["cards"]]} for d in days]}))
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
        (folder / "how.txt").write_text(f"{post['when']}\n\n{post['how']}\n")
        folders.append(folder)
    log.info("Laid out %d posts in %s", len(folders), ready)
    return folders


def kit_emails(kit: Path, sender: str, to: str) -> list[EmailMessage]:
    """One email per post: what it is, both captions to copy, and its images in slide order."""
    data = json.loads((kit / "posts.json").read_text())
    folders = sorted(p for p in (kit / "ready").iterdir() if p.is_dir())
    out = []
    for i, (post, folder) in enumerate(zip(data["posts"], folders), 1):
        how = (folder / "how.txt").read_text().strip()
        msg = EmailMessage()
        msg["Subject"] = f"Matchweek {data['matchweek']} · {i} of {len(folders)} · {post['day']}"
        msg["From"], msg["To"] = sender, to
        msg.set_content(f"{how}\n\n"
                        f"---- INSTAGRAM ----\n\n{(folder / 'instagram.txt').read_text().strip()}\n\n"
                        f"---- X ----\n\n{(folder / 'x.txt').read_text().strip()}\n")
        for img in sorted(folder.glob("*.png")):
            msg.add_attachment(img.read_bytes(), maintype="image", subtype="png", filename=img.name)
        out.append(msg)
    return out


def email_kit(kit: Path) -> int:
    """Send the ready-to-post kit to KIT_EMAIL (or KIT_EMAIL_TO). Skips with a warning when unset."""
    sender, password = os.environ.get("KIT_EMAIL"), os.environ.get("KIT_EMAIL_PASSWORD")
    if not sender or not password:
        log.warning("KIT_EMAIL and KIT_EMAIL_PASSWORD are not both set, so the kit was not emailed.")
        return 0
    msgs = kit_emails(kit, sender, os.environ.get("KIT_EMAIL_TO") or sender)
    with smtplib.SMTP_SSL(os.environ.get("KIT_SMTP_HOST", "smtp.gmail.com"), 465) as smtp:
        smtp.login(sender, password)
        for m in msgs:
            smtp.send_message(m)
    log.info("Emailed %d posts", len(msgs))
    return len(msgs)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--png", type=Path, metavar="KIT",
                       help="render the kit folder written by plpred.site (normally site/social) and lay it out")
    group.add_argument("--email", type=Path, metavar="KIT", help="email a rendered kit, one message per post")
    args = parser.parse_args(argv)
    if args.png:
        render_pngs(args.png)
        ready_to_post(args.png)
    else:
        email_kit(args.email)


if __name__ == "__main__":
    main()
