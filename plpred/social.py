"""The weekly posting kit: images and captions for X and Instagram.

    python -m plpred.social --png site/social   # render the cards built by plpred.site to PNG

`plpred.site` calls `build_kit` to write the kit into site/social/: one HTML card per
image (1080 x 1080, the site's dark look) and a page listing every card with its
caption, ready to copy. Rendering the cards to PNG needs Playwright and Chromium, so
it is a separate step that the Publish site workflow runs after the build.
"""
import argparse
import json
import logging
import os
import re
from pathlib import Path

from jinja2 import Environment

log = logging.getLogger("plpred.social")
BRAND = "Inkwell & Algorithm"
HANDLE = "@InkwellAlgo"
SITE_URL = os.environ.get("SITE_URL", "https://garygunn94.github.io/premier-league-predictions/")
HASHTAGS = "#PremierLeague #FootballAnalytics #xG"
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
    return f"{SITE_URL}match/{v['slug']}.html"


def match_post(v: dict) -> str:
    close = ", too close to call" if v["confidence"] == "too close to call" else ""
    return fit_x(f"{v['home_team']} v {v['away_team']}\n"
                 f"Our call: {v['call_long'] if v['call'] != 'D' else 'draw'} {v['score_text']} ({v['our_p']}%{close}).\n"
                 f"Why: {reason(v)}")


def captions(ctx: dict, views: list[dict]) -> dict:
    mw, lead, upset, rec = ctx["matchweek"], ctx["lead"], ctx["features"]["upset"], ctx["record"]
    thread = [fit_x(f"Matchweek {mw}: every Premier League game called, with the data behind each call.\n"
                    f"The thread is below, and the full breakdowns are at", SITE_URL)]
    thread += [match_post(v) for v in views]
    thread.append(f"Every call, chart and reason: {SITE_URL}")
    lines = "\n".join(f"{v['home_team']} v {v['away_team']}: {call_phrase(v).removeprefix('a ')} ({v['our_p']}%)"
                      for v in views)
    out = {
        "thursday_x": thread,
        "thursday_instagram": (f"Matchweek {mw}, called.\n\n{lines}\n\nEvery call comes with the reasons and the "
                               f"numbers behind it. Full breakdowns at the link in bio.\n\n{HASHTAGS}"),
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
    out["friday_x"], out["friday_instagram"], out["friday_slug"] = fit_x(x, match_url(feature)), ig, feature["slug"]
    if rec:
        last = rec["last"]
        season = f"Season so far: {rec['rate']}% of results right"
        if rec["fav_rate"] is not None:
            season += f", against {rec['fav_rate']}% for the bookmakers' favourite"
        body = (f"Last week: {last['right']} of {last['n']} results right, {last['exact']} exact "
                f"{'score' if last['exact'] == 1 else 'scores'}.\n{season}.")
        out["monday_x"] = fit_x(f"{body}\nEvery call, right or wrong:", SITE_URL)
        out["monday_instagram"] = f"{body}\n\nWe publish every call, right or wrong.\n\n{HASHTAGS}"
    return out


# ---------------------------------------------------------------- the kit

def build_kit(ctx: dict, views: list[dict], env: Environment, out: Path) -> Path:
    """Write the cards and the kit page into out (normally site/social)."""
    cards_dir = out / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    card = env.get_template("social_card.html")
    base = {**ctx, "brand": BRAND, "handle": HANDLE, "size": SIZE, "root": "../.."}
    cap = captions(ctx, views)
    feature = next(v for v in views if v["slug"] == cap["friday_slug"])
    cards = [{"name": "round", "kind": "round", "title": f"Matchweek {ctx['matchweek']}: all the calls", "day": "Thursday"}]
    cards += [{"name": f"call-{v['slug']}", "kind": "call", "g": v, "day": "Thursday",
               "title": f"{v['home_team']} v {v['away_team']}"} for v in views]
    cards.append({"name": f"reasons-{feature['slug']}", "kind": "reasons", "g": feature, "day": "Friday",
                  "title": f"How we got to {feature['our_p']}%: {feature['home_team']} v {feature['away_team']}"})
    if ctx["record"]:
        cards.append({"name": "scorecard", "kind": "scorecard", "day": "Monday", "title": "Last week's scorecard"})
    for c in cards:
        (cards_dir / f"{c['name']}.html").write_text(card.render(**base, card=c, g=c.get("g")))
    (out / "cards.json").write_text(json.dumps([{"name": c["name"]} for c in cards]))
    days = [
        {"day": "Thursday", "when": "When the calls go live",
         "x": cap["thursday_x"], "instagram": cap["thursday_instagram"],
         "cards": [c for c in cards if c["day"] == "Thursday"],
         "how": "Post the X thread with the round card on the first post. On Instagram, post the round card "
                "and the match cards as one carousel."},
        {"day": "Friday", "when": "The bold call",
         "x": [cap["friday_x"]], "instagram": cap["friday_instagram"],
         "cards": [c for c in cards if c.get("g") is feature and c["kind"] == "call"]
                  + [c for c in cards if c["day"] == "Friday"],
         "how": "Post the reasons chart on X. On Instagram, post the match card then the chart as a carousel."},
    ]
    if "monday_x" in cap:
        days.append({"day": "Monday", "when": "The scorecard", "x": [cap["monday_x"]],
                     "instagram": cap["monday_instagram"], "cards": [c for c in cards if c["day"] == "Monday"],
                     "how": "The record covers the newest results the site has. Rebuild after the weekend's "
                            "results are in, so the scorecard includes them."})
    (out / "index.html").write_text(env.get_template("social_kit.html").render(
        **ctx, brand=BRAND, days=days, root="..", x_length=x_length, x_limit=X_LIMIT))
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


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--png", type=Path, metavar="KIT", required=True,
                        help="the kit folder written by plpred.site, normally site/social")
    args = parser.parse_args(argv)
    render_pngs(args.png)


if __name__ == "__main__":
    main()
