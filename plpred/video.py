"""Short narrated videos for the posting kit: one per game, one for the round and one for the round's
results, 1080 x 1920 for Reels and X.

    python -m plpred.video site/social           # voice and render the videos written by plpred.site
    python -m plpred.video site/social --today   # just those of the kit editions emailed today

`plpred.site` calls `build_videos` (through `plpred.social.build_kit`) to write each video's script and
an animated HTML page into social/videos/. Rendering is a separate step because it needs a voice,
Playwright and ffmpeg: each line of the script is voiced on its own, so the page knows when every
line starts, and its charts grow and count up as the narrator reaches them. Playwright captures the
page frame by frame and ffmpeg joins the frames and the voice into social/video/<name>.mp4.

The voice is Google Cloud Text-to-Speech (Chirp 3: HD), called over its REST API with the API key in
GOOGLE_TTS_API_KEY. Without it, or
if Google fails, the video is read by Kokoro instead (kokoro-onnx, free, runs on the CPU; its model,
about 350 MB, is downloaded once into KOKORO_DIR), so a video is never lost to a voice outage.

Each video is read by one of two voices, picked at random but fixed for that video and matchweek,
so we can see which one people prefer. The voice is named in the kit beside every video, and
kit/video/voices.json records the voice each video was actually read by.
"""
import base64
import io
import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from jinja2 import Environment

log = logging.getLogger("plpred.video")
# The two voices we compare: a British Google voice each, and the Kokoro voice that stands in for it.
VOICES = {"charon": {"name": "Charon", "google": "en-GB-Chirp3-HD-Charon", "kokoro": "bm_george"},
          "kore": {"name": "Kore", "google": "en-GB-Chirp3-HD-Kore", "kokoro": "bf_emma"}}
KOKORO_NAMES = {"bm_george": "George", "bf_emma": "Emma"}
WIDTH, HEIGHT, FPS = 1080, 1920, 30
# A brisk read, for Kokoro and Google: short-video viewers swipe away from a slow one.
SPEED, GOOGLE_RATE = 1.15, 1.1
# Seconds of silence around the lines. The voice starts almost at once: most viewers decide in the first second.
LEAD, GAP, SCENE_GAP, TAIL = 0.1, 0.15, 0.3, 0.9
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
SPOKEN = {"Nott'm Forest": "Forest", "Atletico Madrid": "Atlético Madrid", "Real Betis": "Betis",
          "Deportivo La Coruna": "Deportivo", "Racing Santander": "Racing Santander", "Alaves": "Alavés",
          "Leganes": "Leganés", "Malaga": "Málaga", "Cadiz": "Cádiz", "Almeria": "Almería"}  # names as a commentator would say them
# The last line of every video: why it's worth following.
FOLLOW = ["Follow, and we'll tell you if we got it right.", "Follow for the verdict after the game.",
          "Follow to see if the model gets it right."]
WORDS = ["nil", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


# ---------------------------------------------------------------- scripts

def pick_voice(matchweek, name: str) -> str:
    """A random-looking but repeatable voice: the daily rebuilds keep each video's voice."""
    digest = hashlib.sha256(f"{matchweek}/{name}".encode()).digest()
    return list(VOICES)[digest[0] % len(VOICES)]


def spoken(text: str) -> str:
    """A team name, or a line that mentions one, as it should be read out."""
    for name, say in SPOKEN.items():
        text = text.replace(name, say)
    return text


def say_score(v: dict) -> str:
    """'1–2' for an away win reads winner first: 'two-one'. Draws are 'one-all'."""
    home, away = (int(x) for x in re.split(r"\D+", v["score_text"]))
    if home == away:
        return f"{WORDS[home]}-all" if home else "nil-nil"
    hi, lo = max(home, away), min(home, away)
    return f"{WORDS[min(hi, 9)]}-{WORDS[lo]}"


def call_words(v: dict) -> str:
    if v["call"] == "D":
        return "a draw"
    team = spoken(v["home_team"] if v["call"] == "H" else v["away_team"])
    return f"{'an' if team[0] in 'AEIOU' else 'a'} {team} win"


def videoable(v: dict) -> bool:
    """Only games still to come: a video of a call made after kick-off would read as a prediction."""
    return not v.get("after_kickoff") and not v.get("result_text")


ORDINALS = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
            "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
            "eighteenth", "nineteenth", "twentieth"]
COUNTS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
OUTCOME = {"H": "home win", "D": "draw", "A": "away win"}


class Phrasing:
    """Picks wording for one video. Each slot's choice rotates with the game's place in the round, from
    a start that changes every matchweek, so neighbouring videos never share a phrasing and no week
    sounds like the last. The same matchweek always gives the same script, so rebuilds don't change it."""

    def __init__(self, matchweek, index: int = 0):
        self.matchweek, self.index = matchweek, index

    def choice(self, slot: str, options: list):
        start = hashlib.sha256(f"{self.matchweek}/{slot}".encode()).digest()[0]
        return options[(start + self.index) % len(options)]

    def __call__(self, slot: str, options: list[str], **fill) -> str:
        return self.choice(slot, options).format(**fill)


def name_of(v: dict, code: str) -> str:
    return spoken(v["home_team"] if code == "H" else v["away_team"])


def owner(team: str) -> str:
    """Leeds' and Arsenal's."""
    return team + ("'" if team.endswith("s") else "'s")


def a_an(words: str) -> str:
    return ("an " if words[0].lower() in "aeiou" else "a ") + words


def streak(results: list[dict]) -> tuple[str, int]:
    """The run a team is on, newest game last: ('W', 4) is four wins in a row."""
    codes = [r["result"] for r in results]
    if not codes:
        return "", 0
    n = next((i for i, c in enumerate(reversed(codes)) if c != codes[-1]), len(codes))
    return codes[-1], n


def meetings(v: dict, n: int = 3) -> list[str]:
    """Who won each of the last n meetings ('H', 'A' as in this game, or 'D'), newest first."""
    out = []
    for m in v.get("head_to_head", [])[:n]:
        hit = re.match(r"^(.*) (\d+)-(\d+) (.*)$", m["line"])
        if not hit:
            continue
        home, hg, ag, _ = hit.groups()
        if hg == ag:
            out.append("D")
        else:
            winner = home if int(hg) > int(ag) else hit.group(4)
            out.append("H" if winner == v["home_team"] else "A")
    return out


def hooks(v: dict) -> list[tuple[str, float, str]]:
    """Ways to open this game's video, as (kind, how strong a story it is, the line), strongest first."""
    home, away = spoken(v["home_team"]), spoken(v["away_team"])
    out = []
    if v.get("mkt"):
        fav = max(v["mkt"], key=v["mkt"].get)
        if fav != v["call"]:
            them = "a draw" if fav == "D" else f"{name_of(v, fav)} as favourites"
            out.append(("market", 0.95, f"The bookies have {them}. We don't."))
    st = v.get("standings") or {}
    if st.get("home") and st.get("away"):
        ph, pa = st["home"]["position"], st["away"]["position"]
        top, low = ("H", "A") if ph < pa else ("A", "H")
        pt, pl = min(ph, pa), max(ph, pa)
        if pt == 1:
            verb = "host" if top == "H" else "go to"
            out.append(("table", 0.8, f"Top of the table {name_of(v, top)} {verb} {name_of(v, low)}, "
                                      f"who are {ORDINALS[pl]}."))
        elif pl - pt >= 10:
            where = "bottom" if pl == 20 else ORDINALS[pl]
            out.append(("table", 0.5 + (pl - pt) / 40, f"{name_of(v, top)} are {ORDINALS[pt]}. "
                                                      f"{name_of(v, low)} are {where}."))
        elif pt >= 16:
            out.append(("table", 0.7, f"{ORDINALS[ph].capitalize()} against {ORDINALS[pa]}. "
                                      "A big one at the bottom already."))
        elif pl <= 6:
            out.append(("table", 0.6, f"{ORDINALS[ph].capitalize()} against {ORDINALS[pa]}. "
                                      "A proper top-six meeting."))
    luck = v.get("luck") or {}
    if luck.get("home") and luck.get("away"):
        side = max(("home", "away"), key=lambda s: abs(luck[s]["difference"]))
        l, team = luck[side], home if side == "home" else away
        if abs(l["difference"]) >= 4:
            gap = abs(round(l["difference"]))
            line = (f"{team} have {gap} more points than their chances deserve."
                    if l["difference"] > 0 else f"{team} deserve {gap} more points than they've got.")
            out.append(("luck", min(abs(l["difference"]) / 9, 0.9), line))
    for side, team in (("home", home), ("away", away)):
        code, n = streak(v.get("form", {}).get(side, []))
        if n >= 3 and code in "WL":
            word = "won" if code == "W" else "lost"
            line = (f"{team} have {word} their last {COUNTS[n]}." if n == 5 else
                    f"{team} have {word} {COUNTS[n]} in a row.")
            out.append(("form", 0.4 + n / 10, line))
        results = [r["result"] for r in v.get("form", {}).get(side, [])]
        if len(results) == 5 and "W" not in results:
            out.append(("form", 0.75, f"{team} haven't won in five."))
        else:
            dry = next((i for i, r in enumerate(reversed(results)) if r == "W"), len(results))
            if 3 <= dry < 5:
                out.append(("form", 0.45 + dry / 20, f"{team} haven't won in {COUNTS[dry]}."))
    met = meetings(v)
    if len(met) == 3 and len(set(met)) == 1:
        line = ("The last three meetings have all been draws." if met[0] == "D" else
                f"{name_of(v, met[0])} have won the last three meetings.")
        out.append(("history", 0.7, line))
    elif len(met) == 3 and any(met.count(c) == 2 for c in "HA"):
        out.append(("history", 0.45, f"{name_of(v, next(c for c in 'HA' if met.count(c) == 2))} have won "
                                     "two of the last three meetings."))
    ranks = v.get("ranks") or {}
    for att, dfn in (("home", "away"), ("away", "home")):
        a_team, d_team = v[f"{att}_team"], v[f"{dfn}_team"]
        attack, defence = ranks.get("attack", []), ranks.get("defence", [])
        if a_team in attack and d_team in defence:
            ra, rd, n = attack.index(a_team) + 1, defence.index(d_team) + 1, len(attack)
            if ra <= 5 and rd >= n - 4:
                out.append(("clash", 0.85, f"{spoken(a_team)} have {rank_words(ra, n, 'attack')}. "
                                           f"{spoken(d_team)} have {rank_words(rd, n, 'defence')}."))
    if st.get("home") and st.get("away") and "points" in st["home"] and "points" in st["away"]:
        gap = abs(st["home"]["points"] - st["away"]["points"])
        if gap == 0:
            out.append(("points", 0.45, f"{home} and {away} are level on points."))
        elif gap <= 2:
            out.append(("points", 0.4, f"Just {COUNTS[gap]} {'point' if gap == 1 else 'points'} "
                                       f"between {home} and {away}."))
    for n in v.get("team_news", []):
        if n.get("impact") in ("Big", "Medium"):
            team = spoken(n["team"])
            verb = "could be without" if n["status"].startswith("Doubtful") else "are without"
            out.append(("news", 0.75 if n["impact"] == "Big" else 0.45, f"{team} {verb} {n['player']}."))
            break
    if v["confidence"] == "too close to call":
        out.append(("close", 0.25, f"{home} or {away}? Our model can barely split them."))
    if not v["headline"].startswith("Nothing to split"):
        out.append(("headline", 0.3, spoken(v["headline"]) + "."))
    out.append(("fixture", 0.1, f"{home} against {away}."))
    return sorted(out, key=lambda h: -h[1])


def pick_hooks(games: list[dict]) -> list[tuple[str, float, str]]:
    """One opener per game, mixing the kinds across the round: a kind already used counts for less."""
    used, picks = {}, []
    for v in games:
        # a bare fixture is the last resort: it gives nobody a reason to stay
        best = max(hooks(v), key=lambda h: -1 if h[0] == "fixture" else h[1] - 0.45 * used.get(h[0], 0))
        used[best[0]] = used.get(best[0], 0) + 1
        picks.append(best)
    return picks


def rank_words(rank: int, n: int, kind: str) -> str:
    """'the fifth-best attack', 'the worst defence': a place in the league anyone can picture."""
    best = rank <= (n + 1) // 2
    k = rank if best else n + 1 - rank
    return f"the {'' if k == 1 else ORDINALS[k] + '-'}{'best' if best else 'worst'} {kind}"


def ladder(order: list[str], picks: list[str], end: str | None) -> list[dict]:
    """At most five places of a ranking with the picked clubs in them: from the top for a strength,
    from the bottom for a weakness, around the clubs when comparing two."""
    from plpred.site import SHORT
    n, at = len(order), sorted(order.index(t) for t in picks)
    if end == "top":
        keep = list(range(min(5, n))) if at[-1] < 5 else [0, 1, 2, *at]
    elif end == "bottom":
        keep = list(range(max(0, n - 5), n)) if at[0] >= n - 5 else [*at, n - 3, n - 2, n - 1]
    elif at[-1] - at[0] < 5:
        first = max(0, min(at[0] - (4 - (at[-1] - at[0])) // 2, n - 5))
        keep = list(range(first, min(first + 5, n)))
    else:
        keep = at
    keep = sorted(set(keep))
    return [{"rank": i + 1, "team": SHORT.get(order[i], order[i]), "hot": order[i] in picks,
             "gap": k > 0 and i != keep[k - 1] + 1} for k, i in enumerate(keep)]


def news_names(step: dict) -> str:
    names = [x.strip() for x in step["detail"].split(":", 1)[-1].split(",")]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def old_reason(step: dict) -> str:
    """Calls made before the ranks were kept: the reason in words, without its percentage."""
    more = " more " in step["detail"]
    return ({"attack": ["{t} score more than most.", "{t} struggle to score."],
             "defence": ["{t} concede more than most.", "{t} concede less than most."]}[step["kind"]]
            [0 if more else 1]).format(t=spoken(step["team"]))


def main_reason(v: dict) -> dict | None:
    """The biggest thing behind the call that a fan can check, as {kind, team, text, ...}. A rating is
    said as a place in the league, and only when that place points the same way as the call and is
    striking: our side in the top six, or the other side in the bottom six. 'Everton have the eighth-best defence' can't be the
    reason we back Newcastle, even if Everton concede a little more than an average side. When no
    single rating says it plainly, the reason is how the model rates the two clubs overall."""
    ranks = v.get("ranks")
    steps = [s for s in (v.get("waterfall") or {}).get("steps", []) if s["change"] > 0]
    ours = {"H": v["home_team"], "A": v["away_team"]}.get(v["call"])
    if ours and steps:
        if not ranks:  # calls made before the ranks were kept
            top = steps[0]
            if top["kind"] == "news":
                return {"kind": "news", "team": top["team"], "text": f"{spoken(top['team'])} are missing {news_names(top)}."}
            return {"kind": "old", "team": top["team"], "text": old_reason(top)}
        for step in steps:
            if step["kind"] == "news":
                return {"kind": "news", "team": step["team"],
                        "text": f"{spoken(step['team'])} are missing {news_names(step)}."}
            order = ranks.get(step["kind"], [])
            if step["team"] not in order:
                continue
            rank, n = order.index(step["team"]) + 1, len(order)
            words = rank_words(rank, n, step["kind"])
            # near the right end of the table: 'the ninth-best attack' explains nothing
            if words.split()[1].endswith("best") == (step["team"] == ours) and min(rank, n + 1 - rank) <= 6:
                return {"kind": step["kind"], "team": step["team"], "words": words, "order": order,
                        "text": f"{spoken(step['team'])} have {words} in the league."}
    if not ranks:
        return None
    order = ranks["overall"]
    h, a = (ORDINALS[order.index(t) + 1] for t in (v["home_team"], v["away_team"]))
    level = v["call"] == "D" or abs(order.index(v["home_team"]) - order.index(v["away_team"])) <= 2
    return {"kind": "overall", "order": order, "level": level,
            "h": h, "a": a, "text": f"Our model rates {spoken(v['home_team'])} {h} and {spoken(v['away_team'])} {a}"
                                   + (": almost level." if level else " overall.")}


def reason_text(v: dict) -> str | None:
    """The call's main reason as a fan would say it, for the captions and cards: 'Arsenal have the
    best defence in the league.' rather than '...26% less than an average side, worth 6 points'."""
    r = main_reason(v)
    return r["text"] if r else None


def why_scene(v: dict, say: Phrasing) -> dict | None:
    """The one reason a viewer can repeat, from main_reason, with its place in the league on screen."""
    r = main_reason(v)
    if not r:
        return None
    if r["kind"] == "overall":
        home, away = spoken(v["home_team"]), spoken(v["away_team"])
        if r["level"]:
            line = say("level", ["Our model rates {h} {ph} and {a} {pa}. Almost level.",
                                 "{h} {ph}, {a} {pa} in our ratings. Hard to split."],
                       h=home, a=away, ph=r["h"], pa=r["a"])
            head = "Almost level."
        else:
            line = say("overall", ["Simple: our model rates {h} {ph} and {a} {pa}.",
                                   "The reason: {h} are {ph} in our ratings, {a} {pa}."],
                       h=home, a=away, ph=r["h"], pa=r["a"])
            head = "The better side."
        return {"kind": "why", "lines": [line], "head": head, "sub": "Our overall ratings",
                "ladder": ladder(r["order"], [v["home_team"], v["away_team"]], None)}
    if r["kind"] == "news":
        return {"kind": "why", "lines": [say("why.news", ["The reason: {l}", "It's about who's missing. {l}"],
                                             l=r["text"])], "head": r["text"]}
    if r["kind"] == "old":
        return {"kind": "why", "lines": [say("why.old", ["The reason: {d}", "The big one: {d}"], d=r["text"])],
                "head": r["text"]}
    words = r["words"]
    line = say("why", ["The reason: {t} have {w} in the league.", "It comes down to this: {t} have {w} in the league.",
                       "Why? {t} have {w} in the league."], t=spoken(r["team"]), w=words)
    return {"kind": "why", "lines": [line], "head": f"{words[0].upper()}{words[1:]}.",
            "sub": f"{'Attack' if r['kind'] == 'attack' else 'Defence'}: the model's ratings now",
            "ladder": ladder(r["order"], [r["team"]], "top" if words.split()[1].endswith("best") else "bottom")}


def step_fact(v: dict, step: dict) -> str | None:
    """One factor behind the call as a fact a fan can check, or None when it's too small to say: a
    place in the league when the ranks are kept and it's striking, otherwise how far from average."""
    team = spoken(step["team"])
    if step["kind"] == "news":
        return f"{team} are missing {news_names(step)}." if abs(step["change"]) >= 1.5 else None
    order = (v.get("ranks") or {}).get(step["kind"], [])
    if step["team"] in order:
        rank, n = order.index(step["team"]) + 1, len(order)
        if min(rank, n + 1 - rank) <= 6:
            return f"{team} have {rank_words(rank, n, step['kind'])} in the league."
    pct = re.search(r"(\d+)%", step["detail"])
    pct, more = (int(pct.group(1)) if pct else 0), " more " in step["detail"]
    if pct < 10 or step["kind"] not in ("attack", "defence"):
        return None
    far = "far " if pct >= 25 else ""
    phrase = ({True: f"score {far}more than most", False: "struggle to score"} if step["kind"] == "attack" else
              {True: f"concede {far}more than most", False: "are hard to score against"})[more]
    return f"{team} {phrase}."


def clues(v: dict, hook_kind: str) -> list[str]:
    """Up to three facts before the reveal: the two biggest things behind the call, then the biggest
    one against it when the call is close, topped up with form or the table so no reel is thin."""
    steps = sorted((v.get("waterfall") or {}).get("steps", []), key=lambda st: -abs(st["change"]))
    pro = [f for f in (step_fact(v, st) for st in steps if st["change"] > 0) if f][:2]
    con = next((f for f in (step_fact(v, st) for st in steps if st["change"] <= -3) if f), None)
    out = pro + ([f"But {con}"] if con and pro and v["confidence"] != "strong" else [])
    home, away = spoken(v["home_team"]), spoken(v["away_team"])
    extra = []
    form = v.get("form") or {}
    if hook_kind != "form" and len(form.get("home", [])) == 5 and len(form.get("away", [])) == 5:
        wh, wa = (sum(r["result"] == "W" for r in form[s]) for s in ("home", "away"))
        if abs(wh - wa) >= 2:
            (bt, bw), (wt, ww) = sorted(((home, wh), (away, wa)), key=lambda x: -x[1])
            extra.append(f"{bt} have won {COUNTS[bw]} of their last five. {wt}, {COUNTS[ww] if ww else 'none'}.")
    st = v.get("standings") or {}
    if hook_kind not in ("table", "points") and st.get("home") and st.get("away"):
        extra.append(f"In the table, {home} are {ORDINALS[st['home']['position']]} "
                     f"and {away} {ORDINALS[st['away']['position']]}.")
    for line in extra:
        if len(out) < 3:
            out.append(line)
    return out[:3]


def tease(hook: tuple, v: dict, say: Phrasing) -> str:
    """The second line of a reveal's opener: the question the hook raises, and that the call answers."""
    kind, line = hook[0], hook[2]
    question = {
        "market": "Who's right?",
        "table": ("Who blinks first?" if "at the bottom" in line else
                  "Who comes out on top?" if "top-six" in line else "As one-sided as it looks?"),
        "luck": "Is their luck about to turn?" if "deserve " in line else "Can they keep it up?",
        "form": "Can anyone stop them?" if " won " in line and "haven't" not in line else "Is this where it turns?",
        "history": "Another draw?" if "draws" in line else "Again?",
        "news": "How much does it hurt?",
        "clash": "Something has to give.",
        "points": "Who pulls away?",
        "close": "Three clues, then our call.",
    }.get(kind)
    end = say("tease.end", ["Our call's at the end.", "Guess, then watch.", "Our call comes last."])
    if question and question.endswith("our call."):
        return question
    return f"{question} {end}" if question else say("tease", ["Who do we back? Our call's at the end.",
                                                              "Guess first. Our call comes at the end.",
                                                              "Make your pick. Ours is at the end."])


STYLES = ("reveal", "upfront")


def pick_style(matchweek, index: int) -> str:
    """Every game keeps our call for the end ("reveal"): the wait for it is a reason to watch to the
    end. "upfront" (the call straight after the opener) stays available to match_scenes."""
    return "reveal"


def match_scenes(v: dict, matchweek=None, index: int = 0, hook=None, style: str | None = None) -> list[dict]:
    """The script for one game, one idea a scene. Most viewers decide in the first second, so it always
    opens on its hook, on screen from frame one. A "reveal" then teases the call, gives the one reason
    as a clue and makes the call last; "upfront" makes the call straight after the hook. The numbers
    behind it (percentages, expected goals, the bookies) stay in the caption and on the site."""
    say = Phrasing(matchweek, index)
    style = style or pick_style(matchweek, index)
    reveal = style == "reveal"
    home, away = spoken(v["home_team"]), spoken(v["away_team"])
    day = (v.get("kickoff_long") or "").split(" ")[0]
    hook = hook or hooks(v)[0]
    if reveal and hook[0] == "headline":  # the headline gives our call away
        hook = next(h for h in hooks(v) if h[0] != "headline")
    if hook[0] == "fixture":
        hook = (hook[0], hook[1], say("match", ["{h} against {a}, {d}.", "{h} host {a} on {d}.", "{a} go to {h} on {d}.",
                                                "{d}: {h} versus {a}."] if day else ["{h} against {a}."],
                                      h=home, a=away, d=day))
    opener = [hook[2]]
    if reveal:
        opener.append(tease(hook, v, say))
    scenes = [{"kind": "hook", "story": hook[0], "lines": opener}]

    if v["call"] == "D":
        call = say("draw", ["So, our call: a draw, {s}.", "Our call? A draw, {s}."] if reveal else
                   ["Our call: a draw, {s}.", "We think they share it, {s}."], s=say_score(v))
    else:
        call = say("call", ["So, our call: {c}, {s}.", "Our call? {t}, {s}.", "The verdict: {c}, {s}."] if reveal else
                   ["Our call: {c}, {s}.", "We're going with {t}, {s}.", "We have it {s} to {t}."],
                   c=call_words(v), t=name_of(v, v["call"]), s=say_score(v))
    # A hook about draws, or about the bookies, is answered by the call itself.
    if hook[0] == "history" and "draws" in hook[2] and v["call"] != "D":
        call = ("So this time, we lean " if reveal else "This time, we lean ") + name_of(v, v["call"]) + f", {say_score(v)}."
    conf = {"strong": ["A strong call.", "One of our strongest this week.", "We're confident on this one."],
            "too close to call": ["But only just.", "It's close, though.", "It could go any way."]}.get(
        v["confidence"], ["Favourites, not certainties.", "A solid call.", "Clear, but not certain."])
    call_scene = {"kind": "call", "lines": [call, say("conf." + v["confidence"], conf)], "call_line": 0}
    if reveal:
        facts = clues(v, hook[0])
        if facts:
            scenes.append({"kind": "clues", "lines": facts})
        scenes.append(call_scene)
    else:
        scenes += [s for s in (call_scene, why_scene(v, say)) if s]
    asks = (["Did you guess right? Tell us below.", "Same pick as you? Tell us below."] if reveal else
            ["Agree? Tell us your score.", "Think we've got it wrong? Tell us below.",
             "What's your score? Comments are open."])
    scenes.append(outro(say, asks))
    return scenes


def outro(say: Phrasing, asks: list[str]) -> dict:
    """The sign-off: ask for a comment, then give a reason to follow. Comments push a reel's reach;
    the follow line is the promise that brings people back."""
    return {"kind": "outro", "lines": [say("ask", asks), say("follow", FOLLOW)]}


def round_line(v: dict, say: Phrasing) -> str:
    home, away = spoken(v["home_team"]), spoken(v["away_team"])
    if v["call"] == "D":
        line = say("r.draw", ["{h} and {a} to draw.", "{h} and {a} to share the points.", "A draw at {h}, against {a}."],
                   h=home, a=away)
    elif v["call"] == "H":
        line = say("r.home", ["{h} to beat {a}.", "{h} at home to {a}: a home win.", "{h} over {a}."],
                   h=home, a=away)
    else:
        line = say("r.away", ["{a} to win at {h}.", "{a} to beat {h}.", "An away win for {a} at {h}."],
                   h=home, a=away)
    return line


def round_scenes(ctx: dict, views: list[dict]) -> list[dict]:
    """Every call on screen, but only the strong ones read out: nobody holds ten calls in their head."""
    games = [v for v in views if videoable(v)]
    say = Phrasing(ctx["matchweek"])
    n = COUNTS[len(games)] if len(games) < len(COUNTS) else str(len(games))
    first = say("r.open", ["Matchweek {mw}, called.", "{n} games, {m} calls. Here's matchweek {mw}.",
                           "Every call for matchweek {mw}."], mw=ctx["matchweek"], n=n.capitalize(), m=n)
    league = ctx.get("league") or {}
    if league and not league.get("main", True):
        first = f"{league['name']}. {first}"
    strong = [v for v in games if v["confidence"] == "strong"]
    if strong:
        picks = [round_line(v, Phrasing(ctx["matchweek"], i)) for i, v in enumerate(strong[:3])]
        lines = [first, ("Our strongest call: " if len(picks) == 1 else "Our strongest calls: ") + " ".join(picks),
                 "The rest are on screen."]
    else:
        lines = [first, "No strong calls this week. It's a tight round.", "Every call is on screen."]
    return [{"kind": "round", "games": games, "lines": lines},
            outro(say, ["Which one have we got wrong? Tell us below.", "Agree? Tell us in the comments."])]


def results_scenes(res: dict, ctx: dict) -> list[dict]:
    """The round's verdict: how many we got right, against the bookies, the best call and the miss."""
    say = Phrasing(res["matchweek"], 7)
    live = [r for r in res["rows"] if not r["g"].get("after_kickoff")]
    right, n = sum(r["right"] for r in live), len(live)
    exact = sum(r["exact"] for r in live)
    lines = [f"Matchweek {res['matchweek']}: we got {COUNTS[right] if right < len(COUNTS) else right} "
             f"of {COUNTS[n] if n < len(COUNTS) else n} right."
             + (f" {'One' if exact == 1 else COUNTS[exact].capitalize() if exact < len(COUNTS) else exact} "
                f"exact {'score' if exact == 1 else 'scores'}, too." if exact else "")]
    priced = [r for r in live if r["fav_right"] is not None]
    if priced:
        fav = sum(r["fav_right"] for r in priced)
        lines.append(f"The bookies' favourite got {COUNTS[fav] if fav < len(COUNTS) else fav}.")
    hits = [r for r in live if r["right"]]
    if hits:
        r = max(hits, key=lambda r: (r["exact"], r["g"]["our_p"]))
        g = r["g"]
        score = say_score({"score_text": r["result_text"]})
        lines.append(f"Best call: {spoken(g['home_team'])} against {spoken(g['away_team'])}. "
                     + (f"We said {score}, and it finished {score}." if r["exact"] else
                        f"We had {call_words(g)}, and it finished {score}."))
    misses = [r for r in live if not r["right"]]
    if misses:
        r = max(misses, key=lambda r: r["g"]["our_p"])
        g = r["g"]
        verdict = {"Unlucky": " The chances backed us, though.",
                   "Our mistake": " The chances say we misread it."}.get(r.get("kind"), "")
        lines.append(f"The miss: {spoken(g['home_team'])} against {spoken(g['away_team'])}. We had {call_words(g)}. "
                     f"It finished {say_score({'score_text': r['result_text']})}.{verdict}")
    rec = ctx.get("record")
    if rec and rec.get("rate") is not None:
        lines.append(f"Season so far: {rec['rate']} percent right.")
    return [{"kind": "results", "rows": res["rows"], "right": right, "n": n, "lines": lines},
            outro(say, ["How did your predictions do? Tell us below.", "Did you call it better? Tell us below."])]


def build_videos(ctx: dict, views: list[dict], env: Environment, out: Path) -> list[dict]:
    """Write each video's animated page and script into out/videos/ and list them in out/videos.json."""
    folder = out / "videos"
    folder.mkdir(parents=True, exist_ok=True)
    videos = []
    if any(videoable(v) for v in views):
        videos.append({"name": "video-round", "title": f"Matchweek {ctx['matchweek']}, called",
                       "scenes": round_scenes(ctx, views)})
    # In La Liga only the featured games get a Reel (see plpred.leagues.featured).
    games = [v for v in views if videoable(v) and v.get("featured", True)]
    videos += [{"name": f"video-{v['slug']}", "slug": v["slug"], "title": f"{v['home_team']} v {v['away_team']}",
                "g": v, "style": pick_style(ctx["matchweek"], i),
                "scenes": match_scenes(v, ctx["matchweek"], i, hook, pick_style(ctx["matchweek"], i))}
               for i, (v, hook) in enumerate(zip(games, pick_hooks(games)))]
    for vid in videos:
        if vid.get("style"):
            vid["hook"] = vid["scenes"][0]["lines"][0]
    res = ctx.get("results")
    if res and res["complete"] and any(not r["g"].get("after_kickoff") for r in res["rows"]):
        videos.append({"name": "video-results", "slug": "results", "edition": "results",
                       "matchweek": res["matchweek"], "title": f"Matchweek {res['matchweek']}: the verdict",
                       "scenes": results_scenes(res, ctx)})
    page = env.get_template("social_video.html")
    for vid in videos:
        vid["voice"] = pick_voice(vid.get("matchweek", ctx["matchweek"]), vid["name"])
        (folder / f"{vid['name']}.html").write_text(page.render(
            **{**ctx, "matchweek": vid.get("matchweek", ctx["matchweek"])},
            video=vid, g=vid.get("g"), width=WIDTH, height=HEIGHT,
            root="/".join([".."] * (2 + (ctx.get("league") or {}).get("prefix", "").count("/")))))
    (out / "videos.json").write_text(json.dumps([
        {"name": v["name"], "title": v["title"], "edition": v.get("edition", "predictions"),
         "voice": v["voice"], "voice_name": VOICES[v["voice"]]["name"], "style": v.get("style"),
         "scenes": [{"kind": s["kind"], "lines": s["lines"]} for s in v["scenes"]]} for v in videos], indent=1))
    log.info("Wrote %d video scripts into %s", len(videos), folder)
    return videos


# ---------------------------------------------------------------- rendering

def model_dir() -> Path:
    """The Kokoro model files, downloaded on first use (the Publish site workflow caches the folder)."""
    d = Path(os.environ.get("KOKORO_DIR") or Path.home() / ".cache" / "kokoro")
    d.mkdir(parents=True, exist_ok=True)
    for f in MODEL_FILES:
        if not (d / f).exists():
            log.info("Downloading %s", f)
            tmp = d / f"{f}.part"
            urllib.request.urlretrieve(MODEL_URL + f, tmp)
            tmp.rename(d / f)
    return d


class Google:
    """Google Cloud Text-to-Speech. Chirp 3: HD has 1M free characters a month; a week's videos use
    about 5,000."""

    URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

    def __init__(self, key: str):
        self.key = key

    def speak(self, line: str, voice: str):
        import requests
        import soundfile as sf

        body = {"input": {"text": line}, "voice": {"languageCode": "en-GB", "name": voice},
                "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000, "speakingRate": GOOGLE_RATE}}
        res = requests.post(self.URL, headers={"X-Goog-Api-Key": self.key}, timeout=60, json=body)
        if res.status_code == 400 and "peaking" in res.text:  # should a voice refuse a rate, read at its own
            del body["audioConfig"]["speakingRate"]
            res = requests.post(self.URL, headers={"X-Goog-Api-Key": self.key}, timeout=60, json=body)
        if not res.ok:  # Google's message says what is wrong (API not enabled, key restricted, ...)
            raise RuntimeError(f"HTTP {res.status_code}: {res.text[:300]}")
        audio, sr = sf.read(io.BytesIO(base64.b64decode(res.json()["audioContent"])), dtype="float32")
        return audio, sr


class Kokoro:
    """The free fallback, loaded only when a video needs it."""

    def __init__(self):
        from kokoro_onnx import Kokoro as Model
        d = model_dir()
        self.model = Model(str(d / MODEL_FILES[0]), str(d / MODEL_FILES[1]))

    def speak(self, line: str, voice: str):
        return self.model.create(line, voice=voice, speed=SPEED, lang="en-gb")


def narrate(speak, scenes: list[dict], wav: Path) -> dict:
    """Voice every line with speak(line) -> (audio, rate), join them with short pauses, and time
    each scene and line."""
    import numpy as np
    import soundfile as sf

    sr, t, parts = 24000, LEAD, []
    silence = lambda s: np.zeros(int(s * sr), dtype=np.float32)  # noqa: E731
    parts.append(silence(LEAD))
    timed = []
    for i, sc in enumerate(scenes):
        start = 0.0 if i == 0 else t - SCENE_GAP / 2
        lines = []
        for line in sc["lines"]:
            audio, sr = speak(line)
            lines.append({"text": line, "start": round(t, 3), "end": round(t + len(audio) / sr, 3)})
            parts += [audio.astype(np.float32), silence(GAP)]
            t += len(audio) / sr + GAP
        parts.append(silence(SCENE_GAP))
        t += SCENE_GAP
        timed.append({"start": round(start, 3), "end": round(t - SCENE_GAP / 2, 3), "timed": lines})
    parts.append(silence(TAIL))
    duration = round(t + TAIL, 3)
    timed[-1]["end"] = duration
    sf.write(wav, np.concatenate(parts), sr)
    return {"duration": duration, "scenes": timed}


def render(kit: Path, names: list[str] | None = None, editions: list[str] | None = None) -> list[Path]:
    """Voice and render every video in kit/videos.json (or just those named, or of those editions)
    into kit/video/<name>.mp4."""
    videos = json.loads((kit / "videos.json").read_text())
    if names:
        videos = [v for v in videos if v["name"] in names]
    if editions is not None:
        videos = [v for v in videos if v.get("edition", "predictions") in editions]
    if not videos:
        log.info("No videos to render")
        return []
    from playwright.sync_api import sync_playwright

    google = kokoro = None
    if os.environ.get("GOOGLE_TTS_API_KEY"):
        google = Google(os.environ["GOOGLE_TTS_API_KEY"])
    else:
        log.info("GOOGLE_TTS_API_KEY is not set, so Kokoro reads the videos")
    (kit / "video").mkdir(exist_ok=True)
    used_file = kit / "video" / "voices.json"
    used = json.loads(used_file.read_text()) if used_file.exists() else {}
    done = []
    with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
        path = os.environ.get("CHROMIUM_PATH")
        browser = p.chromium.launch(executable_path=path) if path else p.chromium.launch()
        for vid in videos:
            work = Path(tmp) / vid["name"]
            (work / "frames").mkdir(parents=True)
            voice = VOICES[vid["voice"]]
            timeline = None
            if google:
                try:
                    timeline = narrate(lambda line: google.speak(line, voice["google"]), vid["scenes"],
                                       work / "voice.wav")
                    used[vid["name"]] = voice["name"]
                except Exception as e:
                    log.warning("Google Text-to-Speech failed on %s (%s), so Kokoro reads it", vid["name"], e)
            if timeline is None:
                kokoro = kokoro or Kokoro()
                timeline = narrate(lambda line: kokoro.speak(line, voice["kokoro"]), vid["scenes"],
                                   work / "voice.wav")
                used[vid["name"]] = KOKORO_NAMES[voice["kokoro"]]
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
            page.add_init_script(f"window.TIMELINE = {json.dumps(timeline)};")
            page.goto((kit / "videos" / f"{vid['name']}.html").resolve().as_uri())
            page.wait_for_load_state("networkidle")
            page.evaluate("document.fonts.ready")
            for i in range(int(timeline["duration"] * FPS) + 1):
                page.evaluate("t => window.render(t)", i / FPS)
                page.screenshot(path=str(work / "frames" / f"{i:05d}.jpg"), type="jpeg", quality=92)
            page.close()
            target = kit / "video" / f"{vid['name']}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                            "-i", str(work / "frames" / "%05d.jpg"), "-i", str(work / "voice.wav"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
                            "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(target)],
                           check=True)
            shutil.rmtree(work)
            log.info("Rendered %s (%.0fs, %s)", target.name, timeline["duration"], used[vid["name"]])
            done.append(target)
        browser.close()
    used_file.write_text(json.dumps(used, indent=1))
    return done


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kit", type=Path, help="the kit folder written by plpred.site (normally site/social)")
    parser.add_argument("--only", nargs="*", metavar="NAME", help="render just these videos, e.g. video-round")
    parser.add_argument("--due", action="store_true",
                        help="render nothing; exit 0 if an edition of the kit is emailed today, else 1")
    parser.add_argument("--today", action="store_true",
                        help="render only the videos of the editions emailed today")
    args = parser.parse_args(argv)
    if args.due or args.today:
        import pandas as pd

        from plpred import social
        due = social.due_editions(args.kit, pd.Timestamp.now(tz="Europe/London").date())
        if args.due:
            raise SystemExit(0 if due else 1)
    render(args.kit, args.only, due if args.today else None)


if __name__ == "__main__":
    main()
