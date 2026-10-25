"""Short narrated videos for the posting kit: one per game and one for the round, 1080 x 1920 for Reels and X.

    python -m plpred.video site/social   # voice and render the videos written by plpred.site

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
SPEED = 1.08
LEAD, GAP, SCENE_GAP, TAIL = 0.4, 0.25, 0.45, 1.2  # seconds of silence around the lines
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
SPOKEN = {"Nott'm Forest": "Forest"}  # names as a commentator would say them
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
            pts, xp = l["points"], round(l["xpts"])
            games = COUNTS[l["games"]] if l["games"] < len(COUNTS) else l["games"]
            line = (f"{team} have {pts} points from their last {games} games. Their chances were worth {xp}."
                    if l["difference"] > 0 else
                    f"{owner(team)} chances were worth {xp} points in their last {games} games. "
                    f"They've got just {pts}.")
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
    met = meetings(v)
    if len(met) == 3 and len(set(met)) == 1:
        line = ("The last three meetings have all been draws." if met[0] == "D" else
                f"{name_of(v, met[0])} have won the last three meetings.")
        out.append(("history", 0.7, line))
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
        best = max(hooks(v), key=lambda h: h[1] - 0.45 * used.get(h[0], 0))
        used[best[0]] = used.get(best[0], 0) + 1
        picks.append(best)
    return picks


def step_line(step: dict, v: dict, say: Phrasing, n: int, k: int) -> str:
    """A reason in words a listener can check, with what it's worth to our call."""
    detail = re.sub(r"(\d+)%", r"\1 percent", step["detail"]).replace("than an average side", "than average")
    if step["label"].endswith("team news"):
        names = [x.strip() for x in detail.split(":", 1)[-1].split(",")]
        names = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        detail = f"{step['team']} are missing {names}"
    detail = spoken(detail)
    say = Phrasing(say.matchweek, say.index + k)
    pts = f"{n} {'point' if n == 1 else 'points'}"
    if step["helps_call"]:
        return say("helps", ["{d}. That adds {p}.", "{d}: worth {p}.", "{d}, which is worth {p}."],
                   d=detail, p=pts)
    return say("hurts", ["Against that, {d}. Minus {p}.", "But {d}, and that costs {p}.",
                         "Pulling the other way: {d}. Minus {p}."], d=detail, p=pts)


def flip_scene(v: dict) -> dict | None:
    """'What would change our mind', drawn as our chance now against the chance without the biggest factor."""
    text = v.get("change_our_mind") or ""
    factor = re.search(r"If (.+?) were only average", text)
    swing = re.search(r"swing to (?:a |an )?(home win|away win|draw) \((\d+)%\)", text)
    drop = re.search(r"chance would drop to (\d+)%", text)
    if not factor or not (swing or drop):
        return None
    outcome, p = (swing.group(1), int(swing.group(2))) if swing else (OUTCOME[v["call"]], int(drop.group(1)))
    f = spoken(factor.group(1))
    if swing:
        lines = [f"What would change our mind? If {f} were only average, it swings to {a_an(outcome)}."]
    else:
        lines = [f"What would change our mind? Make {f} average, and the {outcome} falls to {p} percent."]
    label = {"home win": v["home_short"] + " win", "away win": v["away_short"] + " win", "draw": "Draw"}[outcome]
    return {"kind": "flip", "lines": lines, "factor": factor.group(1), "rows": [
        {"label": f"Our call: {v['call_long']}", "v": v["our_p"], "at": "0:0.3"},
        {"label": f"If {factor.group(1)} were average: {label}", "v": p, "at": "0:2.6"}]}


def match_scenes(v: dict, matchweek=None, index: int = 0, hook=None) -> list[dict]:
    """The script for one game: an opener, the call, the reasons, one more angle, and the sign-off."""
    say = Phrasing(matchweek, index)
    home, away = spoken(v["home_team"]), spoken(v["away_team"])
    day = (v.get("kickoff_long") or "").split(" ")[0]
    hook = hook or hooks(v)[0]
    scenes = []
    if hook[0] != "fixture":
        scenes.append({"kind": "hook", "story": hook[0], "lines": [hook[2]]})

    match = say("match", ["{h} against {a}, {d}.", "{h} host {a} on {d}.", "{a} go to {h} on {d}.",
                          "{d}: {h} versus {a}."] if day else ["{h} against {a}."], h=home, a=away, d=day)
    if v["call"] == "D":
        call = say("draw", ["Our call: a draw, {s}.", "We think they share it, {s}.", "The model says a draw, {s}."],
                   s=say_score(v))
    else:
        call = say("call", ["Our call: {c}, {s}.", "We're going with {t}, {s}.", "The model says {c}, {s}.",
                            "We have it {s} to {t}."], c=call_words(v), t=name_of(v, v["call"]), s=say_score(v))
    p = v["our_p"]
    conf = {"strong": ["{p} percent. A strong call.", "We make that {p} percent, one of our strongest.",
                       "{p} percent, so we're confident on this one."],
            "too close to call": ["Only {p} percent, so it's too close to call.", "But it's tight: just {p} percent.",
                                  "{p} percent. Barely a favourite.", "Just {p} percent. It could go any way.",
                                  "{p} percent, which is as close as it gets."]}.get(
        v["confidence"], ["We make it {p} percent.", "{p} percent, a solid call.", "That's {p} percent."])
    scenes.append({"kind": "call", "lines": [match, call, say("conf." + v["confidence"], conf, p=p)]})

    rows = [dict(r) for r in v["waterfall_rows"]["rows"]]
    steps = (v.get("waterfall") or {}).get("steps", [])
    if len(rows) >= 3 and len(steps) == len(rows) - 2:
        order = sorted(range(len(steps)), key=lambda i: -rows[i + 1]["width"])[:2]
        lines = []
        rows[0]["at"] = "0:0"
        for i in order:
            step = dict(steps[i], helps_call=not rows[i + 1]["val"].startswith(("−", "-")))
            lines.append(step_line(step, v, say, int(round(abs(step["change"]))), len(lines)))
            rows[i + 1]["at"] = f"{len(lines) - 1}:0.4"
        if rows[order[0] + 1]["val"][0] not in "−-":  # "Why? Against that..." wouldn't make sense
            lines[0] = say("why", ["Why? {l}", "The big one: {l}", "{l}"], l=lines[0])
        last = len(lines) - 1
        for r in rows[1:-1]:
            r.setdefault("at", f"{last}:0")
        rows[-1]["at"] = f"{last}:1.2"
        scenes.append({"kind": "reasons", "lines": lines, "rows": rows})

    angles = []
    if v.get("mkt"):
        fav = max(v["mkt"], key=v["mkt"].get)
        us = {"H": v["h"], "D": v["d"], "A": v["a"]}
        name = "a draw" if fav == "D" else name_of(v, fav)
        agree = fav == v["call"]
        first = (say("agree", ["The bookies agree.", "The market's with us.", "The bookies see it the same way."])
                 if agree else say("disagree", ["The bookies see it differently.", "The market disagrees.",
                                                "Here's where we split from the bookies."]))
        angles.append(("market", {"kind": "market", "agree": agree, "lines": [
            first, f"They make it {v['mkt'][fav]} percent {name}, to our {us[fav]}."]}))
    xg = {(x["side"], x["kind"]): x for x in v.get("xg_rows") or []}
    if len(xg) == 4:
        ats = ("0:0.6", "0:1.6", "0:2.6", "0:3.6")
        hf, ha, af, aa = (xg[k]["v"] for k in (("home", "for"), ("home", "against"), ("away", "for"), ("away", "against")))
        both = "At home, {h} create {hf:.1f} expected goals a game and allow {ha:.1f}. Away, {a} create {af:.1f} and allow {aa:.1f}."
        pair = {"H": ["{h} create {hf:.1f} expected goals a game at home. {a} allow {aa:.1f} on the road.",
                      "On the road, {a} allow {aa:.1f} expected goals a game. At home, {h} create {hf:.1f}."],
                "A": ["{a} create {af:.1f} expected goals a game on the road. {h} allow {ha:.1f} at home.",
                      "At home, {h} allow {ha:.1f} expected goals a game. Away, {a} create {af:.1f}."],
                "D": []}[v["call"]]
        first = say("xg", pair or [both], h=home, a=away, hf=hf, ha=ha, af=af, aa=aa)
        second = None
        luck = v.get("luck") or {}
        if hook[0] != "luck" and luck.get("home") and luck.get("away"):
            side = max(("home", "away"), key=lambda s: abs(luck[s]["difference"]))
            d = luck[side]["difference"]
            team = home if side == "home" else away
            if abs(d) >= 3:
                second = (f"And {owner(team)} results flatter them by {round(d)} points." if d > 0 else
                          f"And {team} have {round(-d)} points fewer than their chances deserved.")
        if not second and hook[0] != "form":
            wins = [sum(r["result"] == "W" for r in v.get("form", {}).get(s, [])) for s in ("home", "away")]
            second = say("form", ["Form: {h} have won {wh} of five, {a} {wa}.",
                                  "In their last five, {h} won {wh} and {a} won {wa}."],
                         h=home, a=away, wh=COUNTS[wins[0]], wa=COUNTS[wins[1]])
        if second:
            angles.append(("numbers", {"kind": "numbers", "xg": [dict(xg[k], at=at) for k, at in zip(
                (("home", "for"), ("home", "against"), ("away", "for"), ("away", "against")), ats)],
                "lines": [first, second]}))
    flip = flip_scene(v)
    if flip:
        angles.append(("flip", flip))
    if angles:
        disagree = next((s for k, s in angles if k == "market" and not s["agree"]), None)
        scenes.append(disagree or say.choice(f"angle{len(angles)}", angles)[1])

    scenes.append({"kind": "outro", "lines": [
        say("outro", ["Agree? Tell us your score in the comments.", "Every call, with the reasons, at Inkwell and Algorithm dot com.",
                      "Think we've got it wrong? Tell us below.", "Follow for every call, every Thursday.",
                      "The full breakdown is at Inkwell and Algorithm dot com.", "What's your call? Comments are open."])]})
    return scenes


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
    if v["confidence"] == "strong":
        line += " " + say("r.strong", ["That's a strong one.", "We're confident there.", "A strong call."])
    return line


def round_scenes(ctx: dict, views: list[dict]) -> list[dict]:
    games = [v for v in views if videoable(v)]
    say = Phrasing(ctx["matchweek"])
    n = COUNTS[len(games)] if len(games) < len(COUNTS) else str(len(games))
    first = say("r.open", ["Matchweek {mw}, called.", "{n} games, {n} calls. Here's matchweek {mw}.",
                           "Every call for matchweek {mw}."], mw=ctx["matchweek"], n=n.capitalize())
    lines = [round_line(v, Phrasing(ctx["matchweek"], i)) for i, v in enumerate(games)]
    return [{"kind": "round", "games": games, "lines": [first] + lines},
            {"kind": "outro", "lines": [say("r.ask", ["Which one have we got wrong? Tell us below.",
                                                      "Agree? Tell us in the comments."]),
                                        "Every call, with the reasons, at Inkwell and Algorithm dot com."]}]


def build_videos(ctx: dict, views: list[dict], env: Environment, out: Path) -> list[dict]:
    """Write each video's animated page and script into out/videos/ and list them in out/videos.json."""
    folder = out / "videos"
    folder.mkdir(parents=True, exist_ok=True)
    videos = []
    if any(videoable(v) for v in views):
        videos.append({"name": "video-round", "title": f"Matchweek {ctx['matchweek']}, called",
                       "scenes": round_scenes(ctx, views)})
    games = [v for v in views if videoable(v)]
    videos += [{"name": f"video-{v['slug']}", "slug": v["slug"], "title": f"{v['home_team']} v {v['away_team']}",
                "g": v, "scenes": match_scenes(v, ctx["matchweek"], i, hook)}
               for i, (v, hook) in enumerate(zip(games, pick_hooks(games)))]
    page = env.get_template("social_video.html")
    for vid in videos:
        vid["voice"] = pick_voice(ctx["matchweek"], vid["name"])
        (folder / f"{vid['name']}.html").write_text(page.render(
            **ctx, video=vid, g=vid.get("g"), width=WIDTH, height=HEIGHT, root="../.."))
    (out / "videos.json").write_text(json.dumps([
        {"name": v["name"], "title": v["title"], "voice": v["voice"], "voice_name": VOICES[v["voice"]]["name"],
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

        res = requests.post(self.URL, headers={"X-Goog-Api-Key": self.key}, timeout=60, json={
            "input": {"text": line}, "voice": {"languageCode": "en-GB", "name": voice},
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000}})
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


def render(kit: Path, names: list[str] | None = None) -> list[Path]:
    """Voice and render every video in kit/videos.json into kit/video/<name>.mp4."""
    from playwright.sync_api import sync_playwright

    videos = json.loads((kit / "videos.json").read_text())
    if names:
        videos = [v for v in videos if v["name"] in names]
    if not videos:
        log.info("No videos to render")
        return []
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
                        help="render nothing; exit 0 if the predictions edition is emailed today, else 1")
    args = parser.parse_args(argv)
    if args.due:
        import pandas as pd

        from plpred import social
        raise SystemExit(0 if "predictions" in social.due_editions(
            args.kit, pd.Timestamp.now(tz="Europe/London").date()) else 1)
    render(args.kit, args.only)


if __name__ == "__main__":
    main()
