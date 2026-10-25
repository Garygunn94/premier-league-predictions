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


def match_scenes(v: dict) -> list[dict]:
    home, away = spoken(v["home_team"]), spoken(v["away_team"])
    conf = {"strong": f"{v['our_p']} percent. A strong call.",
            "too close to call": f"Only {v['our_p']} percent, so it's too close to call."}.get(
        v["confidence"], f"We make it {v['our_p']} percent.")
    scenes = [{"kind": "call", "lines": [f"{home} against {away}.",
                                         f"Our call: {call_words(v)}, {say_score(v)}.", conf]}]

    rows = [dict(r) for r in v["waterfall_rows"]["rows"]]
    if len(rows) >= 3:
        movers = sorted(rows[1:-1], key=lambda r: -r["width"])[:2]
        lines = ["Here's why."]
        rows[0]["at"] = "0:0.3"
        for r in movers:
            n = r["val"].lstrip("+−-")
            lines.append(f"{spoken(r['label'])} {'takes off' if r['fill'] == 'down' else 'adds'} {n}.")
            r["at"] = f"{len(lines) - 1}:0.2"
        lines.append(f"That makes it {v['our_p']} percent.")
        last = len(lines) - 1
        for r in rows[1:-1]:
            r.setdefault("at", f"{last}:0")
        rows[-1]["at"] = f"{last}:0.4"
        scenes.append({"kind": "reasons", "lines": lines, "rows": rows})

    xg = {(x["side"], x["kind"]): x for x in v.get("xg_rows") or []}
    if len(xg) == 4:
        scenes.append({"kind": "numbers", "xg": [dict(x, at=at) for x, at in (
            (xg["home", "for"], "0:0.8"), (xg["home", "against"], "0:2.4"),
            (xg["away", "for"], "1:0.6"), (xg["away", "against"], "1:1.6"))], "lines": [
            f"At home, {home} create {xg['home', 'for']['v']:.1f} expected goals a game, "
            f"and allow {xg['home', 'against']['v']:.1f}.",
            f"Away, {away} create {xg['away', 'for']['v']:.1f}, and allow {xg['away', 'against']['v']:.1f}."]})

    if v.get("mkt"):
        fav = max(v["mkt"], key=v["mkt"].get)
        us = {"H": v["h"], "D": v["d"], "A": v["a"]}
        name = "a draw" if fav == "D" else spoken(v["home_team"] if fav == "H" else v["away_team"])
        scenes.append({"kind": "market", "agree": fav == v["call"], "lines": [
            "The bookies agree." if fav == v["call"] else "The bookies see it differently.",
            f"They make it {v['mkt'][fav]} percent {name}, to our {us[fav]}."]})
    scenes.append({"kind": "outro", "lines": ["Every call, with the reasons, at Inkwell and Algorithm dot com."]})
    return scenes


def round_scenes(ctx: dict, views: list[dict]) -> list[dict]:
    games = [v for v in views if videoable(v)]
    lines = [f"{spoken(v['home_team'])} and {spoken(v['away_team'])}: "
             f"{'a draw' if v['call'] == 'D' else spoken(v['home_team'] if v['call'] == 'H' else v['away_team']) + ' to win'}."
             for v in games]
    return [{"kind": "round", "games": games, "lines": [f"Matchweek {ctx['matchweek']}, called."] + lines},
            {"kind": "outro", "lines": ["Every call, with the reasons, at Inkwell and Algorithm dot com."]}]


def build_videos(ctx: dict, views: list[dict], env: Environment, out: Path) -> list[dict]:
    """Write each video's animated page and script into out/videos/ and list them in out/videos.json."""
    folder = out / "videos"
    folder.mkdir(parents=True, exist_ok=True)
    videos = []
    if any(videoable(v) for v in views):
        videos.append({"name": "video-round", "title": f"Matchweek {ctx['matchweek']}, called",
                       "scenes": round_scenes(ctx, views)})
    videos += [{"name": f"video-{v['slug']}", "slug": v["slug"], "title": f"{v['home_team']} v {v['away_team']}",
                "g": v, "scenes": match_scenes(v)} for v in views if videoable(v)]
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
