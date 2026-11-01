"""How it works: one narrated 9:16 reel on how Inkwell & Algorithm calls a game, for the posting kit.

    python -m plpred.howitworks out/how-it-works              # write, voice and render the reel
    python -m plpred.howitworks out/how-it-works --scripts    # write the script and page, don't render
    python -m plpred.howitworks out/how-it-works --email      # and email the video and captions

The reel walks through the whole process: the data (three seasons, recent games weighted most), what
is changing in the league this season, the model, one of our latest calls worked through step by
step, how a percentage becomes a call, and our honest record against the bookies.

The process doesn't change, but most of the numbers do. The league trends are the stats whose
this-season average is furthest from the two seasons before it (measured in standard errors, so a
blip over a handful of games isn't picked), the "results v performance" scene picks this season's
most extreme sides, and the worked example is the most confident call in the latest round. Re-run it
every few months and the reel tells the season's new story. Every sentence is a template filled from
these numbers, like the weekly and Club DNA videos, and none of it is phrased as betting advice.

The folder gets the script and animated page, the video (rendered by plpred.video.render, so it uses
the same voices) and ready/how-it-works/ with the video and both captions. With --email they're
emailed as one message, from and to KIT_EMAIL like the weekly posting kit.
"""
import argparse
import json
import logging
import os
import shutil
import smtplib
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from plpred import dna, explain, luck, model, predict, seasons, social, video
from plpred.dna import ORD_WORDS
from plpred.site import SHORT, TEMPLATES, record

log = logging.getLogger("plpred.howitworks")

NAME = "how-it-works"
TAGS = "#PremierLeague #FootballAnalytics #xG #DataScience"
MIN_GAMES = 30     # this season needs this many games before its trends are worth a scene
MIN_Z = 1.5        # a trend must be this many standard errors from the last two seasons
TRENDS = 2

# League-wide stats a trend can be about, one value per match:
# (key, how it's shown, decimals, percent?, rising sentence, falling sentence). {now} and {then} are filled in.
LEAGUE = [
    ("goals", "Goals a game", 2, False, "Goals are up: {now} a game, from {then} over the last two seasons.",
     "Goals are down: {now} a game, from {then} over the last two seasons."),
    ("second_half", "Goals after half-time", 0, True,
     "{now} percent of goals now come after half-time. Over the last two seasons it was {then}.",
     "Fewer goals come late: {now} percent after half-time, down from {then}."),
    ("draws", "Draws", 0, True, "Draws are up: {now} percent of games, from {then}.",
     "Draws are rarer: {now} percent of games, down from {then}."),
    ("home_wins", "Home wins", 0, True, "Home wins are up: {now} percent of games, from {then}.",
     "Home advantage is shrinking: home sides win {now} percent, down from {then}."),
    ("shots", "Shots a game", 1, False, "Teams are shooting more: {now} shots a game, from {then}.",
     "Teams are shooting less: {now} shots a game, from {then}."),
    ("xg", "Expected goals a game", 2, False, "Teams are creating more: {now} expected goals a game, from {then}.",
     "Chances are drying up: {now} expected goals a game, from {then}."),
    ("cards", "Cards a game", 1, False, "Referees are reaching for their pockets: {now} cards a game, from {then}.",
     "Fewer bookings: {now} cards a game, down from {then}."),
    ("fouls", "Fouls a game", 1, False, "More fouls: {now} a game, up from {then}.",
     "Fewer fouls: {now} a game, down from {then}."),
    ("corners", "Corners a game", 1, False, "More corners: {now} a game, from {then}.",
     "Fewer corners: {now} a game, down from {then}."),
    ("clean", "Clean sheets", 0, True, "More clean sheets: {now} percent of team games, from {then}.",
     "Clean sheets are rarer: {now} percent of team games, down from {then}."),
    ("favourite", "Bookies' favourite won", 0, True,
     "Favourites are winning more: the bookies' favourite has won {now} percent of games, from {then}.",
     "Upsets are up: the bookies' favourite has won just {now} percent of games, down from {then}."),
]


# ---------------------------------------------------------------- the numbers

def per_match(matches: pd.DataFrame) -> pd.DataFrame:
    """Each LEAGUE stat for every match (NaN where the data is missing)."""
    m = matches
    goals = m.home_goals + m.away_goals
    inv = 1 / m[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=float)
    fav = np.array(["H", "D", "A"])[np.nan_to_num(inv).argmax(axis=1)]
    out = pd.DataFrame({
        "season": m.season, "goals": goals, "xg": m.home_xg + m.away_xg, "shots": m.home_shots + m.away_shots,
        "cards": m.home_yellows + m.away_yellows + m.home_reds + m.away_reds,
        "fouls": m.home_fouls + m.away_fouls, "corners": m.home_corners + m.away_corners,
        "draws": 100.0 * (m.result == "D"), "home_wins": 100.0 * (m.result == "H"),
        "clean": 50.0 * ((m.home_goals == 0).astype(int) + (m.away_goals == 0).astype(int)),
        "favourite": np.where(np.isnan(inv).any(axis=1), np.nan, 100.0 * (fav == m.result)),
        # share of the goals that came after half-time: weighted by goals, so summed separately below
        "late": goals - m.home_ht_goals - m.away_ht_goals, "all": goals,
    })
    return out


def league_trends(matches: pd.DataFrame, n: int = TRENDS) -> list[dict]:
    """The n league stats that have moved furthest this season from the two seasons before it."""
    pm = per_match(matches)
    current = int(pm.season.max())
    now, then = pm[pm.season == current], pm[pm.season.between(current - 2, current - 1)]
    if len(now) < MIN_GAMES or then.empty:
        return []
    found = []
    for key, label, dp, pct, up, down in LEAGUE:
        if key == "second_half":
            a, b = now.late.sum() / max(now["all"].sum(), 1), then.late.sum() / max(then["all"].sum(), 1)
            se = np.sqrt(b * (1 - b) / max(now["all"].sum(), 1))
            a, b, se = 100 * a, 100 * b, 100 * se
        else:
            x, y = now[key].dropna(), then[key].dropna()
            if len(x) < MIN_GAMES or y.empty:
                continue
            a, b, se = x.mean(), y.mean(), y.std() / np.sqrt(len(x))
        z = (a - b) / se if se else 0.0
        if abs(z) < MIN_Z or round(a, dp) == round(b, dp):
            continue
        fill = {"now": f"{a:.{dp}f}", "then": f"{b:.{dp}f}"}
        found.append({"key": key, "label": label, "pct": pct, "z": float(z), "up": bool(a > b),
                      "now": fill["now"], "then": fill["then"], "line": (up if a > b else down).format(**fill),
                      "bars": bars([b, a])})
    return sorted(found, key=lambda t: -abs(t["z"]))[:n]


def bars(values: list[float]) -> list[float]:
    """Bar widths in percent of the largest value."""
    top = max(abs(v) for v in values) or 1
    return [round(abs(v) / top * 100, 1) for v in values]


def results_v_performance(matches: pd.DataFrame) -> dict | None:
    """This season's side furthest below where its chances say it should be, and the one furthest above."""
    current = int(matches.season.max())
    table = luck.deserved_table(matches, current)
    if not table or table[0]["played"] < 5:
        return None
    under = min(table, key=lambda r: r["difference"])
    over = max(table, key=lambda r: r["difference"])
    if under["difference"] > -3 or over["difference"] < 3:
        return None
    rows = sorted(table, key=lambda r: r["actual"])
    return {"under": under, "over": over, "games": int(np.median([r["played"] for r in table])),
            "rows": [{"team": SHORT.get(r["team"], r["team"]), "actual": r["actual"], "deserved": r["position"],
                      "points": r["points"], "xpts": f"{r['xpts']:.0f}", "me": r["team"] in (under["team"], over["team"])}
                     for r in rows]}


def data_weights(matches: pd.DataFrame, settings: model.Settings) -> list[dict]:
    """Each season's share of the games and of the weight the model gives them today."""
    as_of = pd.to_datetime(matches.date).max() + pd.Timedelta(days=1)
    w = pd.Series(model.weights(matches.date, as_of, settings.half_life_days)).groupby(matches.season.values).sum()
    n = matches.groupby("season").size()
    return [{"season": int(s), "label": seasons.label(int(s)), "games": int(n[s]),
             "games_pct": round(n[s] / n.sum() * 100), "weight_pct": round(w[s] / w.sum() * 100)}
            for s in sorted(n.index)]


def ratings_ladder(matches: pd.DataFrame, settings: model.Settings, n: int = 10) -> list[dict]:
    """The model's overall rating (attack minus defence) for this season's clubs, best first."""
    current = int(matches.season.max())
    clubs = set(matches[matches.season == current].home_team) | set(matches[matches.season == current].away_team)
    t = model.fit(matches, settings=settings).table()
    t = t[t.team.isin(clubs)].head(n)
    # how many more (or fewer) goals than average a team scores less concedes, read off the log ratings
    vals = [(r.team, float(np.exp(r.attack) / np.exp(r.defence))) for r in t.itertuples()]
    top = max(v for _, v in vals)
    return [{"name": team, "team": SHORT.get(team, team), "v": f"{v:.2f}", "w": round(v / top * 100, 1)}
            for team, v in vals]


def latest_calls(data: Path) -> dict | None:
    """The newest round of calls made before kick-off (not a replay or an after-kick-off file)."""
    files = sorted(f for f in (data / "predictions").glob("*.json")
                   if "after-kickoff" not in f.name and not f.name.startswith("replay"))
    return json.loads(files[-1].read_text()) if files else None


def example(data: Path) -> dict | None:
    """The most confident call in the latest round, worked through: where it starts and what moved it."""
    calls = latest_calls(data)
    if not calls or not calls.get("games"):
        return None
    g = max(calls["games"], key=lambda g: max(g["probabilities"].values()))
    wf = g.get("waterfall")
    if not wf:
        return None
    rnd = str(calls.get("round", "")).rsplit("-", 1)[-1].strip()
    steps = sorted(wf["steps"], key=lambda s: -abs(s["change"]))
    shown = [s for s in steps if round(abs(s["change"])) >= 1][:2]
    top = max(g["probabilities"].values())
    level = wf["start"]
    rows = [{"label": "Average home side v average visitor", "left": 0, "width": round(level, 1),
             "fill": "start", "val": f"{level:.0f}%", "at": "1:0"}]
    for i, s in enumerate(shown):
        new = level + s["change"]
        rows.append({"label": s["label"], "left": round(min(level, new), 1), "width": round(abs(s["change"]), 1),
                     "fill": "up" if s["change"] > 0 else "down", "val": f"{s['change']:+.0f}",
                     "at": f"{2 + i}:0"})
        level = new
    rows.append({"label": "Our call", "left": 0, "width": round(top * 100, 1), "fill": "total",
                 "val": f"{top * 100:.0f}%", "at": f"{2 + len(shown)}:0", "bold": True})
    score = f"{g['score'][0]}–{g['score'][1]}"
    return {"g": g, "matchweek": rnd, "steps": shown, "rows": rows, "top": round(top * 100),
            "start": round(wf["start"]), "score": score, "outcome": explain.OUTCOME_NAMES[g["call"]]}


def call_rates(data: Path, season: int) -> dict | None:
    """How often strong and too-close calls came true in a back-tested season."""
    path = data / "backtest" / f"predictions_{season}.csv"
    if not path.exists():
        return None
    p = pd.read_csv(path)
    top = p[["p_home", "p_draw", "p_away"]].max(axis=1)
    second = p[["p_home", "p_draw", "p_away"]].apply(lambda r: sorted(r)[-2], axis=1)
    strong, close = p[top >= explain.STRONG], p[(top < 0.45) | (top - second < 0.05)]
    if strong.empty or close.empty:
        return None
    return {"season": season, "label": seasons.label(season),
            "strong": round((strong.call == strong.result).mean() * 100), "strong_n": len(strong),
            "close": round((close.call == close.result).mean() * 100), "close_n": len(close)}


def honest_record(data: Path, current: int) -> list[dict]:
    """Our hit rate and the bookies' favourite's, last season (replayed) and this season."""
    out = []
    for season in (current - 1, current):
        rec = record(data, season)
        if rec and rec.get("fav_rate") is not None:
            out.append({"season": season, "label": seasons.label(season), "now": season == current,
                        "n": rec["n"], "us": rec["rate"], "fav": rec["fav_rate"]})
    return out


def verdict(rows: list[dict]) -> tuple[str, tuple[str, str]]:
    """Level, ahead or behind the bookies over all the games, said plainly, and as a headline."""
    n = sum(r["n"] for r in rows)
    us = sum(r["us"] * r["n"] for r in rows) / n
    fav = sum(r["fav"] * r["n"] for r in rows) / n
    if abs(us - fav) < 2:
        return "Level with the bookies, not ahead of them.", ("Level with", "the bookies.")
    if us > fav:
        return ("A little ahead of the bookies, though that's too few games to prove anything.",
                ("Just ahead of", "the bookies."))
    return "A little behind the bookies, and we'll keep saying so.", ("Just behind", "the bookies.")


# ---------------------------------------------------------------- the script

def scenes(data: Path) -> tuple[list[dict], dict]:
    matches = pd.read_csv(data / "matches.csv")
    settings = predict.load_settings(data)
    current = int(matches.season.max())
    say = video.Phrasing(pd.Timestamp.now().strftime("%Y-%m"))
    facts = {"current": current, "label": seasons.label(current)}
    out = [{"kind": "hook", "lines": [
        say("hook", ["How does an algorithm call a Premier League game?",
                     "How do you teach a computer to call football matches?",
                     "What happens when you let the data call the Premier League?"])]}]

    # About a minute, one idea a scene: the data, the ratings, one game played out, the record. The trends
    # and the table-v-chances story make reels of their own, and stay in the caption.
    w = data_weights(matches, settings)
    facts["weights"] = w
    out.append({"kind": "data", "seasons": w, "lines": [
        "Step one: the data. Every game from the last three seasons.",
        "Not just goals, but every chance, and the bookies' odds. Recent games count most."]})
    facts["trends"] = league_trends(matches)
    facts["luck"] = results_v_performance(matches)

    ladder = ratings_ladder(matches, settings, n=5)
    out.append({"kind": "model", "rows": ladder, "lines": [
        "Step two: it rates every club's attack and defence.",
        f"Right now, {video.spoken(ladder[0]['name'])} come out on top."]})

    ex = example(data)
    facts["example"] = ex
    if ex:
        g = ex["g"]
        home, away = video.spoken(g["home_team"]), video.spoken(g["away_team"])
        ex["tens"] = max(1, round(ex["top"] / 10))
        k = video.COUNTS[ex["tens"]]
        odds = (f"It's a draw about {k} times in ten." if g["call"] == "D" else
                f"{video.spoken(g['home_team'] if g['call'] == 'H' else g['away_team'])} win about {k} times in ten.")
        out.append({"kind": "example", "ex": ex, "lines": [
            f"Step three: it plays the game out thousands of times. Take {home} against {away}.",
            f"{odds} So that's our call: {video.say_score({'score_text': ex['score']})}."]})
    facts["rates"] = call_rates(data, current - 1)

    rec = honest_record(data, current)
    facts["record"] = rec
    if rec:
        r = next((r for r in rec if r["now"] and r["n"] >= 30), rec[0])
        when = "This season so far" if r["now"] else "Replayed over last season"
        say_v, head = verdict([r])
        out.append({"kind": "record", "rows": [r], "head": head, "lines": [
            "Are we any good?",
            f"{when}: {r['us']} percent right. The bookies' favourite: {r['fav']}.",
            say_v + " Every call is posted before kick-off."]})

    out.append({"kind": "outro", "lines": [
        say("ask", ["What should the model look at next? Tell us below.",
                    "What would you add to the model? Tell us below."]),
        say("follow", ["Follow to keep us honest.", "Follow, and check our calls every Thursday."])]})
    return out, facts


def xg_mix(settings: model.Settings) -> str:
    if settings.xg_weight == 0.5:
        return "half from goals, half from expected goals"
    return f"{100 - settings.xg_weight * 100:.0f} percent from goals and {settings.xg_weight * 100:.0f} from expected goals"


def a_an(words: str) -> str:
    return ("an " if words[0] in "aeiou" else "a ") + words


def captions(facts: dict) -> dict:
    """Instagram and X captions for the reel."""
    lines = ["1️⃣ Data: three seasons of results, shots, xG and odds, recent games weighted most"]
    hook = facts["trends"][0]["line"] if facts["trends"] else None
    if hook:
        lines.append(f"2️⃣ This season. {hook}")
    lines.append("3️⃣ Model: attack and defence ratings per club, turned into the chance of every score")
    lines.append("4️⃣ Calls: 55%+ is strong, under 45% is too close to call")
    rec = facts["record"]
    head = "How does an algorithm call a Premier League game? Here's the whole process."
    tail = ""
    if rec:
        r = rec[-1]
        tail = f"\n\nThe honest bit: {r['us']}% right this season, the bookies' favourite {r['fav']}%."
    insta = (f"{head}\n\n" + "\n".join(lines) + tail + "\n\nWhat should the model look at next? 👇\n\n"
             f"AI voice. Analysis, not betting advice. Every call, with the reasons: {social.SITE_URL}\n\n{TAGS}")
    x = social.with_link(social.fit_x(
        "How our algorithm calls a Premier League game, start to finish: the data, the model, the call, "
        "and our honest record against the bookies." + (f"\n\nThis season. {hook}" if hook else "")
        + "\n\n#PremierLeague"), f"{social.SITE_URL}about/", "How it all works:")
    return {"instagram": insta, "x": x}


# ---------------------------------------------------------------- building

def build(data: Path, out: Path) -> dict:
    """Write the reel's script, captions and animated page into out/, listed in out/videos.json."""
    sc, facts = scenes(data)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATES / "style.css", out / "style.css")
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]),
                      trim_blocks=True, lstrip_blocks=True)
    stamp = pd.Timestamp.now().strftime("%Y-%m")
    voice = video.pick_voice(f"how-{stamp}", NAME)
    (out / "videos" / f"{NAME}.html").write_text(env.get_template("how_video.html").render(
        video={"title": "How it works", "scenes": sc}, facts=facts,
        width=video.WIDTH, height=video.HEIGHT, root=".."))
    entry = {"name": NAME, "title": "How it works", "voice": voice, "voice_name": dna.voice_name(voice),
             "captions": captions(facts), "scenes": [{"kind": s["kind"], "lines": s["lines"]} for s in sc]}
    (out / "videos.json").write_text(json.dumps([entry], indent=1, ensure_ascii=False))
    log.info("Wrote the how-it-works script into %s", out)
    return entry


def ready_to_post(out: Path) -> Path | None:
    """out/ready/how-it-works/: the video (named with its voice) and both captions."""
    v = json.loads((out / "videos.json").read_text())[0]
    clip = out / "video" / f"{NAME}.mp4"
    if not clip.exists():
        return None
    used = out / "video" / "voices.json"
    voice = (json.loads(used.read_text()) if used.exists() else {}).get(NAME, v["voice_name"])
    folder = out / "ready" / NAME
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(clip, folder / f"{NAME}-{voice.lower()}.mp4")
    (folder / "instagram.txt").write_text(v["captions"]["instagram"] + "\n")
    (folder / "x.txt").write_text(social.x_text(v["captions"]["x"]))
    (folder / "how.txt").write_text(
        f"How it works, read by {voice}. Post it as a Reel and on X, and pin it to the top of both profiles: "
        "it's the video that tells new followers what we do. Its trends and record are from "
        f"{pd.Timestamp.now():%d %B %Y}, so remake it every few months.\n")
    return folder


def email(out: Path) -> int:
    """Send the reel to KIT_EMAIL (or KIT_EMAIL_TO). Skips with a warning when the secrets are unset."""
    sender, password = os.environ.get("KIT_EMAIL"), os.environ.get("KIT_EMAIL_PASSWORD")
    if not sender or not password:
        log.warning("KIT_EMAIL and KIT_EMAIL_PASSWORD are not both set, so the reel was not emailed.")
        return 0
    folder = out / "ready" / NAME
    if not folder.exists():
        return 0
    msg = social.post_email({"day": f"{pd.Timestamp.now():%d %B %Y}"}, folder, "How it works", sender,
                            os.environ.get("KIT_EMAIL_TO") or sender)
    with smtplib.SMTP_SSL(os.environ.get("KIT_SMTP_HOST", "smtp.gmail.com"), 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)
    log.info("Emailed the how-it-works reel")
    return 1


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="the folder to write into, e.g. out/how-it-works")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--scripts", action="store_true", help="write the script and page only, don't render")
    parser.add_argument("--email", action="store_true", help="email the video and captions to KIT_EMAIL")
    args = parser.parse_args(argv)
    v = build(args.data, args.out)
    print(f"\n{v['title']} ({v['voice_name']})")
    for s in v["scenes"]:
        for line in s["lines"]:
            print("  ", line)
    if not args.scripts:
        video.render(args.out)
        ready_to_post(args.out)
        if args.email:
            email(args.out)


if __name__ == "__main__":
    main()
