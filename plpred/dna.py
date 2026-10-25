"""Club DNA: one narrated 9:16 video per club on what makes them the side they are, for the posting kit.

    python -m plpred.dna out/club-dna                    # every club in the league now
    python -m plpred.dna out/club-dna --only Arsenal     # just these clubs
    python -m plpred.dna out/club-dna --shard 2/4        # every fourth club, for parallel runs
    python -m plpred.dna out/club-dna --scripts          # write the scripts and pages, don't render
    python -m plpred.dna out/club-dna --email            # and email each club's video and captions

Unlike the weekly videos these stay true for months, so they can be posted any time. Each club is
ranked against the league on the STATS below in each of its last two seasons. A trait is a stat
where it sits near the top in BOTH seasons, so it describes the team rather than a lucky run.
Stats are grouped into families (attack, defence, control, ...) and only the best trait per family
is kept, so a video doesn't say "lots of shots, lots of xG, lots of corners" as three points. Then
come a flaw (bottom six in both seasons), the stat whose rank moved most, and whether the strengths
are holding up this season.

A club's two seasons are its last two full Premier League seasons. A club with one (promoted the
year before last) compares it with this season so far; a club promoted last summer gets this
season alone. Every sentence is a template filled from these numbers, like the weekly videos.

The folder gets the scripts and animated pages, the videos (rendered by plpred.video.render, so
they use the same voices) and ready/, one folder per club with the video and both captions. With
--email each club's video and captions are emailed as one message, from and to KIT_EMAIL like the
weekly posting kit.
"""
import argparse
import json
import logging
import os
import re
import shutil
import smtplib
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from plpred import social, video
from plpred.site import SHORT, TEMPLATES
from plpred.sources import api_football

log = logging.getLogger("plpred.dna")

# (key, family, higher is better, a top rank means they..., a bottom rank means they..., unit)
STATS = [
    ("ppg", "results", True, "collect points", "struggle for points", "points a game"),
    ("xf", "attack", True, "create chances", "struggle to create", "expected goals a game"),
    ("shots_f", "attack", True, "shoot on sight", "rarely shoot", "shots a game"),
    ("xa", "defence", False, "give away almost nothing", "leak chances", "expected goals against a game"),
    ("shots_a", "defence", False, "smother shots", "let opponents shoot", "shots faced a game"),
    ("clean", "defence", True, "keep clean sheets", "rarely keep clean sheets", "percent clean sheets"),
    ("share", "control", True, "dominate the ball and territory", "spend games under siege", "percent of the shots"),
    ("xg_shot", "chance quality", True, "work the ball into big chances", "take low-percentage shots",
     "expected goals per shot"),
    ("xg_shot_a", "chance quality", False, "only allow speculative shots", "allow clear chances",
     "expected goals per shot faced"),
    ("finish", "finishing", True, "finish better than their chances", "waste their chances",
     "goals above expected a game"),
    ("keep", "goalkeeping", True, "concede fewer than their chances say", "concede more than their chances say",
     "goals saved above expected a game"),
    ("corners", "set pieces", True, "win a lot of corners", "win few corners", "corners a game"),
    ("fouls", "discipline", False, "rarely foul", "foul a lot", "fouls a game"),
    ("cards", "discipline", False, "stay out of the book", "pick up cards", "cards a game"),
    ("home_edge", "home and away", True, "turn home into a fortress", "get no lift from home",
     "more points a game at home"),
    ("second_half", "game state", True, "finish games strongly", "do their damage early",
     "percent of their goals after half-time"),
    ("comeback", "game state", True, "recover from losing positions", "rarely recover once behind",
     "points a game from behind at half-time"),
    ("hold", "game state", True, "see out leads", "let leads slip", "percent of half-time leads won"),
    ("vs_market", "expectations", True, "beat what the bookies expected", "fall short of what the bookies expected",
     "points a game above the bookies' expectation"),
]
HIGHER = {k: h for k, _, h, *_ in STATS}
TIERS = (4, 6, 8)      # strengths need a top-4 rank every season; clubs with fewer than WANT relax to 6, then 8
WANT = 3
FLAW = 15              # a flaw is 15th or worse every season
MOVED = 8              # the big change moved at least this many places
MIN_GAMES = 6          # this season counts as a season once a club has played this many games
STYLE = {"second_half"}  # a way of playing, never called a flaw
DP = {"finish": 2, "keep": 2, "xg_shot": 2, "xg_shot_a": 2, "ppg": 2, "home_edge": 2, "comeback": 2, "vs_market": 2,
      "share": 0, "second_half": 0, "hold": 0, "clean": 0}
# Stats that can go below zero, read as "0.3 goals below expected", never "-0.3 goals above".
SIGNED = {"finish": ("above", "below"), "keep": ("above", "below"), "vs_market": ("above", "below"),
          "home_edge": ("more", "fewer")}
FAMILY_NAME = {"attack": "attack", "defence": "defence", "control": "control of games",
               "chance quality": "chance quality", "finishing": "finishing", "goalkeeping": "goalkeeping",
               "set pieces": "set-piece threat", "discipline": "discipline", "home and away": "home form",
               "game state": "game management", "expectations": "record against the odds"}
ORD_WORDS = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
             "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
             "eighteenth", "nineteenth", "twentieth"]
NUMBER_WORDS = {19: "nineteen"}
COUNT = ["One", "Two", "Three"]
TAGS = "#PremierLeague #FootballAnalytics #xG"


# ---------------------------------------------------------------- the stats

def team_games(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, from that team's point of view."""
    inv = 1 / matches[["odds_home", "odds_draw", "odds_away"]]
    inv = inv.div(inv.sum(axis=1), axis=0)
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        m = matches
        gf, ga = m[f"{side}_goals"], m[f"{other}_goals"]
        rows.append(pd.DataFrame({
            "season": m.season, "team": m[f"{side}_team"], "venue": side[0].upper(),
            "gf": gf, "ga": ga, "hf": m[f"{side}_ht_goals"], "ha": m[f"{other}_ht_goals"],
            "xf": m[f"{side}_xg"], "xa": m[f"{other}_xg"], "sf": m[f"{side}_shots"], "sa": m[f"{other}_shots"],
            "corners": m[f"{side}_corners"], "fouls": m[f"{side}_fouls"],
            "cards": m[f"{side}_yellows"] + m[f"{side}_reds"],
            "points": np.select([gf > ga, gf == ga], [3, 1], 0),
            "exp_points": 3 * (inv.odds_home if side == "home" else inv.odds_away) + inv.odds_draw}))
    return pd.concat(rows, ignore_index=True)


def season_stats(g: pd.DataFrame) -> pd.DataFrame:
    """Per-game stats for every team in one season's team_games, and its league position."""
    out = {}
    for team, t in g.groupby("team"):
        home, away = t[t.venue == "H"], t[t.venue == "A"]
        behind, ahead = t[t.hf < t.ha], t[t.hf > t.ha]
        out[team] = {
            "games": len(t), "points": t.points.sum(), "gf": t.gf.sum(), "ga": t.ga.sum(),
            "ppg": t.points.mean(), "xf": t.xf.mean(), "xa": t.xa.mean(),
            "shots_f": t.sf.mean(), "shots_a": t.sa.mean(), "share": 100 * t.sf.sum() / (t.sf.sum() + t.sa.sum()),
            "xg_shot": t.xf.sum() / t.sf.sum(), "xg_shot_a": t.xa.sum() / t.sa.sum(),
            "finish": (t.gf - t.xf).mean(), "keep": (t.xa - t.ga).mean(),
            "corners": t.corners.mean(), "fouls": t.fouls.mean(), "cards": t.cards.mean(),
            "home_edge": home.points.mean() - away.points.mean(),
            "second_half": 100 * (t.gf - t.hf).sum() / max(t.gf.sum(), 1),
            "comeback": behind.points.mean() if len(behind) else np.nan,
            "hold": 100 * (ahead.points == 3).mean() if len(ahead) else np.nan,
            "clean": 100 * (t.ga == 0).mean(),
            "vs_market": (t.points - t.exp_points).mean(),
        }
    df = pd.DataFrame(out).T.astype(float)
    order = df.assign(gd=df.gf - df.ga).sort_values(["points", "gd", "gf"], ascending=False).index
    df["position"] = pd.Series(range(1, len(df) + 1), index=order)
    return df


def ranks(stats: pd.DataFrame) -> pd.DataFrame:
    """1 = best in the league on every stat."""
    return pd.DataFrame({k: stats[k].rank(ascending=not HIGHER[k], method="min") for k in HIGHER})


def periods(seasons: dict[int, pd.DataFrame], team: str, current: int) -> list[int]:
    """The (up to) two seasons a club is judged on: its last two full ones, else its last full one
    and this season so far, else this season alone."""
    full = [s for s in sorted(seasons) if s != current and team in seasons[s].index]
    now = [current] if team in seasons[current].index and seasons[current].at[team, "games"] >= MIN_GAMES else []
    return full[-2:] if len(full) >= 2 else (full + now)[-2:]


def profile(matches: pd.DataFrame, team: str) -> dict | None:
    """A club's traits, as dicts the script and the page are written from."""
    current = int(matches.season.max())
    g = team_games(matches.dropna(subset=["home_xg", "away_xg", "home_shots", "away_shots"]))
    stats = {int(s): season_stats(g[g.season == s]) for s in sorted(g.season.unique())}
    if current not in stats:
        return None
    rk = {s: ranks(df) for s, df in stats.items()}
    have = periods(stats, team, current)
    if not have:
        return None
    traits = []
    for key, family, higher, top, bottom, unit in STATS:
        rs = [rk[s].at[team, key] for s in have]
        if any(np.isnan(rs)):
            continue
        traits.append({"key": key, "family": family, "unit": unit, "top": top, "bottom": bottom,
                       "ranks": [int(r) for r in rs], "values": [float(stats[s].at[team, key]) for s in have],
                       "league_avg": [float(stats[s][key].mean()) for s in have],
                       "change": int(rs[0] - rs[-1]),
                       # the league ladder in the newest season, best first, for the chart
                       "ladder": sorted(((t, float(v)) for t, v in stats[have[-1]][key].dropna().items()),
                                        key=lambda x: -x[1] if higher else x[1])})

    def one_per_family(pool, n, used=()):
        seen, out = set(used), []
        for t in pool:
            if t["family"] not in seen:
                seen.add(t["family"])
                out.append(t)
        return out[:n]

    # league position is the intro, not a trait
    pool = sorted((t for t in traits if t["family"] != "results"), key=lambda t: (max(t["ranks"]), sum(t["ranks"])))
    strengths = []
    for tier in TIERS:
        strengths = one_per_family([t for t in pool if max(t["ranks"]) <= tier], WANT)
        if len(strengths) >= WANT:
            break
    if len(strengths) < WANT:  # still short: the best ranks in the newest season fill the gaps
        strengths += one_per_family(sorted((t for t in pool if t["ranks"][-1] <= 5 and t not in strengths),
                                           key=lambda t: t["ranks"][-1]),
                                    WANT - len(strengths), {t["family"] for t in strengths})
    for t in strengths:
        t.update(kind="strength", words=t["top"])
    used = {t["family"] for t in strengths}
    flaw = one_per_family([t for t in sorted(traits, key=lambda t: (-min(t["ranks"]), -sum(t["ranks"])))
                           if min(t["ranks"]) >= FLAW and t["key"] not in STYLE], 1, used)
    for t in flaw:
        t.update(kind="weakness", words=t["bottom"])
    moved = [t for t in sorted(traits, key=lambda t: -abs(t["change"]))
             if len(have) == 2 and abs(t["change"]) >= MOVED and t not in strengths + flaw][:1]
    for t in moved:
        t.update(kind="change", words=t["top"] if t["change"] > 0 else t["bottom"])

    now = None
    if have[-1] != current and team in stats[current].index:
        cs = stats[current]
        now = {"games": int(cs.at[team, "games"]), "position": int(cs.at[team, "position"]),
               "points": int(cs.at[team, "points"]), "ranks": {t["key"]: int(rk[current].at[team, t["key"]])
                                                              for t in strengths}}
    return {"team": team, "current": current,
            "seasons": [{"season": s, "label": season_label(s), "now": s == current,
                         "position": int(stats[s].at[team, "position"]), "points": int(stats[s].at[team, "points"]),
                         "games": int(stats[s].at[team, "games"])} for s in have],
            "traits": strengths + flaw + moved, "now": now}


# ---------------------------------------------------------------- the script

def season_label(s: int) -> str:
    return f"{s}/{str(s + 1)[-2:]}"


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def fmt(key: str, v: float) -> str:
    out = f"{v:.{DP.get(key, 1)}f}"
    return out[1:] if out.startswith("-") and float(out) == 0 else out


def read_out(t: dict, i: int, unit: bool = True) -> str:
    """A value as the narrator reads it, e.g. '0.3 goals below expected a game'."""
    v, u = t["values"][i], t["unit"]
    if t["key"] in SIGNED and v < 0:
        u, v = u.replace(*SIGNED[t["key"]], 1), -v
    if unit or t["key"] in SIGNED:
        return f"{fmt(t['key'], v)} {u}"
    return fmt(t["key"], v) + (" percent" if u.startswith("percent") else "")


def when(p: dict) -> list[str]:
    """How the narrator names each of the club's seasons."""
    ago = {0: "so far this season", 1: "last season", 2: "two seasons ago"}
    return [ago.get(p["current"] - s["season"], f"in {s['label']}") for s in p["seasons"]]


def rank_phrase(p: dict, ranks: list[int]) -> str:
    both = "last season and this" if p["seasons"][-1]["now"] else "both seasons"
    if len(ranks) == 1:
        return "the best in the league" if ranks[0] == 1 else f"{ORD_WORDS[ranks[0]]} in the league"
    if all(r == 1 for r in ranks):
        return f"the best in the league, {'two seasons running' if both == 'both seasons' else both}"
    for n, word in ((3, "three"), (6, "six")):
        if max(ranks) <= n:
            return f"top {word} in the league {both}"
    return f"{ORD_WORDS[ranks[0]]} and then {ORD_WORDS[ranks[1]]} in the league"


def finish_words(s: dict) -> str:
    if s["now"]:
        return f"{ORD_WORDS[s['position']]} after {s['games']} games this season"
    return "champions" if s["position"] == 1 else ORD_WORDS[s["position"]]


def intro_line(p: dict) -> str:
    name, s, w = video.spoken(p["team"]), p["seasons"], when(p)
    if len(s) == 1:
        return f"{name}. Promoted last summer, and {finish_words(s[0])}."
    if s[-1]["now"]:
        return f"{name}. {finish_words(s[0]).capitalize()} {w[0]}, {finish_words(s[1])}."
    return f"{name}. {finish_words(s[0]).capitalize()} {w[0]}, {finish_words(s[1])} {w[1]}."


def ladder(t: dict, team: str) -> list[dict]:
    """The chart: every club's value, best first, bars measured from zero."""
    vals = [v for _, v in t["ladder"]]
    lo, hi = min(0, min(vals)), max(0, max(vals))
    span = (hi - lo) or 1
    return [{"team": SHORT.get(name, name), "me": name == team, "v": fmt(t["key"], v),
             "left": round((min(v, 0) - lo) / span * 100, 1), "w": max(round(abs(v) / span * 100, 1), 0.6)}
            for name, v in t["ladder"]]


def scenes(p: dict) -> list[dict]:
    team, w = p["team"], when(p)
    out = [{"kind": "intro", "lines": [
        intro_line(p),
        f"We ranked them against the league on {NUMBER_WORDS.get(len(STATS), len(STATS))} stats. "
        "Here's what makes them tick."]}]
    n = 0
    for t in p["traits"]:
        if t["kind"] == "strength":
            mean = fmt(t["key"], t["league_avg"][-1])
            avg = "" if (t["key"] in SIGNED and t["key"] != "home_edge") or mean == fmt(t["key"], t["values"][-1]) \
                else f" The league average is {mean}."
            out.append({"kind": "trait", "t": t, "label": f"Trait {COUNT[n].lower()} · {t['family']}",
                        "rows": ladder(t, team), "lines": [
                            f"{COUNT[n]}: they {t['words']}.",
                            f"{read_out(t, -1)} {w[-1]}, {rank_phrase(p, t['ranks'])}.{avg}"]})
            n += 1
        elif t["kind"] == "weakness":
            out.append({"kind": "trait", "t": t, "label": f"The flaw · {t['family']}", "rows": ladder(t, team),
                        "lines": [f"The flaw: they {t['words']}.",
                                  f"{read_out(t, -1)} {w[-1]}, {ORD_WORDS[t['ranks'][-1]]} of twenty."]})
        else:
            up = t["change"] > 0
            out.append({"kind": "change", "t": t, "up": up, "vals": [fmt(t["key"], v) for v in t["values"]], "lines": [
                f"{'The big change' if up else 'The big worry'}: they now {t['words']}.",
                f"{read_out(t, 0)} {w[0]}, {ORD_WORDS[t['ranks'][0]]}. "
                f"{read_out(t, -1, unit=False)} {w[-1]}, {ORD_WORDS[t['ranks'][-1]]}."]})
    if p["now"]:
        now = p["now"]
        rows = [{"label": FAMILY_NAME.get(t["family"], t["family"]).capitalize(), "then": t["ranks"][-1],
                 "rank": now["ranks"][t["key"]]} for t in p["traits"] if t["kind"] == "strength"]
        held = [r for r in rows if r["rank"] <= 3]
        slipped = [r for r in rows if r["rank"] > 8]
        verdict = f"And the {held[0]['label'].lower()} is still ranked {ORD_WORDS[held[0]['rank']]}." if held else ""
        if slipped:
            verdict += f" But the {slipped[0]['label'].lower()} has slipped to {ORD_WORDS[slipped[0]['rank']]}."
        out.append({"kind": "now", "rows": rows, "now": now, "lines": [
            f"This season, after {now['games']} games, they're {ORD_WORDS[now['position']]}.",
            verdict.strip() or "The same traits are holding up."]})
    out.append({"kind": "outro", "lines": ["Follow for your club next. More at Inkwell and Algorithm dot com."]})
    return out


def captions(p: dict) -> dict:
    """Instagram and X captions for a club's video."""
    team, w = p["team"], when(p)
    lines = []
    for t in p["traits"]:
        rk = " then ".join(ordinal(r) for r in t["ranks"])
        if t["kind"] == "strength":
            lines.append(f"✅ They {t['words']} ({rk})")
        elif t["kind"] == "weakness":
            lines.append(f"❌ The flaw: they {t['words']} ({rk})")
        else:
            lines.append(f"{'📈' if t['change'] > 0 else '📉'} They now {t['words']} ({rk})")
    span = (f"{p['seasons'][0]['label']} and {p['seasons'][1]['label']}" if len(p["seasons"]) == 2
            else f"{p['seasons'][0]['label']} so far")
    head = f"What makes {team} tick? We ranked every club on {len(STATS)} stats over {span}."
    insta = (f"{head}\n\n" + "\n".join(lines) + "\n\nRanks are out of 20. Which club should we do next? 👇\n\n"
             f"AI voice. Every call, with the reasons: {social.SITE_URL}\n\n{TAGS} {social.club_tag(team)}")
    strengths = [t for t in p["traits"] if t["kind"] == "strength"]
    x = social.with_link(social.fit_x(f"What makes {team} tick? Two seasons of data, three traits:\n\n" if len(
        p["seasons"]) == 2 else f"What makes {team} tick? The data so far, three traits:\n\n", "") +
        "\n".join(f"{i}. They {t['words']}" for i, t in enumerate(strengths, 1)) + f"\n\n{social.club_tag(team)}",
        f"{social.SITE_URL}club/", "Every club's season in numbers:")
    return {"instagram": insta, "x": x}


# ---------------------------------------------------------------- building

def slug(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", team.lower().replace("'", "")).strip("-")


def voice_name(voice: str) -> str:
    v = video.VOICES[voice]
    return v["name"] if isinstance(v, dict) else v


def clubs_now(matches: pd.DataFrame) -> list[str]:
    cur = matches[matches.season == matches.season.max()]
    return sorted(set(cur.home_team) | set(cur.away_team))


def build(data: Path, out: Path, teams: list[str] | None = None) -> list[dict]:
    """Write each club's script, captions and animated page into out/, listed in out/videos.json."""
    matches = pd.read_csv(data / "matches.csv")
    (out / "videos").mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATES / "style.css", out / "style.css")
    badges = {}
    if (data / "badges").exists():
        shutil.copytree(data / "badges", out / "badges", dirs_exist_ok=True)
        badges = {t: f"{api_football.badge_name(t)}.png" for t in clubs_now(matches)
                  if (data / "badges" / f"{api_football.badge_name(t)}.png").exists()}
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]),
                      trim_blocks=True, lstrip_blocks=True)
    page = env.get_template("club_video.html")
    videos = []
    for team in clubs_now(matches) if teams is None else teams:
        p = profile(matches, team)
        if p is None:
            log.warning("No data for %s, so no video", team)
            continue
        name = f"dna-{slug(team)}"
        sc = scenes(p)
        voice = video.pick_voice(f"dna-{p['current']}", name)
        (out / "videos" / f"{name}.html").write_text(page.render(
            p=p, video={"title": f"{team}: club DNA", "scenes": sc}, badges=badges, clubs={},
            width=video.WIDTH, height=video.HEIGHT, root=".."))
        videos.append({"name": name, "team": team, "title": f"{team}: club DNA", "voice": voice,
                       "voice_name": voice_name(voice), "captions": captions(p),
                       "scenes": [{"kind": s["kind"], "lines": s["lines"]} for s in sc]})
    (out / "videos.json").write_text(json.dumps(videos, indent=1, ensure_ascii=False))
    log.info("Wrote %d club scripts into %s", len(videos), out)
    return videos


def ready_to_post(out: Path) -> list[Path]:
    """out/ready/: a folder per rendered club with the video (named with its voice) and both captions."""
    videos = json.loads((out / "videos.json").read_text())
    used = out / "video" / "voices.json"   # the voice each video was actually read by, when recorded
    used = json.loads(used.read_text()) if used.exists() else {}
    folders = []
    for v in videos:
        clip = out / "video" / f"{v['name']}.mp4"
        if not clip.exists():
            continue
        folder = out / "ready" / slug(v["team"])
        folder.mkdir(parents=True, exist_ok=True)
        voice = used.get(v["name"], v["voice_name"])
        shutil.copy(clip, folder / f"{v['name']}-{voice.lower()}.mp4")
        (folder / "instagram.txt").write_text(v["captions"]["instagram"] + "\n")
        (folder / "x.txt").write_text(social.x_text(v["captions"]["x"]))
        (folder / "how.txt").write_text(
            f"{v['title']}, read by {voice}. Post it as a Reel and on X whenever suits: it stays true all season. "
            "Good moments: the week of the club's next big game, or when its fans are talking about it.\n")
        folders.append(folder)
    return folders


def emails(out: Path, sender: str, to: str) -> list:
    """One email per rendered club, laid out like the weekly kit's posts."""
    folders = sorted(p for p in (out / "ready").iterdir() if p.is_dir()) if (out / "ready").exists() else []
    titles = {slug(v["team"]): v["team"] for v in json.loads((out / "videos.json").read_text())}
    return [social.post_email({"day": titles.get(f.name, f.name)}, f, "Club DNA", sender, to) for f in folders]


def email(out: Path) -> int:
    """Send the club videos to KIT_EMAIL (or KIT_EMAIL_TO). Skips with a warning when the secrets are unset."""
    sender, password = os.environ.get("KIT_EMAIL"), os.environ.get("KIT_EMAIL_PASSWORD")
    if not sender or not password:
        log.warning("KIT_EMAIL and KIT_EMAIL_PASSWORD are not both set, so the club videos were not emailed.")
        return 0
    msgs = emails(out, sender, os.environ.get("KIT_EMAIL_TO") or sender)
    if msgs:
        with smtplib.SMTP_SSL(os.environ.get("KIT_SMTP_HOST", "smtp.gmail.com"), 465) as smtp:
            smtp.login(sender, password)
            for m in msgs:
                smtp.send_message(m)
    log.info("Emailed %d club videos", len(msgs))
    return len(msgs)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="the folder to write into, e.g. out/club-dna")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--only", nargs="*", metavar="CLUB", help="just these clubs, e.g. Arsenal \"Man United\"")
    parser.add_argument("--shard", metavar="I/N", help="every Nth club starting with the Ith, e.g. 1/4")
    parser.add_argument("--scripts", action="store_true", help="write the scripts and pages only, don't render")
    parser.add_argument("--email", action="store_true", help="email each club's video and captions to KIT_EMAIL")
    args = parser.parse_args(argv)
    teams = args.only or clubs_now(pd.read_csv(args.data / "matches.csv"))
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        teams = teams[i - 1::n]
    if not teams:
        log.info("No clubs for this shard")
        return
    videos = build(args.data, args.out, teams)
    for v in videos:
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
