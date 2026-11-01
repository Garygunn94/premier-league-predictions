import json

import numpy as np
import pandas as pd

from plpred import dna, video


def league(seed=3) -> pd.DataFrame:
    """Two full seasons of a 20-club league and 8 rounds of a third. "Wall" gives away almost nothing,
    "Hacker" fouls the most, "Riser" learns to defend in the second season, "Newco" replaces
    "Gone" this season."""
    rng = np.random.default_rng(seed)
    rows = []
    for season, rounds in ((2024, 38), (2025, 38), (2026, 8)):
        teams = [f"Club{i:02d}" for i in range(16)] + ["Wall", "Hacker", "Riser", "Gone" if season < 2026 else "Newco"]
        pairs = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(pairs)
        for n, (h, a) in enumerate(pairs[:rounds * 10]):
            xa_scale = {"Wall": 0.5, "Riser": 1.6 if season == 2024 else 0.6}
            hx = 1.4 * xa_scale.get(a, 1.0) * rng.uniform(0.7, 1.3)
            ax = 1.1 * xa_scale.get(h, 1.0) * rng.uniform(0.7, 1.3)
            hg, ag = rng.poisson(hx), rng.poisson(ax)
            rows.append({"season": season, "date": str(pd.Timestamp(f"{season}-08-16") + pd.Timedelta(days=n // 10 * 7)),
                         "home_team": h, "away_team": a, "home_goals": hg, "away_goals": ag,
                         "home_ht_goals": rng.binomial(hg, 0.45), "away_ht_goals": rng.binomial(ag, 0.45),
                         "home_shots": int(hx * 9) + 3, "away_shots": int(ax * 9) + 3,
                         "home_corners": rng.integers(2, 9), "away_corners": rng.integers(2, 9),
                         "home_fouls": 20 if h == "Hacker" else rng.integers(7, 14),
                         "away_fouls": 20 if a == "Hacker" else rng.integers(7, 14),
                         "home_yellows": rng.integers(0, 4), "away_yellows": rng.integers(0, 4),
                         "home_reds": 0, "away_reds": 0, "odds_home": 2.2, "odds_draw": 3.4, "odds_away": 3.3,
                         "home_xg": hx, "away_xg": ax})
    return pd.DataFrame(rows)


def test_traits_are_what_sets_a_club_apart_in_both_seasons():
    p = dna.profile(league(), "Wall")
    assert [s["season"] for s in p["seasons"]] == [2024, 2025]
    first = p["traits"][0]
    assert first["kind"] == "strength" and first["key"] == "xa" and first["ranks"] == [1, 1]
    strengths = [t for t in p["traits"] if t["kind"] == "strength"]
    assert len(strengths) == 3 and len({t["family"] for t in strengths}) == 3  # one per family
    assert p["now"]["games"] > 0 and "xa" in p["now"]["ranks"]


def test_flaws_and_big_changes():
    hacker = dna.profile(league(), "Hacker")
    flaw = next(t for t in hacker["traits"] if t["kind"] == "weakness")
    assert flaw["family"] == "discipline" and flaw["words"] in ("foul a lot", "pick up cards")
    riser = dna.profile(league(), "Riser")
    change = [t for t in riser["traits"] if t["kind"] == "change"]
    strengths = {t["key"] for t in riser["traits"] if t["kind"] == "strength"}
    assert change or "xa" not in strengths  # defending only got good in the second season
    assert all(max(t["ranks"]) <= 8 for t in riser["traits"] if t["kind"] == "strength" and t["ranks"][0] <= 8)


def test_promoted_clubs_get_this_season_alone():
    p = dna.profile(league(), "Newco")
    assert [s["season"] for s in p["seasons"]] == [2026] and p["now"] is None
    assert not any(t["kind"] == "change" for t in p["traits"])
    sc = dna.scenes(p)
    assert sc[0]["lines"][0].startswith("Newco. Promoted last summer, and ")
    assert "so far this season" in sc[1]["lines"][1]


def test_scripts_read_like_speech():
    t = {"key": "finish", "unit": "goals above expected a game", "values": [-0.3, 0.12]}
    assert dna.read_out(t, 0) == "0.30 goals below expected a game"
    assert dna.read_out(t, 1) == "0.12 goals above expected a game"
    t = {"key": "hold", "unit": "percent of half-time leads won", "values": [60.0, 94.4]}
    assert dna.read_out(t, 1, unit=False) == "94 percent"
    p = dna.profile(league(), "Wall")
    lines = [line for s in dna.scenes(p) for line in s["lines"]]
    assert lines[0].startswith("Wall. ") and "two seasons ago" in lines[0] and "last season" in lines[0]
    assert lines[2] == "One: they give away almost nothing."
    assert lines[3] == "The best in the league, two seasons running."
    assert lines[-1].startswith("Follow for your club next")
    assert not any("-0" in line for line in lines)


def test_build_writes_a_page_script_and_captions_per_club(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    league().to_csv(data / "matches.csv", index=False)
    out = tmp_path / "club-dna"
    videos = dna.build(data, out, ["Wall", "Newco"])
    assert [v["name"] for v in videos] == ["dna-wall", "dna-newco"]
    listing = json.loads((out / "videos.json").read_text())
    for v in listing:
        assert v["voice"] in video.VOICES and v["voice_name"] == video.VOICES[v["voice"]]["name"]
        page = (out / "videos" / f"{v['name']}.html").read_text()
        assert "window.render" in page and 'data-scene="0"' in page and "../style.css" in page
        assert v["scenes"][0]["kind"] == "intro" and v["scenes"][-1]["kind"] == "outro"
        assert "#Wall" in v["captions"]["instagram"] or "#Newco" in v["captions"]["instagram"]
        assert v["captions"]["instagram"].count("#") <= 5 and "AI voice" in v["captions"]["instagram"]
    assert (out / "style.css").exists()
    assert dna.build(data, tmp_path / "none", []) == []  # an empty shard makes nothing


def test_ready_folders_hold_the_video_and_captions(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    league().to_csv(data / "matches.csv", index=False)
    out = tmp_path / "club-dna"
    dna.build(data, out, ["Wall", "Hacker"])
    (out / "video").mkdir()
    (out / "video" / "dna-wall.mp4").write_bytes(b"fake mp4")
    (out / "video" / "voices.json").write_text(json.dumps({"dna-wall": "Emma"}))  # Kokoro stood in
    folders = dna.ready_to_post(out)
    assert [f.name for f in folders] == ["wall"]
    assert [p.name for p in folders[0].glob("*.mp4")] == ["dna-wall-emma.mp4"]
    assert "What makes Wall tick?" in (folders[0] / "instagram.txt").read_text()
    assert (folders[0] / "x.txt").exists() and "read by Emma" in (folders[0] / "how.txt").read_text()
    [msg] = dna.emails(out, "me@example.com", "me@example.com")
    assert msg["Subject"] == "Club DNA · Wall" and "---- INSTAGRAM ----" in msg.get_body().get_content()
    assert [p.get_filename() for p in msg.iter_attachments()] == ["dna-wall-emma.mp4"]
