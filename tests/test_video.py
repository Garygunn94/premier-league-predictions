import json

import pytest

from plpred import site, social, video
from tests.test_site import make_data
from tests.test_social import fake_render


def test_kit_has_a_video_script_for_the_round_and_each_game(tmp_path):
    kit = site.build(make_data(tmp_path), tmp_path / "site") / "social"
    videos = json.loads((kit / "videos.json").read_text())
    names = [v["name"] for v in videos]
    assert names[0] == "video-round" and "video-strong-v-weak" in names
    for v in videos:
        assert v["voice"] in video.VOICES and v["voice_name"] == video.VOICES[v["voice"]]["name"]
        page = (kit / "videos" / f"{v['name']}.html").read_text()
        assert "window.render" in page and 'data-scene="0"' in page
        assert v["scenes"][-1]["kind"] == "outro"
    game = next(v for v in videos if v["name"] == "video-strong-v-weak")
    kinds = [s["kind"] for s in game["scenes"]]
    assert kinds.index("call") < kinds.index("reasons") and kinds[0] in ("hook", "call")
    call = next(s for s in game["scenes"] if s["kind"] == "call")
    assert "Strong" in call["lines"][0] and "Weak" in call["lines"][0]
    posts = json.loads((kit / "posts.json").read_text())["posts"]
    assert posts[0]["video"] == "video-round"
    assert any(p["video"] == "video-strong-v-weak" for p in posts)


def test_voices_are_random_but_stable():
    picks = {video.pick_voice(8, f"video-{i}") for i in range(20)}
    assert picks == set(video.VOICES)
    assert video.pick_voice(8, "video-round") == video.pick_voice(8, "video-round")


def test_scripts_vary_across_the_round():
    say = [video.Phrasing(8, i) for i in range(4)]
    options = ["a", "b", "c"]
    picks = [s("slot", options) for s in say]
    assert all(a != b for a, b in zip(picks, picks[1:]))
    assert video.Phrasing(8, 2)("slot", options) == picks[2]  # rebuilds keep the script


def test_openers_tell_the_best_story_and_mix_across_the_round():
    base = {"home_team": "Leeds", "away_team": "Sunderland", "call": "H", "confidence": "solid",
            "headline": "Leeds' attack should be too much for Sunderland", "mkt": None,
            "standings": {"home": {"position": 1}, "away": {"position": 20}},
            "form": {"home": [{"result": "W"}] * 5, "away": [{"result": "L"}] * 5}}
    assert video.hooks(base)[0][2] == "Leeds have won their last five."
    kind, _, line = video.hooks(dict(base, form={}))[0]
    assert kind == "table" and line == "Top of the table Leeds host Sunderland, who are twentieth."
    picks = video.pick_hooks([base, base])
    assert picks[0][0] != picks[1][0]
    upset = dict(base, mkt={"H": 30, "D": 25, "A": 45})
    assert video.hooks(upset)[0] == ("market", 0.95, "The bookies have Sunderland as favourites. We don't.")


def test_scripts_read_like_speech():
    v = {"score_text": "1–2", "call": "A", "home_team": "Ipswich", "away_team": "Nott'm Forest"}
    assert video.say_score(v) == "two-one" and video.call_words(v) == "a Forest win"
    assert video.say_score({"score_text": "0–0"}) == "nil-nil"
    assert video.say_score({"score_text": "1–1"}) == "one-all"
    assert video.call_words({"call": "H", "home_team": "Arsenal"}) == "an Arsenal win"
    assert not video.videoable({"after_kickoff": True}) and not video.videoable({"result_text": "1–0"})


def test_rendered_videos_go_into_the_post_folders_and_emails(tmp_path):
    kit = site.build(make_data(tmp_path), tmp_path / "site") / "social"
    fake_render(kit)
    (kit / "video").mkdir()
    for v in json.loads((kit / "videos.json").read_text()):
        (kit / "video" / f"{v['name']}.mp4").write_bytes(b"fake mp4")
    folders = social.ready_to_post(kit)
    clips = list(folders[0].glob("*.mp4"))
    mw = json.loads((kit / "posts.json").read_text())["matchweek"]
    assert len(clips) == 1 and clips[0].name in (f"matchweek-{mw}-round-charon.mp4", f"matchweek-{mw}-round-kore.mp4")
    game = next(f for f in folders if list(f.glob("strong-v-weak-*.mp4")))
    assert (game / "how.txt").read_text().startswith("Carousel")
    assert "narrated video, read by" in (folders[0] / "how.txt").read_text()
    msgs = social.kit_emails(kit, "me@example.com", "me@example.com")
    assert any(p.get_content_type() == "video/mp4" for p in msgs[1].iter_attachments())


def test_kit_names_the_voice_a_video_was_actually_read_by(tmp_path):
    kit = site.build(make_data(tmp_path), tmp_path / "site") / "social"
    fake_render(kit)
    (kit / "video").mkdir()
    (kit / "video" / "video-round.mp4").write_bytes(b"fake mp4")
    (kit / "video" / "voices.json").write_text(json.dumps({"video-round": "Emma"}))  # Kokoro stood in
    folders = social.ready_to_post(kit)
    assert [p.name.rsplit("-", 1)[1] for p in folders[0].glob("*.mp4")] == ["emma.mp4"]
    assert "read by Emma" in (folders[0] / "how.txt").read_text()


def test_narration_times_every_line(tmp_path):
    import numpy as np
    pytest.importorskip("soundfile")  # installed only where the videos are made

    scenes = [{"lines": ["One.", "Two."]}, {"lines": ["Three."]}]
    tl = video.narrate(lambda line: (np.zeros(24000, dtype=np.float32), 24000), scenes, tmp_path / "v.wav")
    lines = [l for sc in tl["scenes"] for l in sc["timed"]]
    assert [l["text"] for l in lines] == ["One.", "Two.", "Three."]
    assert all(b["start"] > a["end"] for a, b in zip(lines, lines[1:]))
    assert tl["duration"] > lines[-1]["end"] and (tmp_path / "v.wav").exists()


def played_round(tmp_path, both=True):
    """The test round with Strong 2-0 Weak played and, if both, Average 1-1 Good too."""
    import pandas as pd
    data = make_data(tmp_path)
    (data / "upcoming").mkdir()
    rows = {"fixture_id": [1, 2], "status": ["FT", "FT"], "home_team": ["Strong", "Average"],
            "away_team": ["Weak", "Good"], "home_goals": [2, 1], "away_goals": [0, 1]}
    pd.DataFrame({k: v if both else v[:1] for k, v in rows.items()}).to_csv(
        data / "upcoming" / "fixtures.csv", index=False)
    return site.build(data, tmp_path / "site") / "social"


def test_every_video_ends_asking_for_a_comment_and_a_follow(tmp_path):
    kit = site.build(make_data(tmp_path), tmp_path / "site") / "social"
    for v in json.loads((kit / "videos.json").read_text()):
        ask, follow = v["scenes"][-1]["lines"]
        assert "?" in ask and follow in video.FOLLOW
        assert follow in (kit / "videos" / f"{v['name']}.html").read_text()


def test_the_first_frame_shows_the_hook(tmp_path):
    kit = site.build(make_data(tmp_path), tmp_path / "site") / "social"
    page = (kit / "videos" / "video-round.html").read_text()
    assert "first && +l === 0" in page and "Matchweek 9, <span" in page


def test_a_finished_round_gets_a_results_video(tmp_path):
    kit = played_round(tmp_path)
    videos = {v["name"]: v for v in json.loads((kit / "videos.json").read_text())}
    res = videos["video-results"]
    assert res["edition"] == "results" and videos["video-round"]["edition"] == "predictions"
    lines = res["scenes"][0]["lines"]
    assert lines[0].startswith("Matchweek 9: we got") and "of two right" in lines[0]
    assert any(l.startswith("Best call: Strong against Weak") for l in lines)
    page = (kit / "videos" / "video-results.html").read_text()
    assert "Strong 2–0 Weak" in page.replace("\n", " ") or "2–0" in page
    posts = {p["day"]: p for p in json.loads((kit / "posts.json").read_text())["posts"]}
    results = next(p for p in posts.values() if p["when"].endswith("the results"))
    assert results["video"] == "video-results"
    fake_render(kit)
    (kit / "video").mkdir()
    (kit / "video" / "video-results.mp4").write_bytes(b"fake mp4")
    folders = social.ready_to_post(kit)
    assert any(p.name.startswith("matchweek-9-results-") for f in folders for p in f.glob("*.mp4"))


def test_an_unfinished_round_has_no_results_video(tmp_path):
    kit = played_round(tmp_path, both=False)
    assert "video-results" not in {v["name"] for v in json.loads((kit / "videos.json").read_text())}


def test_render_can_keep_to_one_edition(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    (kit / "videos.json").write_text(json.dumps([{"name": "a", "edition": "predictions"},
                                                 {"name": "b", "edition": "results"}]))
    assert video.render(kit, editions=[]) == []  # nothing due: nothing rendered, no browser started


def test_instagram_captions_open_on_the_hook_and_ask_a_question(tmp_path):
    kit = site.build(make_data(tmp_path), tmp_path / "site") / "social"
    posts = json.loads((kit / "posts.json").read_text())["posts"]
    game = next(p for p in posts if p.get("video") == "video-strong-v-weak")
    hook = next(v for v in json.loads((kit / "videos.json").read_text()) if v["name"] == "video-strong-v-weak")
    first = hook["scenes"][0]
    if first["kind"] == "hook":
        assert game["instagram"].startswith(first["lines"][0])
    assert "👇" in game["instagram"] and social.FOLLOW in game["instagram"]
    assert "👇" in posts[0]["instagram"]
