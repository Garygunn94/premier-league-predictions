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
    assert [s["kind"] for s in game["scenes"]][:2] == ["call", "reasons"]
    assert game["scenes"][0]["lines"][0] == "Strong against Weak."
    posts = json.loads((kit / "posts.json").read_text())["posts"]
    assert posts[0]["video"] == "video-round"
    assert any(p["video"] == "video-strong-v-weak" for p in posts)


def test_voices_are_random_but_stable():
    picks = {video.pick_voice(8, f"video-{i}") for i in range(20)}
    assert picks == set(video.VOICES)
    assert video.pick_voice(8, "video-round") == video.pick_voice(8, "video-round")


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
