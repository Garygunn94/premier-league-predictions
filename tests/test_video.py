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
    assert kinds == ["hook", "clues", "call", "outro"]  # every game's call is revealed at the end
    assert 2 <= len(game["scenes"][1]["lines"]) <= 3  # enough substance before the reveal
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


def test_a_reason_is_said_as_a_place_in_the_league():
    assert video.rank_words(1, 20, "attack") == "the best attack"
    assert video.rank_words(5, 20, "attack") == "the fifth-best attack"
    assert video.rank_words(18, 20, "defence") == "the third-worst defence"
    assert video.rank_words(20, 20, "defence") == "the worst defence"
    order = [f"T{i}" for i in range(1, 21)]
    assert [r["rank"] for r in video.ladder(order, ["T2"], "top")] == [1, 2, 3, 4, 5]
    rows = video.ladder(order, ["T9"], "top")
    assert [r["rank"] for r in rows] == [1, 2, 3, 9] and rows[-1]["gap"] and rows[-1]["hot"]
    assert [r["rank"] for r in video.ladder(order, ["T18"], "bottom")] == [16, 17, 18, 19, 20]
    assert [r["rank"] for r in video.ladder(order, ["T6", "T8"], None)] == [5, 6, 7, 8, 9]


def test_match_reel_is_one_idea_a_scene(tmp_path):
    kit = site.build(make_data(tmp_path), tmp_path / "site") / "social"
    for v in json.loads((kit / "videos.json").read_text()):
        if v["name"].startswith("video-") and v["name"] not in ("video-round", "video-results"):
            assert len(v["scenes"]) <= 4
            text = " ".join(line for s in v["scenes"] for line in s["lines"])
            # no win percentages, expected goals or "worth N points" read out
            assert "percent" not in text and "expected goals" not in text and " points" not in text


def test_captions_give_the_reason_as_a_fan_would_say_it():
    v = {"home_team": "Arsenal", "away_team": "Everton", "call": "H",
         "ranks": {"attack": ["Man City", "Arsenal", "Everton"], "defence": ["Arsenal", "Man City", "Everton"],
                   "overall": ["Arsenal", "Man City", "Everton"]},
         "waterfall": {"steps": [{"label": "Arsenal's attack", "kind": "attack", "team": "Arsenal", "change": 11,
                                  "detail": "Arsenal create and score 32% more than an average side"}]}}
    assert video.reason_text(v) == "Arsenal have the second-best attack in the league."
    assert social.reason(dict(v, front_reasons=[])) == "Arsenal have the second-best attack in the league."
    # calls made before the ranks were kept: still no percentages or points
    assert video.reason_text(dict(v, ranks=None)) == "Arsenal score more than most."


def test_a_reason_never_points_the_wrong_way():
    teams = [f"T{i}" for i in range(1, 21)]
    order = lambda *front: [*front, *[t for t in teams if t not in front]]
    v = {"home_team": "Newcastle", "away_team": "Everton", "call": "H",
         "ranks": {"defence": order("T1", "T2", "Everton"), "attack": order("Newcastle"),
                   "overall": order("Newcastle", "T1", "Everton")},
         "waterfall": {"steps": [{"kind": "defence", "team": "Everton", "change": 5,
                                  "detail": "Everton concede 15% more than an average side"},
                                 {"kind": "attack", "team": "Newcastle", "change": 3,
                                  "detail": "Newcastle create and score 20% more than an average side"}]}}
    # Everton's defence is third-best, so that can't be why we back Newcastle: the next reason is
    assert video.reason_text(v) == "Newcastle have the best attack in the league."
    # a middling place explains nothing, so the reason becomes the overall ratings
    v["ranks"]["attack"] = order(*teams[:8], "Newcastle")
    assert video.reason_text(v) == "Our model rates Newcastle first and Everton third: almost level."
    v["ranks"]["overall"] = order("Newcastle", *teams[:8], "Everton")
    assert video.reason_text(v) == "Our model rates Newcastle first and Everton tenth overall."


def test_every_game_keeps_the_call_for_the_end(tmp_path):
    kit = site.build(make_data(tmp_path), tmp_path / "site") / "social"
    games = [v for v in json.loads((kit / "videos.json").read_text()) if v.get("style")]
    assert games and {v["style"] for v in games} == {"reveal"}
    for v in games:
        assert v["scenes"][0]["kind"] == "hook" and len(v["scenes"][0]["lines"]) == 2  # the hook, then its question
        assert [s["kind"] for s in v["scenes"]][-2:] == ["call", "outro"]
        assert not any(l.startswith(("The reason", "Why?")) for s in v["scenes"] for l in s["lines"])


def test_upfront_gives_the_call_straight_after_the_opener():
    v = {"home_team": "Leeds", "away_team": "Sunderland", "call": "H", "confidence": "solid", "our_p": 50,
         "score_text": "2–1", "headline": "Nothing to split them", "kickoff_long": "Saturday 15:00"}
    scenes = video.match_scenes(v, 9, 0, style="upfront")
    assert [s["kind"] for s in scenes] == ["hook", "call", "outro"] and len(scenes[0]["lines"]) == 1


def test_reveal_captions_dont_give_the_call_away(tmp_path):
    kit = site.build(make_data(tmp_path), tmp_path / "site") / "social"
    posts = [p for p in json.loads((kit / "posts.json").read_text())["posts"] if p.get("style")]
    assert posts and all(p["reel"] and "Our call:" not in p["reel"] and p["reel"].count("#") <= 5 for p in posts)
    fake_render(kit)
    (kit / "video").mkdir()
    for v in json.loads((kit / "videos.json").read_text()):
        (kit / "video" / f"{v['name']}.mp4").write_bytes(b"fake mp4")
    folders = social.ready_to_post(kit)
    game = next(f for f in folders if list(f.glob("strong-v-weak-*.mp4")))
    [clip] = game.glob("*.mp4")
    assert clip.stem.endswith("-reveal") and (game / "reel.txt").exists()
    msgs = social.kit_emails(kit, "me@example.com", "me@example.com")
    assert any("---- REEL AND TIKTOK ----" in m.get_body().get_content() for m in msgs)


def test_hooks_raise_a_question_and_a_fixture_is_the_last_resort():
    base = {"home_team": "Leeds", "away_team": "Sunderland", "call": "H", "confidence": "too close to call",
            "headline": "Nothing to split them", "mkt": None,
            "standings": {"home": {"position": 8, "points": 11}, "away": {"position": 11, "points": 10}},
            "form": {"home": [{"result": r} for r in "WLWLD"], "away": [{"result": r} for r in "WDLLD"]}}
    kinds = [h[0] for h in video.hooks(base)]
    assert "points" in kinds and "form" in kinds  # "haven't won in three", "just one point between"
    picks = video.pick_hooks([base] * 4)
    assert all(h[0] != "fixture" for h in picks)
    say = video.Phrasing(9, 0)
    assert video.tease(("form", 0.6, "Sunderland haven't won in three."), base, say).startswith("Is this where it turns?")
    assert video.tease(("luck", 0.6, "Forest deserve 10 more points than they've got."), base, say).startswith(
        "Is their luck about to turn?")


def test_clues_give_two_or_three_facts_before_the_reveal():
    v = {"home_team": "Newcastle", "away_team": "Everton", "confidence": "too close to call", "call": "H",
         "waterfall": {"steps": [
             {"kind": "defence", "team": "Newcastle", "change": -7.1, "detail": "Newcastle concede 33% more than an average side"},
             {"kind": "defence", "team": "Everton", "change": 5.4, "detail": "Everton concede 15% more than an average side"},
             {"kind": "attack", "team": "Newcastle", "change": 4.1, "detail": "Newcastle create and score 12% more than an average side"},
             {"kind": "attack", "team": "Everton", "change": 1.8, "detail": "Everton create and score 7% less than an average side"}]},
         "form": {}, "standings": {}}
    assert video.clues(v, "points") == ["Everton concede more than most.", "Newcastle score more than most.",
                                        "But Newcastle concede far more than most."]
    v["ranks"] = {"defence": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
                              "Everton", "R", "S", "T"], "attack": []}
    assert video.clues(v, "points")[0] == "Everton have the fourth-worst defence in the league."
