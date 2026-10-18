import json

from plpred import site, social
from tests.test_site import make_data


def test_site_build_writes_the_posting_kit(tmp_path):
    out = site.build(make_data(tmp_path), tmp_path / "site")
    kit = out / "social"
    names = [c["name"] for c in json.loads((kit / "cards.json").read_text())]
    assert names[0] == "round" and "call-strong-v-weak" in names
    assert any(n.startswith("reasons-") for n in names)
    for n in names:
        page = (kit / "cards" / f"{n}.html").read_text()
        assert 'class="amp"' in page and "../../style.css" in page
    index = (kit / "index.html").read_text()
    assert "Thursday" in index and "Friday" in index and "png/round.png" in index


def test_every_game_gets_a_carousel(tmp_path):
    out = site.build(make_data(tmp_path), tmp_path / "site")
    kit = out / "social"
    names = {c["name"] for c in json.loads((kit / "cards.json").read_text())}
    slugs = [n.removeprefix("call-") for n in names if n.startswith("call-")]
    assert slugs
    for s in slugs:
        assert {f"reasons-{s}", f"numbers-{s}", f"scores-{s}"} <= names
    index = (kit / "index.html").read_text()
    assert "Carousel" in index and "png/numbers-" in index


def test_captions_keep_links_out_of_the_main_x_post():
    v = {"home_team": "Arsenal", "away_team": "Liverpool", "confidence": "strong", "call": "H",
         "call_long": "Arsenal win", "score_text": "2–1", "our_p": 58, "kickoff_long": "Saturday 24 October",
         "front_reasons": [{"text": "Arsenal concede 27% less. Worth +6 points."}]}
    post = social.match_post(v)
    assert post.endswith("#ARSLIV") and "http" not in post and social.x_length(post) <= social.X_LIMIT
    ig = social.carousel_caption(v)
    tags = [w for w in ig.split() if w.startswith("#")]
    assert len(tags) <= 5 and "#Arsenal" in tags and "#LFC" in tags
    assert social.match_tag({"home_team": "Somewhere Town", "away_team": "Hull"}) == "#SOMHUL"


def test_x_posts_fit_the_limit():
    long = "Why: " + "word " * 80
    post = social.fit_x(long.strip(), "https://example.com/" + "a" * 100)
    assert social.x_length(post) <= social.X_LIMIT
    assert post.endswith("a" * 100) and "…" in post
    assert social.x_length("see https://example.com/" + "a" * 100) == 4 + social.X_URL_LENGTH


def fake_render(kit):
    (kit / "png").mkdir(exist_ok=True)
    for c in json.loads((kit / "cards.json").read_text()):
        (kit / "png" / f"{c['name']}.png").write_bytes(b"\x89PNG fake")


def test_kit_is_laid_out_one_folder_per_post(tmp_path):
    kit = site.build(make_data(tmp_path), tmp_path / "site") / "social"
    fake_render(kit)
    folders = social.ready_to_post(kit)
    assert folders[0].name == "01-thursday"
    game = next(f for f in folders if (f / "how.txt").read_text().startswith("Carousel"))
    images = sorted(p.name for p in game.glob("*.png"))
    assert images[0].startswith("01-call-") and images[1].startswith("02-reasons-")
    assert "#PremierLeague" in (game / "instagram.txt").read_text()
    assert "[First reply]" in (game / "x.txt").read_text()
    assert "[Post 1 of" in (folders[0] / "x.txt").read_text()


def test_kit_emails_carry_captions_and_images(tmp_path, monkeypatch):
    kit = site.build(make_data(tmp_path), tmp_path / "site") / "social"
    fake_render(kit)
    folders = social.ready_to_post(kit)
    msgs = social.kit_emails(kit, "me@example.com", "me@example.com")
    assert len(msgs) == len(folders)
    first = msgs[0]
    assert first["Subject"].startswith("Matchweek ") and f"1 of {len(folders)} · Thursday" in first["Subject"]
    body = first.get_body(("plain",)).get_content()
    assert "INSTAGRAM" in body and "---- X ----" in body
    names = [a.get_filename() for a in first.iter_attachments()]
    assert names[0] == "01-round.png" and all(n.endswith(".png") for n in names)
    monkeypatch.delenv("KIT_EMAIL", raising=False)
    assert social.email_kit(kit) == 0


def test_results_posts_score_the_published_calls(tmp_path):
    import pandas as pd
    data = make_data(tmp_path)
    (data / "upcoming").mkdir()
    pd.DataFrame({"fixture_id": [1], "status": ["FT"], "home_team": ["Strong"], "away_team": ["Weak"],
                  "home_goals": [2], "away_goals": [0]}).to_csv(data / "upcoming" / "fixtures.csv", index=False)
    kit = site.build(data, tmp_path / "site") / "social"
    names = [c["name"] for c in json.loads((kit / "cards.json").read_text())]
    assert {"results", "results-list", "verdict-strong-v-weak"} <= set(names)
    assert "verdict-average-v-good" not in names  # not played yet
    posts = {p["day"]: p for p in json.loads((kit / "posts.json").read_text())["posts"]}
    assert "1 of 1 results right" in posts["Monday"]["instagram"]
    assert posts["Result: Strong v Weak"]["x"][0].startswith("Strong 2–0 Weak\nWe said Strong win")
