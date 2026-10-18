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
        assert "Inkwell &amp; Algorithm" in page and "../../style.css" in page
    index = (kit / "index.html").read_text()
    assert "Thursday" in index and "Friday" in index and "png/round.png" in index


def test_x_posts_fit_the_limit():
    long = "Why: " + "word " * 80
    post = social.fit_x(long.strip(), "https://example.com/" + "a" * 100)
    assert social.x_length(post) <= social.X_LIMIT
    assert post.endswith("a" * 100) and "…" in post
    assert social.x_length("see https://example.com/" + "a" * 100) == 4 + social.X_URL_LENGTH
