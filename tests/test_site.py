import json
import shutil
from pathlib import Path

from plpred import explain, model, site
from tests.test_model import synthetic_matches


def make_data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    m = synthetic_matches()
    m["season"] = 2026
    m.to_csv(data / "matches.csv", index=False)
    ratings = model.fit(m)
    games = []
    for home, away in [("Strong", "Weak"), ("Average", "Good")]:
        g = explain.explain(ratings, m, home, away, None, (2.0, 3.4, 3.8), 2026)
        g["kickoff"] = "2026-10-24T14:00:00+00:00"
        g["venue"] = None
        games.append(g)
    (data / "predictions").mkdir()
    (data / "predictions" / "2026-10-24.json").write_text(json.dumps(
        {"as_of": "2026-12-31", "season": "2026/27", "round": "Regular Season - 9", "games": games}))
    (data / "badges").mkdir()
    (data / "badges" / "strong.png").write_bytes(b"\x89PNG")
    return data


def test_build_writes_front_page_and_match_pages(tmp_path):
    out = site.build(make_data(tmp_path), tmp_path / "site")
    index = (out / "index.html").read_text()
    assert "Matchweek 9" in index
    assert "Replay" not in index.split("<body>")[1].split("</header>")[0]
    assert (out / "match" / "strong-v-weak.html").exists()
    assert (out / "badges" / "strong.png").exists()
    assert 'src="./badges/strong.png"' in index
    page = (out / "match" / "strong-v-weak.html").read_text()
    assert "How we got to" in page and "../style.css" in page


def test_replay_files_are_used_only_when_nothing_else_exists(tmp_path):
    data = make_data(tmp_path)
    shutil.copy(data / "predictions" / "2026-10-24.json", data / "predictions" / "replay-2026-10-31.json")
    preds, is_replay = site.latest_predictions(data)
    assert not is_replay
    (data / "predictions" / "2026-10-24.json").unlink()
    assert site.latest_predictions(data)[1]
