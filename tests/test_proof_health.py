import hashlib
import json

import pandas as pd

from plpred import health, proof, site
from tests.test_site import make_data


def test_fingerprints_match_sha256_and_skip_replays(tmp_path):
    data = make_data(tmp_path)
    (data / "predictions" / "replay-2026-10-10.json").write_text("{}")
    rows = proof.proofs(data)
    assert [r["file"] for r in rows] == ["2026-10-24.json"]
    raw = (data / "predictions" / "2026-10-24.json").read_bytes()
    assert rows[0]["sha256"] == hashlib.sha256(raw).hexdigest() and rows[0]["short"] == rows[0]["sha256"][:16]
    assert rows[0]["raw"].endswith("/main/data/predictions/2026-10-24.json")


def test_record_page_x_thread_and_legal_pages(tmp_path):
    data = make_data(tmp_path)
    out = site.build(data, tmp_path / "site")
    short = proof.proofs(data)[0]["short"]
    assert proof.proofs(data)[0]["sha256"] in (out / "record" / "index.html").read_text()
    posts = json.loads((out / "social" / "posts.json").read_text())["posts"]
    assert short in posts[0]["x"][-1]
    for page in ("contact", "privacy", "terms"):
        assert (out / page / "index.html").exists()
    home = (out / "index.html").read_text()
    assert 'href="./privacy/"' in home and "BeGambleAware" in home


def fixtures(data, rows):
    (data / "upcoming").mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["gameweek", "kickoff", "home_team", "away_team", "finished"]).to_csv(
        data / "upcoming" / "fpl_fixtures.csv", index=False)


def test_health_flags_stuck_sources_and_missing_calls(tmp_path):
    data = make_data(tmp_path)
    now = pd.Timestamp("2026-10-22T06:00Z")
    fixtures(data, [
        (8, "2026-10-10T14:00Z", "Strong", "Weak", True),     # finished but missing from matches.csv
        (8, "2026-10-19T14:00Z", "Good", "Average", False),   # FPL never marked it finished
        (9, "2026-10-24T14:00:00Z", "Strong", "Weak", False),  # called
        (9, "2026-10-24T14:00:00Z", "Good", "Strong", False),  # not called
        (10, "2026-10-28T19:00Z", "Weak", "Good", False),     # a later round, called nearer the time
    ])
    found = health.problems(data, now, calls=True)
    assert len(found) == 3
    assert "FPL" in found[0] and "football-data" in found[1] and "1 of the 2 games" in found[2]
    assert len(health.problems(data, now)) == 2
