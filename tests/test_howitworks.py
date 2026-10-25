import json

import numpy as np
import pandas as pd

from plpred import howitworks
from tests.test_dna import league


def matches(fouls_now=12) -> pd.DataFrame:
    """The Club DNA test league, with results, and every side fouling fouls_now times a game this season."""
    m = league()
    m["result"] = np.select([m.home_goals > m.away_goals, m.home_goals == m.away_goals], ["H", "D"], "A")
    now = m.season == 2026
    m.loc[now, "home_fouls"] = fouls_now
    m.loc[now, "away_fouls"] = fouls_now
    return m


def data_dir(tmp_path, m: pd.DataFrame):
    m.to_csv(tmp_path / "matches.csv", index=False)
    (tmp_path / "predictions").mkdir()
    (tmp_path / "backtest").mkdir()
    game = {"home_team": "Wall", "away_team": "Hacker", "call": "H", "probabilities": {"H": 0.62, "D": 0.22, "A": 0.16},
            "score": [1, 0], "waterfall": {"start": 40.1, "steps": [
                {"label": "Wall's defence", "detail": "Wall concede 50% less than an average side", "kind": "defence",
                 "team": "Wall", "change": 15.0},
                {"label": "Hacker's attack", "detail": "Hacker create and score 10% more than an average side",
                 "kind": "attack", "team": "Hacker", "change": -3.2}]}}
    weaker = {**game, "home_team": "Club01", "away_team": "Club02", "probabilities": {"H": 0.4, "D": 0.3, "A": 0.3}}
    (tmp_path / "predictions" / "2026-10-23.json").write_text(json.dumps(
        {"round": "Regular Season - 8", "games": [weaker, game]}))
    (tmp_path / "predictions" / "replay-2026-10-30.json").write_text(json.dumps({"games": []}))
    rng = np.random.default_rng(1)
    for season in (2025, 2026):
        played = m[m.season == season]
        p = rng.dirichlet([4, 2, 3], size=len(played))
        call = np.array(["H", "D", "A"])[p.argmax(axis=1)]
        pd.DataFrame({"season": season, "gameweek": 1, "date": played.date.values, "home_team": played.home_team.values,
                      "away_team": played.away_team.values, "result": played.result.values,
                      "home_goals": played.home_goals.values, "away_goals": played.away_goals.values,
                      "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2], "call": call, "score": "1-0",
                      "mkt_home": 0.45, "mkt_draw": 0.27, "mkt_away": 0.28}).to_csv(
            tmp_path / "backtest" / f"predictions_{season}.csv", index=False)
    return tmp_path


def test_trends_are_real_changes_not_noise():
    trends = howitworks.league_trends(matches(fouls_now=20))
    assert trends and trends[0]["key"] == "fouls" and trends[0]["up"]
    assert trends[0]["line"].startswith("More fouls: 40.0 a game, up from ")
    assert all(abs(t["z"]) >= howitworks.MIN_Z for t in trends)
    # too few games this season to call anything a trend
    m = matches(fouls_now=20)
    assert howitworks.league_trends(m[(m.season < 2026) | (m.index % 4 == 0)]) == []


def test_the_reel_walks_through_the_process(tmp_path):
    data = data_dir(tmp_path, matches(fouls_now=20))
    sc, facts = howitworks.scenes(data)
    kinds = [s["kind"] for s in sc]
    assert kinds[0] == "hook" and kinds[1] == "data" and kinds[-1] == "outro"
    assert kinds.index("model") < kinds.index("example") < kinds.index("calls") < kinds.index("record")
    lines = [line for s in sc for line in s["lines"]]
    assert any(line.startswith("Step one: the data. 840 games from the last three seasons") for line in lines)
    assert "Take Wall against Hacker, matchweek 8." in lines  # the latest round's most confident call
    assert any("That adds 15 points" in line or "worth 15 points" in line for line in lines)
    assert any(line.startswith("We land on 62 percent") for line in lines)
    assert facts["record"][0]["label"] == "2025/26" and facts["record"][1]["now"]
    # never phrased as betting advice
    text = " ".join(lines).lower()
    assert not any(w in text for w in ("bet", "tip", "stake", "value", "banker", "lock"))


def test_record_verdict_is_plain():
    level = [{"n": 380, "us": 50, "fav": 49}]
    assert howitworks.verdict(level)[0].startswith("Level with the bookies, not ahead")
    assert howitworks.verdict([{"n": 380, "us": 46, "fav": 50}])[0].startswith("A little behind")
    assert "too few games" in howitworks.verdict([{"n": 70, "us": 55, "fav": 49}])[0]


def test_build_writes_script_page_and_captions(tmp_path):
    (tmp_path / "data").mkdir()
    data = data_dir(tmp_path / "data", matches(fouls_now=20))
    v = howitworks.build(data, tmp_path / "out")
    assert (tmp_path / "out" / "videos" / "how-it-works.html").exists()
    assert json.loads((tmp_path / "out" / "videos.json").read_text())[0]["name"] == "how-it-works"
    assert "not betting advice" in v["captions"]["instagram"]
    assert v["captions"]["x"][1].endswith("/about/")
