import json

import numpy as np
import pandas as pd
import pytest

from plpred import explain, luck, model, site
from tests.test_model import synthetic_matches
from tests.test_site import make_data


def test_expected_points_favour_the_side_with_more_xg():
    h, a = luck.match_xpts(2.0, 0.5)
    assert h > 2 and a < 0.6
    even = luck.match_xpts(1.2, 1.2)
    assert even[0] == pytest.approx(even[1])


def test_deserved_table_and_recent_luck():
    m = synthetic_matches()
    t = luck.deserved_table(m, 2025)
    assert [r["position"] for r in t] == list(range(1, len(t) + 1))
    assert t[0]["xpts"] >= t[-1]["xpts"]
    assert sorted(r["actual"] for r in t) == list(range(1, len(t) + 1))
    games = luck.team_games(m)
    r = luck.recent(games, "Strong")
    assert r["games"] == luck.RECENT_GAMES and r["difference"] == pytest.approx(r["points"] - r["xpts"], abs=0.11)
    assert luck.luck_gap(games, "Strong", "Weak") is not None
    assert luck.luck_gap(games.iloc[:0], "Strong", "Weak") is None


def test_lucky_side_gets_a_results_reason():
    m = synthetic_matches()
    ratings = model.fit(m)
    r = explain.explain(ratings, m, "Strong", "Weak", None, None, 2025)
    r["luck"] = {"home": {"games": 10, "points": 25, "xpts": 18.0, "difference": 7.0}, "away": None}
    reason = explain.results_vs_performance(r)
    assert reason["title"] == "Strong's results flatter them by 7 points"
    assert "Strong get less credit" in reason["text"]
    r["luck"]["home"]["difference"] = 1.0
    assert explain.results_vs_performance(r) is None


def test_power_method_removes_the_margin_unevenly():
    odds = np.array([[1.5, 4.2, 7.0]])
    p = luck.power_margin(odds)
    assert p.sum() == pytest.approx(1)
    plain = (1 / odds) / (1 / odds).sum()
    assert p[0, 0] > plain[0, 0] and p[0, 2] < plain[0, 2]  # outsiders lose more of the margin


def test_preregistered_test_counts_only_live_calls_in_the_gap(tmp_path):
    data = make_data(tmp_path)
    matches = pd.read_csv(data / "matches.csv", parse_dates=["date"])
    last = matches.iloc[-1]
    game = {"home_team": last.home_team, "away_team": last.away_team, "kickoff": f"{last.date.date()}T15:00:00+00:00",
            "probabilities": {"H": 0.5, "D": 0.3, "A": 0.2}, "luck_gap": 0.6}
    calm = dict(game, home_team=matches.iloc[-2].home_team, away_team=matches.iloc[-2].away_team, luck_gap=0.1)
    for name in ("2026-10-24.json", "replay-2026-10-25.json", "2026-10-01.json"):
        (data / "predictions" / name).write_text(json.dumps({"games": [game, calm]}))
    out = luck.preregistered(data, matches)
    assert out["n"] == 1 and out["scored"] == 2
    assert out["us_rps"] >= 0 and out["us_right"] <= 1


def test_record_page_is_built(tmp_path):
    data = make_data(tmp_path)
    m = pd.read_csv(data / "matches.csv", parse_dates=["date"])
    (data / "backtest").mkdir()
    m.assign(p_home=0.5, p_draw=0.3, p_away=0.2, call="H", score="1-0",
             mkt_home=0.45, mkt_draw=0.3, mkt_away=0.25).to_csv(data / "backtest" / "predictions_2026.csv", index=False)
    out = site.build(data, tmp_path / "site")
    page = (out / "record" / "index.html").read_text()
    assert "When we say 60%, does it happen?" in page and "Our public test" in page
    assert 'href="../record/"' in (out / "about" / "index.html").read_text()
    assert "The table they deserved" in (out / "matchweek" / "9" / "index.html").read_text()
