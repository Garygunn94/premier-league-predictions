import pandas as pd
import pytest

from plpred import explain, model
from tests.test_model import synthetic_matches


def players():
    rows = [
        ("Strong", "Star", "FWD", "i", 0, 900, 8.0),
        ("Strong", "Keeper", "GK", "a", None, 900, 0.0),
        ("Strong", "Mid", "MID", "a", None, 900, 2.0),
        ("Weak", "Back", "DEF", "d", 50, 900, 0.5),
        ("Weak", "Forward", "FWD", "a", None, 900, 3.0),
    ]
    return pd.DataFrame(rows, columns=["team", "player", "position", "status", "chance_next_round",
                                       "minutes", "expected_goal_involvements"]).assign(news="")


@pytest.fixture(scope="module")
def setup():
    m = synthetic_matches()
    return m, model.fit(m)


def test_waterfall_adds_up_to_the_probability(setup):
    m, ratings = setup
    r = explain.explain(ratings, m, "Strong", "Weak", players(), (1.5, 4.0, 6.0), 2025)
    w = r["waterfall"]
    assert w["start"] + sum(s["change"] for s in w["steps"]) == pytest.approx(w["total"], abs=0.3)
    assert w["total"] == pytest.approx(r["probabilities"][r["call"]] * 100, abs=0.1)
    assert r["call"] == "H"
    assert sum(r["market"].values()) == pytest.approx(1, abs=0.01)


def test_shapley_shares_sum_exactly():
    fs = [explain.Factor("a", "", log_home=0.3), explain.Factor("b", "", log_away=-0.2),
          explain.Factor("c", "", log_home=-0.1, log_away=0.1)]
    shares = explain.contributions(1.3, 1.0, -0.08, fs, "H")
    start = explain.probabilities(1.3, 1.0, -0.08)["H"]
    end = explain.probabilities(1.3 * 2.718281828 ** 0.2, 1.0 * 2.718281828 ** -0.1, -0.08)["H"]
    assert start + sum(shares) == pytest.approx(end)


def test_missing_star_weakens_attack():
    attack, conceded, rows = explain.team_news(players(), "Strong")
    assert attack == pytest.approx(1 - explain.MAX_NEWS_EFFECT)  # 80% of xGI missing hits the cap
    assert rows[0]["player"] == "Star" and rows[0]["impact"] == "Big"


def test_doubtful_defender_counts_by_chance_of_playing():
    attack, conceded, rows = explain.team_news(players(), "Weak")
    assert attack == pytest.approx(1 - 0.5 * 0.5 * 0.5 / 3.5)
    assert conceded == pytest.approx(1 + 0.15 * 0.5)
    assert rows[0]["status"] == "Doubtful, 50% to play"


def test_reasons_and_words(setup):
    m, ratings = setup
    r = explain.explain(ratings, m, "Strong", "Weak", players(), None, 2025)
    assert r["market"] is None
    topics = [x["topic"] for x in r["reasons"]]
    assert topics[:3] == ["Chances created and allowed", "Team news", "Form"]
    assert "Strong" in r["headline"]
    assert r["change_our_mind"].endswith(".")
    assert r["standings"]["home"]["position"] == 1


def test_confidence_labels():
    assert explain.confidence({"H": 0.40, "D": 0.25, "A": 0.36}) == "too close to call"
    assert explain.confidence({"H": 0.56, "D": 0.24, "A": 0.2}) == "strong"
    assert explain.confidence({"H": 0.5, "D": 0.25, "A": 0.25}) == "solid"
    assert explain.confidence({"H": 0.42, "D": 0.3, "A": 0.28}) == "too close to call"


def test_possessive():
    assert explain._pos("Spurs") == "Spurs'"
    assert explain._pos("Arsenal") == "Arsenal's"
