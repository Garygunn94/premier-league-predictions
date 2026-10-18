import numpy as np
import pandas as pd
import pytest

from plpred import backtest, model

TEAMS = ["Strong", "Good", "Average", "Weak"]
STRENGTH = {"Strong": 0.6, "Good": 0.2, "Average": 0.0, "Weak": -0.6}


def synthetic_matches(rounds=12, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, day = [], pd.Timestamp("2025-08-16")
    for r in range(rounds):
        for home in TEAMS:
            for away in TEAMS:
                if home == away:
                    continue
                lam = np.exp(0.2 + STRENGTH[home] - STRENGTH[away] * 0.5)
                mu = np.exp(STRENGTH[away] - STRENGTH[home] * 0.5)
                hg, ag = rng.poisson(lam), rng.poisson(mu)
                rows.append({"season": 2025, "date": day + pd.Timedelta(days=7 * r), "home_team": home,
                             "away_team": away, "home_goals": hg, "away_goals": ag,
                             "home_xg": lam, "away_xg": mu,
                             "result": "H" if hg > ag else "A" if ag > hg else "D",
                             "odds_home": 2.0, "odds_draw": 3.4, "odds_away": 3.8})
    return pd.DataFrame(rows)


def test_fit_ranks_teams_by_true_strength():
    ratings = model.fit(synthetic_matches())
    assert list(ratings.table().team) == TEAMS
    assert ratings.home_adv > 0
    assert abs(ratings.attack.mean()) < 1e-6


def test_fit_ignores_matches_on_or_after_as_of():
    m = synthetic_matches()
    cutoff = m.date.iloc[len(m) // 2]
    assert model.fit(m, as_of=cutoff).matches_used == (m.date < cutoff).sum()


def test_score_matrix_is_a_distribution():
    probs = model.score_matrix(1.6, 1.1, -0.08)
    assert probs.shape == (model.MAX_GOALS + 1,) * 2
    assert probs.sum() == pytest.approx(1)
    assert (probs >= 0).all()


def test_prediction_score_agrees_with_call():
    ratings = model.fit(synthetic_matches())
    p = model.predict(ratings, "Strong", "Weak")
    assert p.call == "H" and p.score[0] > p.score[1]
    assert p.p_home + p.p_draw + p.p_away == pytest.approx(1)
    p = model.predict(ratings, "Weak", "Strong")
    assert p.call == "A" and p.score[0] < p.score[1]


def test_unknown_team_is_rated_like_the_three_weakest():
    ratings = model.fit(synthetic_matches())
    assert not ratings.knows("Promoted")
    weakest = [ratings.rating(t) for t in ["Good", "Average", "Weak"]]
    assert ratings.rating("Promoted") == pytest.approx(tuple(np.mean(weakest, axis=0)))
    model.predict(ratings, "Strong", "Promoted")  # predicts without error


def test_backtest_beats_coin_flip_and_scores_sensibly():
    m = synthetic_matches(rounds=20)
    m.loc[m.date >= "2025-11-01", "season"] = 2026
    preds = backtest.run(m, [2026], model.Settings())
    s = backtest.score(preds)
    assert s["matches"] == (m.season == 2026).sum()
    assert s["accuracy"] > 0.45
    assert 0 < s["rps"] < 0.25
    summary = backtest.report(preds)
    assert set(summary.index) == {"Model", "Bookmakers", "Always home"}
