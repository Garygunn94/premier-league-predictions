import pandas as pd

from plpred import update
from plpred.sources import api_football


class FakeClient:
    remaining = "90"

    def __init__(self):
        self.odds_calls = []


def test_odds_fetched_once_per_fixture(tmp_path, monkeypatch):
    client = FakeClient()
    fixtures = pd.DataFrame({"fixture_id": [1, 2], "round": ["Regular Season - 9"] * 2})
    monkeypatch.setattr(api_football, "Client", lambda: client)
    monkeypatch.setattr(api_football, "upcoming_round", lambda c, season: fixtures)

    def fake_odds(c, ids):
        c.odds_calls.append(list(ids))
        return pd.DataFrame({"fixture_id": ids, "odds_home": 2.0, "odds_draw": 3.4, "odds_away": 3.8})

    monkeypatch.setattr(api_football, "odds", fake_odds)
    monkeypatch.setattr(api_football, "save_badges", lambda f, folder: 0)

    update.update_fixtures(tmp_path)
    update.update_fixtures(tmp_path)

    assert client.odds_calls == [[1, 2], []]
    odds = pd.read_csv(tmp_path / "upcoming" / "odds.csv")
    assert sorted(odds.fixture_id) == [1, 2]


def test_free_sources_take_over_when_api_football_refuses(tmp_path, monkeypatch):
    from plpred.sources import football_data_uk, fpl

    def refuse():
        raise RuntimeError("Free plans do not have access to this season")

    monkeypatch.setattr(api_football, "Client", refuse)
    fixtures = pd.DataFrame({"fixture_id": [70, 71], "round": ["Regular Season - 7"] * 2,
                             "home_team": ["Arsenal", "Tottenham"], "away_team": ["Leeds", "Coventry"]})
    monkeypatch.setattr(fpl, "fetch_raw", lambda: {})
    monkeypatch.setattr(fpl, "fetch_fixtures_raw", lambda: [])
    monkeypatch.setattr(fpl, "round_fixtures", lambda raw, games: fixtures)
    monkeypatch.setattr(api_football, "save_badges", lambda f, folder: 0)
    prices = [pd.DataFrame({"home_team": ["Tottenham"], "away_team": ["Coventry"],
                            "odds_home": [1.6], "odds_draw": [4.2], "odds_away": [5.5]}),
              pd.DataFrame({"home_team": ["Tottenham", "Arsenal"], "away_team": ["Coventry", "Leeds"],
                            "odds_home": [9.0, 1.5], "odds_draw": [9.0, 4.0], "odds_away": [9.0, 6.0]})]
    monkeypatch.setattr(football_data_uk, "upcoming_odds", lambda: prices.pop(0))

    update.update_fixtures(tmp_path)
    update.update_fixtures(tmp_path)

    odds = pd.read_csv(tmp_path / "upcoming" / "odds.csv").set_index("fixture_id")
    assert odds.loc[71, "odds_home"] == 1.6  # first prices kept
    assert odds.loc[70, "odds_home"] == 1.5  # priced on the second run
