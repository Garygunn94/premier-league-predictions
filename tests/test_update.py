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

    update.update_fixtures(tmp_path)
    update.update_fixtures(tmp_path)

    assert client.odds_calls == [[1, 2], []]
    odds = pd.read_csv(tmp_path / "upcoming" / "odds.csv")
    assert sorted(odds.fixture_id) == [1, 2]
