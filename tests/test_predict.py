import json

import pandas as pd

from plpred import predict
from tests.test_model import synthetic_matches


def test_after_kickoff_calls_only_started_games_from_pre_round_results(tmp_path):
    m = synthetic_matches()
    m["season"] = 2026
    (tmp_path / "upcoming").mkdir()
    m.to_csv(tmp_path / "matches.csv", index=False)
    start = m.date.max() + pd.Timedelta(days=3)
    pd.DataFrame({"fixture_id": [1, 2], "kickoff": [start, start + pd.Timedelta(days=2)],
                  "round": ["Regular Season - 7"] * 2, "status": ["FT", "NS"],
                  "home_team": ["Strong", "Average"], "away_team": ["Weak", "Good"],
                  "home_goals": [3, None], "away_goals": [0, None]}).to_csv(tmp_path / "upcoming" / "fixtures.csv", index=False)

    predict.main(["--data", str(tmp_path), "--after-kickoff"])
    late = json.loads((tmp_path / "predictions" / f"{start.date()}-after-kickoff.json").read_text())
    assert [(g["home_team"], g["after_kickoff"], g["result"]) for g in late["games"]] == [("Strong", True, [3, 0])]

    predict.main(["--data", str(tmp_path)])
    live = json.loads((tmp_path / "predictions" / f"{(start + pd.Timedelta(days=2)).date()}.json").read_text())
    assert [g["home_team"] for g in live["games"]] == ["Average"] and "after_kickoff" not in live["games"][0]


def test_if_due_calls_only_an_uncalled_round_starting_tomorrow(tmp_path):
    (tmp_path / "upcoming").mkdir()
    (tmp_path / "predictions").mkdir()
    pd.DataFrame({"gameweek": [9, 10], "kickoff": ["2026-10-24T14:00:00Z", "2026-10-28T19:00:00Z"],
                  "home_team": ["Strong", "Weak"], "away_team": ["Weak", "Good"], "finished": [True, False]}
                 ).to_csv(tmp_path / "upcoming" / "fpl_fixtures.csv", index=False)
    pd.DataFrame({"round": ["Regular Season - 10"]}).to_csv(tmp_path / "upcoming" / "fixtures.csv", index=False)
    day = pd.Timestamp("2026-10-27").date()
    assert predict.due(tmp_path, day)
    assert not predict.due(tmp_path, pd.Timestamp("2026-10-26").date())  # nothing starts tomorrow
    (tmp_path / "predictions" / "2026-10-28.json").write_text(json.dumps(
        {"games": [{"home_team": "Weak", "away_team": "Good", "kickoff": "2026-10-28T19:00:00+00:00"}]}))
    assert not predict.due(tmp_path, day)  # already called on Thursday: the calls stay frozen
    (tmp_path / "predictions" / "2026-10-28.json").unlink()
    pd.DataFrame({"round": ["Regular Season - 9"]}).to_csv(tmp_path / "upcoming" / "fixtures.csv", index=False)
    assert not predict.due(tmp_path, day)  # the fetched fixtures are still last round's
