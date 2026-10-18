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


def test_build_writes_home_week_and_match_pages(tmp_path):
    out = site.build(make_data(tmp_path), tmp_path / "site")
    home = (out / "index.html").read_text()
    assert "Matchweek 9" in home and 'href="./matchweek/9/strong-v-weak.html"' in home  # the front page is this week
    assert 'href="./about/index.html"' in home
    about = (out / "about" / "index.html").read_text()
    assert "Where the data comes from" in about and "Understat" in about
    assert 'href="../index.html"' in about
    assert "Inkwell <span class=\"amp\">&amp;</span> Algorithm" in home and (out / "favicon.svg").exists()
    week = (out / "matchweek" / "9" / "index.html").read_text()
    assert "Matchweek 9" in week
    assert "Replay" not in week.split("<body>")[1].split("</header>")[0]
    assert (out / "badges" / "strong.png").exists()
    assert 'src="../../badges/strong.png"' in week and 'href="../../matchweek/9/strong-v-weak.html"' in week
    page = (out / "matchweek" / "9" / "strong-v-weak.html").read_text()
    assert "How we got to" in page and "../../style.css" in page


def test_each_matchweek_keeps_its_own_page(tmp_path):
    data = make_data(tmp_path)
    first = json.loads((data / "predictions" / "2026-10-24.json").read_text())
    first.update(round="Regular Season - 8", as_of="2026-12-24")
    (data / "predictions" / "2026-10-17.json").write_text(json.dumps(first))
    out = site.build(data, tmp_path / "site")
    assert (out / "matchweek" / "8" / "strong-v-weak.html").exists()
    home = (out / "about" / "index.html").read_text()
    assert home.index("Matchweek 9</span>") < home.index("Matchweek 8</span>")  # newest first
    week8 = (out / "matchweek" / "8" / "index.html").read_text()
    assert 'href="../../matchweek/9/index.html"' in week8


def test_replay_files_are_used_only_when_nothing_else_exists(tmp_path):
    data = make_data(tmp_path)
    shutil.copy(data / "predictions" / "2026-10-24.json", data / "predictions" / "replay-2026-10-31.json")
    preds, is_replay = site.latest_predictions(data)
    assert not is_replay
    (data / "predictions" / "2026-10-24.json").unlink()
    assert site.latest_predictions(data)[1]


def test_record_counts_strong_calls_against_the_bookies(tmp_path):
    (tmp_path / "backtest").mkdir()
    rows = ["season,gameweek,date,home_team,away_team,result,home_goals,away_goals,p_home,p_draw,p_away,call,score,mkt_home,mkt_draw,mkt_away",
            "2026,1,2026-08-15,A,B,H,2,0,0.60,0.25,0.15,H,2-0,0.65,0.20,0.15",
            "2026,1,2026-08-15,C,D,A,0,1,0.56,0.24,0.20,H,1-0,0.30,0.30,0.40",
            "2026,1,2026-08-15,E,F,H,1,0,0.40,0.30,0.30,H,1-0,0.45,0.30,0.25"]
    (tmp_path / "backtest" / "predictions_2026.csv").write_text("\n".join(rows))
    r = site.record(tmp_path, 2026)
    assert r["strong"] == {"n": 2, "right": 1, "rate": 50, "fav_rate": 100}


def test_games_called_after_kickoff_join_the_live_calls_and_are_labelled(tmp_path):
    data = make_data(tmp_path)
    live = json.loads((data / "predictions" / "2026-10-24.json").read_text())
    late = {**live, "as_of": "2026-12-30", "games": [{**g, "after_kickoff": True, "result": [2, 0]}
                                                      for g in live["games"]]}
    live["games"] = live["games"][1:]  # only the second game was still to come
    (data / "predictions" / "2026-10-24.json").write_text(json.dumps(live))
    (data / "predictions" / "2026-10-23-after-kickoff.json").write_text(json.dumps(late))
    out = site.build(data, tmp_path / "site")
    week = (out / "matchweek" / "9" / "index.html").read_text()
    assert "after 1 of its 2 games had kicked off" in week and "Final 2–0" in week
    assert "After kick-off" not in (out / "matchweek" / "9" / "average-v-good.html").read_text()
    assert "test, not a prediction" in (out / "matchweek" / "9" / "strong-v-weak.html").read_text()
    kit = (out / "social" / "index.html").read_text()
    assert "would have called" in kit and "not part of our record" in kit
