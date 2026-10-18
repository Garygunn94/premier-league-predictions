import json

import numpy as np
import pandas as pd

from plpred import fantasy, model, pundits, simulate, site, social
from plpred.sources import fpl
from tests.test_model import synthetic_matches
from tests.test_site import make_data
from tests.test_sources import FPL_RAW


def test_simulation_chances_add_up_and_follow_strength():
    history = synthetic_matches(rounds=6)  # last season, for the ratings
    now = synthetic_matches(rounds=1, seed=2).head(4)  # this season: 4 of 12 games played
    now["season"], now["date"] = 2026, history.date.max() + pd.Timedelta(days=7)
    m = pd.concat([history, now], ignore_index=True)
    sim = simulate.run(m, 2026, now.date.max() + pd.Timedelta(days=1), runs=2000)
    early = now
    rows = {r["team"]: r for r in sim["teams"]}
    assert sim["fixtures_left"] == 12 - len(early)
    assert abs(sum(r["title"] for r in sim["teams"]) - 1) < 1e-9
    assert abs(sum(r["relegation"] for r in sim["teams"]) - simulate.RELEGATED) < 1e-9
    assert rows["Strong"]["title"] > rows["Weak"]["title"]
    assert all(1 <= r["low"] <= r["high"] <= 4 for r in sim["teams"])


def test_simulation_changes_and_moves():
    now = {"teams": [{"team": "A", "title": 0.6, "top": 1.0, "relegation": 0.0}]}
    before = {"teams": [{"team": "A", "title": 0.4, "top": 1.0, "relegation": 0.0}]}
    simulate.with_changes(now, before)
    assert now["teams"][0]["title_change"] == 20 and now["teams"][0]["top_change"] == 0
    assert simulate.headline_moves(now) == [{"team": "A", "what": "title", "change": 20, "now": 60}]


def test_fpl_chances_match_the_score_matrix():
    c = fantasy.chances(1.8, 0.6, -0.13)
    m = model.score_matrix(1.8, 0.6, -0.13)
    assert np.isclose(c["home"]["cs"], m[:, 0].sum()) and c["home"]["cs"] > c["away"]["cs"]
    assert c["home"]["win"] > c["away"]["win"] and 0 < c["away"]["two"] < c["home"]["two"]


def test_fpl_fixtures_and_strip():
    raw = [{"event": 9, "kickoff_time": "2026-10-24T14:00:00Z", "team_h": 1, "team_a": 2, "finished": False},
           {"event": None, "kickoff_time": None, "team_h": 2, "team_a": 1, "finished": False}]
    f = fpl.fixtures(FPL_RAW, raw)
    assert list(f.home_team) == ["Arsenal", "Man United"]
    m = synthetic_matches()
    ratings = model.fit(m)
    fixtures = pd.DataFrame([{"gameweek": 9, "home_team": "Strong", "away_team": "Weak", "finished": False},
                             {"gameweek": 10, "home_team": "Weak", "away_team": "Strong", "finished": False}])
    strip = fantasy.strip(fixtures, ratings, [{"home_team": "Strong", "away_team": "Weak"}], weeks=2)
    assert strip["gameweeks"] == [9, 10]
    strong = next(r for r in strip["rows"] if r["team"] == "Strong")
    assert strong["weeks"][0][0]["venue"] == "H" and strong["weeks"][0][0]["band"] == "5"
    assert fantasy.strip(None, ratings, []) is None


def test_pundits_are_scored_on_the_same_games(tmp_path):
    data = make_data(tmp_path)
    matches = pd.read_csv(data / "matches.csv", parse_dates=["date"])
    g = matches.iloc[-1]
    (data / "pundits.csv").write_text(
        "season,matchweek,pundit,source,home_team,away_team,home_goals,away_goals\n"
        f"2026,9,Pundit,https://example.com,{g.home_team},{g.away_team},{g.home_goals},{g.away_goals}\n"
        f"2026,9,Pundit,,Strong,Nobody,1,0\n")
    preds = json.loads((data / "predictions" / "2026-10-24.json").read_text())
    preds["games"][0].update(home_team=g.home_team, away_team=g.away_team)
    board = pundits.scoreboard(data, matches, [preds])
    row = board["pundits"][0]
    assert row["n"] == 1 and row["right"] == 1 and row["exact"] == 1  # the unplayed game isn't scored
    assert len(board["latest"]["rows"]) == 2
    assert pundits.scoreboard(tmp_path, matches, []) == {"pundits": [], "latest": None}


def test_biggest_miss_says_whether_the_chances_backed_us():
    matches = pd.DataFrame([{"season": 2026, "home_team": "A", "away_team": "B", "home_xg": 2.2, "away_xg": 0.6},
                            {"season": 2026, "home_team": "C", "away_team": "D", "home_xg": 0.5, "away_xg": 1.9}])
    row = dict(wrong=True, season=2026, result="0–1", match="", call_code="H", result_code="A")
    rec = {"last": {"rows": [{**row, "home_team": "A", "away_team": "B", "p": 60},
                             {**row, "home_team": "C", "away_team": "D", "p": 50}]}}
    miss = site.biggest_miss(rec, matches)
    assert miss["home_team"] == "A" and miss["kind"] == "Unlucky"
    rec["last"]["rows"].pop(0)
    assert site.biggest_miss(rec, matches)["kind"] == "Our mistake"
    assert site.biggest_miss({"last": {"rows": []}}, matches) is None


def test_build_writes_the_new_pages_and_feeds(tmp_path):
    out = site.build(make_data(tmp_path), tmp_path / "site")
    for page in ("table/index.html", "fpl/index.html", "pundits/index.html", "club/index.html",
                 "club/strong/index.html", "sitemap.xml", "feed.xml", "robots.txt", "social/newsletter.md"):
        assert (out / page).exists(), page
    week = (out / "matchweek" / "9" / "index.html").read_text()
    assert 'property="og:image" content="https://' in week and "social/png/round.png" in week
    assert 'href="../../club/strong/index.html"' in week
    assert "goatcounter" not in week and "get the newsletter" not in week
    assert "matchweek/9/strong-v-weak.html" in (out / "sitemap.xml").read_text()
    assert "<entry><title>Matchweek 9" in (out / "feed.xml").read_text()
    letter = (out / "social" / "newsletter.md").read_text()
    assert letter.startswith("# Matchweek 9, called") and "not betting advice" in letter


def test_analytics_and_newsletter_switch_on_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GOATCOUNTER", "inkwell")
    monkeypatch.setenv("NEWSLETTER_URL", "https://example.com/subscribe")
    out = site.build(make_data(tmp_path), tmp_path / "site")
    home = (out / "index.html").read_text()
    assert 'data-goatcounter="https://inkwell.goatcounter.com/count"' in home
    assert 'href="https://example.com/subscribe"' in home


def test_extra_captions_fit_x():
    sim = {"runs": 10000, "teams": [{"team": "A", "title": 0.6, "relegation": 0.0},
                                    {"team": "B", "title": 0.1, "relegation": 0.7}]}
    fpl_rows = [{"team": "A", "cs": 45, "xg_for": 2.1}]
    miss = {"match": "A v B", "called": "A to win", "p": 60, "result": "0–1", "kind": "Unlucky", "text": "x " * 200}
    cap = social.extra_captions({"sim": sim, "fpl": {"clean_sheets": fpl_rows, "scoring": fpl_rows},
                                 "miss": miss, "matchweek": 9, "sim_moves": []})
    for k in ("projection_x", "fpl_x", "miss_x"):
        assert social.x_length(cap[k]) <= social.X_LIMIT
    assert "B go down in 70%" in cap["projection_x"]
