import json
import pandas as pd
import re
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
    assert "Matchweek 9" in home and 'href="./2026-27/matchweek/9/strong-v-weak.html"' in home  # the front page is this week
    assert 'href="./about/"' in home
    about = (out / "about" / "index.html").read_text()
    assert "Where the data comes from" in about and "Understat" in about
    assert 'href="../"' in about
    assert "Inkwell <span class=\"amp\">&amp;</span> Algorithm" in home and (out / "favicon.svg").exists()
    week = (out / "2026-27" / "matchweek" / "9" / "index.html").read_text()
    assert "Matchweek 9" in week
    assert "Replay" not in week.split("<body>")[1].split("</header>")[0]
    assert (out / "badges" / "strong.png").exists()
    assert 'src="../../../badges/strong.png"' in week and 'href="../../../2026-27/matchweek/9/strong-v-weak.html"' in week
    page = (out / "2026-27" / "matchweek" / "9" / "strong-v-weak.html").read_text()
    assert "How we got to" in page and "../../../style.css" in page
    # The first addresses had no season; they redirect, keeping any #section.
    old = (out / "matchweek" / "9" / "strong-v-weak.html").read_text()
    new_url = f"{site.social.SITE_URL}2026-27/matchweek/9/strong-v-weak.html"
    assert f'<link rel="canonical" href="{new_url}">' in old and f'url={new_url}"' in old and "location.hash" in old
    assert "matchweek/9/" not in (out / "sitemap.xml").read_text().replace("2026-27/matchweek/9/", "")


def test_each_matchweek_keeps_its_own_page(tmp_path):
    data = make_data(tmp_path)
    first = json.loads((data / "predictions" / "2026-10-24.json").read_text())
    first.update(round="Regular Season - 8", as_of="2026-12-24")
    (data / "predictions" / "2026-10-17.json").write_text(json.dumps(first))
    out = site.build(data, tmp_path / "site")
    assert (out / "2026-27" / "matchweek" / "8" / "strong-v-weak.html").exists()
    home = (out / "about" / "index.html").read_text()
    assert home.index("Matchweek 9</span>") < home.index("Matchweek 8</span>")  # newest first
    week8 = (out / "2026-27" / "matchweek" / "8" / "index.html").read_text()
    assert 'href="../../../2026-27/matchweek/9/"' in week8
    # The newest week is the front page, so it points search engines there; older weeks stand alone.
    week9 = (out / "2026-27" / "matchweek" / "9" / "index.html").read_text()
    assert f'<link rel="canonical" href="{site.social.SITE_URL}">' in week9
    assert f'<link rel="canonical" href="{site.social.SITE_URL}2026-27/matchweek/8/">' in week8
    sitemap = (out / "sitemap.xml").read_text()
    assert f"<loc>{site.social.SITE_URL}2026-27/matchweek/8/</loc>" in sitemap
    assert "matchweek/9/</loc>" not in sitemap and "index.html" not in sitemap and "404" not in sitemap


def test_pages_carry_search_titles_and_structured_data(tmp_path):
    out = site.build(make_data(tmp_path), tmp_path / "site")
    page = (out / "2026-27" / "matchweek" / "9" / "strong-v-weak.html").read_text()
    assert "<title>Strong v Weak prediction: " in page and "| Inkwell &amp; Algorithm</title>" in page
    blocks = [json.loads(b) for b in re.findall(r'<script type="application/ld\+json">(.*?)</script>', page)]
    kinds = {b["@type"] for b in blocks}
    assert kinds == {"BreadcrumbList", "SportsEvent"}
    event = next(b for b in blocks if b["@type"] == "SportsEvent")
    assert event["homeTeam"]["name"] == "Strong" and event["awayTeam"]["name"] == "Weak"
    home = (out / "index.html").read_text()
    assert '"@type": "Organization"' in home and "<title>Premier League predictions: Matchweek 9" in home
    missing = (out / "404.html").read_text()
    assert '<meta name="robots" content="noindex">' in missing and 'rel="canonical"' not in missing
    assert f'href="{site.social.SITE_URL}about/#archive"' in missing


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
    week = (out / "2026-27" / "matchweek" / "9" / "index.html").read_text()
    assert "after 1 of its 2 games had kicked off" in week and "Final 2–0" in week
    assert "After kick-off" not in (out / "2026-27" / "matchweek" / "9" / "average-v-good.html").read_text()
    assert "test, not a prediction" in (out / "2026-27" / "matchweek" / "9" / "strong-v-weak.html").read_text()
    kit = (out / "social" / "index.html").read_text()
    assert "would have called" in kit and "not part of our record" in kit


def test_newest_week_table_and_club_pages_include_results_after_the_calls(tmp_path):
    data = make_data(tmp_path)
    m = pd.read_csv(data / "matches.csv", parse_dates=["date"])
    preds = json.loads((data / "predictions" / "2026-10-24.json").read_text())
    preds["as_of"] = str((m.date.max() - pd.Timedelta(days=1)).date())  # the last round was played after the calls
    (data / "predictions" / "2026-10-24.json").write_text(json.dumps(preds))
    out = site.build(data, tmp_path / "site")
    week = (out / "2026-27" / "matchweek" / "9" / "index.html").read_text()
    assert f"Results up to {m.date.max().strftime('%d %B')}" in week
    last = m[m.date == m.date.max()].iloc[0]
    club = (out / "club" / site.club_slug(last.home_team) / "index.html").read_text()
    assert m.date.max().strftime("%d %b") in club
