"""La Liga: its fixture list, its section of the site and its slimmer posting kit."""
import json
import shutil

import pandas as pd

from plpred import backtest, health, leagues, site, social
from plpred.sources import fixture_list, football_data_uk
from tests.test_site import make_data


def test_featured_games_are_the_big_clubs_and_top_four_clashes():
    top = {"Barcelona", "Real Madrid", "Villarreal", "Athletic Club"}
    la = leagues.LA_LIGA
    assert leagues.featured(la, "Real Betis", "Osasuna", top)          # Gary added Betis
    assert leagues.featured(la, "Getafe", "Atletico Madrid", top)
    assert leagues.featured(la, "Villarreal", "Athletic Club", top)    # a top-four clash
    assert not leagues.featured(la, "Villarreal", "Getafe", top)
    assert leagues.featured(leagues.PREMIER_LEAGUE, "Hull", "Ipswich", set())


def test_la_liga_names_join_across_sources():
    from plpred import teams
    for alias in ("Ath Madrid", "Club Atlético de Madrid"):
        assert teams.canonical(alias) == "Atletico Madrid"
    assert teams.canonical("Betis") == teams.canonical("Real Betis Balompié") == "Real Betis"
    assert teams.canonical("Sociedad") == "Real Sociedad"


def test_la_liga_odds_come_from_the_sp1_rows():
    csv = ("Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA\n"
           "E0,19/10/2026,20:00,Tottenham,Coventry,1.6,4.2,5.5\n"
           "SP1,18/10/2026,20:00,Ath Madrid,Betis,1.7,3.8,4.9\n").encode()
    odds = football_data_uk.parse_upcoming(csv, "SP1")
    assert odds[["home_team", "away_team"]].values.tolist() == [["Atletico Madrid", "Real Betis"]]


OPENFOOTBALL = {"matches": [
    {"round": "Matchday 1", "date": "2026-08-15", "time": "19:30", "team1": "Real Madrid CF", "team2": "Getafe CF",
     "score": {"ht": [1, 0], "ft": [2, 0]}},
    {"round": "Matchday 1", "date": "2026-08-16", "time": "21:00", "team1": "FC Barcelona", "team2": "Real Betis Balompié",
     "score": [1, 1]},
    {"round": "Matchday 2", "date": "2026-08-22", "time": "19:30", "team1": "Getafe CF", "team2": "FC Barcelona"},
    {"round": "Matchday 2", "date": "2026-08-23", "team1": "Real Betis Balompié", "team2": "Real Madrid CF"},
    {"round": "Matchday 6", "date": "2026-09-19", "time": "16:15", "team1": "Getafe CF", "team2": "Real Betis Balompié"},
]}


def understat_game(i, home, away, when, goals=None):
    return {"id": str(i), "isResult": goals is not None, "datetime": when, "h": {"title": home}, "a": {"title": away},
            "goals": {"h": str(goals[0]), "a": str(goals[1])} if goals else {"h": None, "a": None}}


def test_fixture_list_takes_kickoffs_from_understat_and_rounds_from_openfootball():
    us = fixture_list.understat_frame([
        understat_game(1, "Real Madrid", "Getafe", "2026-08-15 17:30:00", (2, 0)),
        understat_game(2, "Barcelona", "Real Betis", "2026-08-16 19:00:00", (1, 1)),
        understat_game(3, "Getafe", "Barcelona", "2026-08-22 17:30:00"),
        # Moved a month from its matchday: it joins the round being played then.
        understat_game(4, "Real Betis", "Real Madrid", "2026-09-20 19:00:00"),
    ])
    of = fixture_list.openfootball_frame(OPENFOOTBALL)
    assert of.finished.tolist() == [True, True, False, False, False]  # both score formats
    games = fixture_list.combine(us, of)
    assert {1, 2, 3, 4} <= set(games.fixture_id)
    assert games[["home_team", "gameweek"]].values.tolist() == [
        ["Real Madrid", 1], ["Barcelona", 1], ["Getafe", 2], ["Getafe", 6], ["Real Betis", 6]]
    now = pd.Timestamp("2026-08-20", tz="UTC")
    r = fixture_list.round_fixtures(games, now)
    assert r["round"].unique().tolist() == ["Regular Season - 2"] and r.status.tolist() == ["NS"]
    # Without Understat the list still works, from openfootball's local (Madrid) times.
    alone = fixture_list.combine(None, of)
    assert alone.kickoff.iloc[0] == pd.Timestamp("2026-08-15 17:30", tz="UTC")
    # Games not finished two days after kick-off count as postponed and stop holding a round open.
    assert fixture_list.round_fixtures(games, pd.Timestamp("2026-09-25", tz="UTC")).empty


def test_back_to_back_rounds_are_split_when_a_club_plays_twice():
    dates = pd.Series(pd.to_datetime(["2026-09-12", "2026-09-13", "2026-09-14", "2026-09-15", "2026-09-16"]))
    home = pd.Series(["A", "C", "E", "B", "D"])
    away = pd.Series(["B", "D", "F", "C", "E"])
    assert backtest.gameweeks(dates).nunique() == 1
    assert backtest.gameweeks(dates, home, away).tolist() == [1, 1, 1, 2, 2]


def laliga_data(tmp_path):
    """The Premier League test data, plus the same games as a La Liga folder."""
    data = make_data(tmp_path)
    ll = data / "laliga"
    ll.mkdir()
    for name in ("matches.csv", "predictions"):
        src = data / name
        (shutil.copytree if src.is_dir() else shutil.copy)(src, ll / name)
    leagues.mark(ll, leagues.LA_LIGA)
    return data, ll


def test_la_liga_gets_its_own_section_and_kit(tmp_path, monkeypatch):
    data, ll = laliga_data(tmp_path)
    # Only Strong v Weak is featured, so only it gets a carousel and a Reel.
    monkeypatch.setattr(leagues, "featured", lambda league, h, a, top: league.big_clubs is None or h == "Strong")
    out = tmp_path / "site"
    site.main(["--data", str(data), "--out", str(out)])
    home = (out / "laliga" / "index.html").read_text()
    assert "La Liga 2026/27" in home and "Data-driven La Liga predictions" in home
    assert 'href="../2026-27/matchweek/9/' not in home and 'href="../laliga/2026-27/matchweek/9/strong-v-weak.html"' in home
    assert "fpl/" not in home.split("</header>")[0] and 'href="../">Premier League</a>' in home
    assert '<img class="badge"' not in home  # no crests for La Liga
    assert (out / "laliga" / "2026-27" / "matchweek" / "9" / "strong-v-weak.html").exists()
    pl = (out / "index.html").read_text()
    assert 'href="./laliga/">La Liga</a>' in pl
    assert "laliga/2026-27/matchweek/9/" in (out / "sitemap.xml").read_text()
    kit = out / "social" / "laliga"
    names = [c["name"] for c in json.loads((kit / "cards.json").read_text())]
    assert "round" in names and "call-strong-v-weak" in names
    videos = [v["name"] for v in json.loads((kit / "videos.json").read_text())]
    assert "video-strong-v-weak" in videos and "video-average-v-good" not in videos
    posts = json.loads((kit / "posts.json").read_text())
    assert posts["league"] == "La Liga" and not posts["main"]
    assert not (kit / "superbru.txt").exists() and not (kit / "newsletter.md").exists()
    first = posts["posts"][0]
    assert first["x"][0].startswith("La Liga Matchweek 9") and "#LaLiga" in first["instagram"]
    assert 'href="../../../style.css"' in (kit / "cards" / "round.html").read_text()
    # The Premier League kit is untouched: every game still gets its carousel.
    pl_names = [c["name"] for c in json.loads((out / "social" / "cards.json").read_text())]
    assert {"call-strong-v-weak", "call-average-v-good"} <= set(pl_names)


def test_la_liga_kit_emails_say_which_league(tmp_path):
    data, ll = laliga_data(tmp_path)
    out = tmp_path / "site"
    site.main(["--data", str(data), "--out", str(out)])
    kit = out / "social" / "laliga"
    (kit / "png").mkdir()
    for c in json.loads((kit / "cards.json").read_text()):
        (kit / "png" / f"{c['name']}.png").write_bytes(b"png")
    social.ready_to_post(kit)
    subjects = [m["Subject"] for m in social.kit_emails(kit, "a@b.c", "a@b.c")]
    assert subjects and all(s.startswith("La Liga · Matchweek 9") for s in subjects)


def test_health_reads_the_la_liga_fixture_list(tmp_path):
    data, ll = laliga_data(tmp_path)
    (ll / "upcoming").mkdir()
    pd.DataFrame({"fixture_id": [1], "gameweek": [9], "kickoff": ["2026-10-24T14:00:00+00:00"],
                  "home_team": ["Strong"], "away_team": ["Weak"], "finished": [False]}
                 ).to_csv(ll / "upcoming" / "season_fixtures.csv", index=False)
    found = health.problems(ll, pd.Timestamp("2026-10-27", tz="UTC"))
    assert any(p.startswith("La Liga: Understat still hasn't marked 1 games") for p in found)
    # Long after, it's taken as postponed rather than stuck.
    assert not any("Understat" in p for p in health.problems(ll, pd.Timestamp("2026-11-20", tz="UTC")))
