import json
from datetime import date

from plpred import seasons, teams
from plpred.sources import api_football, football_data_uk, fpl, understat

NEW_CSV = b"""Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,B365H,B365D,B365A,AvgH,AvgD,AvgA
E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,H,1,0,H,A Taylor,19,10,10,3,7,10,6,7,1,2,0,0,1.3,6,9.5,1.31,5.8,9.2
E0,16/08/2025,12:30,Aston Villa,Newcastle,0,0,D,0,0,D,S Hooper,3,16,1,3,12,11,3,9,2,2,1,0,2.2,3.5,3.2,2.18,3.45,3.25
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
"""
OLD_CSV = b"""Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,BbAvH,BbAvD,BbAvA
E0,14/08/10,Aston Villa,West Ham,3,0,H,1.9,3.4,4.2
"""


def test_season_helpers():
    assert seasons.current_season(date(2026, 10, 18)) == 2026
    assert seasons.current_season(date(2027, 3, 1)) == 2026
    assert seasons.fd_code(2026) == "2627"
    assert seasons.fd_code(1999) == "9900"
    assert seasons.label(2026) == "2026/27"


def test_team_aliases():
    assert teams.canonical("Manchester United") == "Man United"
    assert teams.canonical("Spurs") == "Tottenham"
    assert teams.canonical("Arsenal") == "Arsenal"


def test_parse_recent_season_uses_market_average_odds():
    df = football_data_uk.parse(NEW_CSV, 2025)
    assert len(df) == 2  # blank trailing row dropped
    first = df.iloc[0]
    assert (first.home_team, first.away_team, first.home_goals, first.away_goals) == ("Liverpool", "Bournemouth", 4, 2)
    assert first.home_shots_on_target == 10
    assert first.odds_source == "Avg" and first.odds_home == 1.31
    assert str(first.date.date()) == "2025-08-15"


def test_parse_old_season_falls_back_to_betbrain_odds():
    df = football_data_uk.parse(OLD_CSV, 2010)
    assert df.iloc[0].odds_source == "BbAv"
    assert str(df.iloc[0].date.date()) == "2010-08-14"
    assert df.home_shots.isna().all()


def test_build_matches_reads_seasons_from_file_names(tmp_path):
    folder = tmp_path / "raw" / "football-data-uk"
    folder.mkdir(parents=True)
    (folder / "E0_1011.csv").write_bytes(OLD_CSV)
    (folder / "E0_2526.csv").write_bytes(NEW_CSV)
    matches = football_data_uk.build_matches(tmp_path)
    assert sorted(matches.season.unique()) == [2010, 2025]


UNDERSTAT_MATCHES = [
    {"id": "1", "isResult": True, "datetime": "2025-08-15 19:00:00",
     "h": {"title": "Liverpool"}, "a": {"title": "Bournemouth"}, "xG": {"h": "2.71", "a": "1.48"}},
    {"id": "2", "isResult": True, "datetime": "2025-08-16 11:30:00",
     "h": {"title": "Manchester United"}, "a": {"title": "Wolverhampton Wanderers"}, "xG": {"h": "1.9", "a": "0.6"}},
    {"id": "3", "isResult": False, "datetime": "2026-05-24 15:00:00",
     "h": {"title": "Arsenal"}, "a": {"title": "Chelsea"}, "xG": {"h": None, "a": None}},
]


def test_understat_embedded_page_and_frame():
    escaped = json.dumps(UNDERSTAT_MATCHES).replace('"', "\\x22")
    html = f"<script>var datesData = JSON.parse('{escaped}');</script>"
    raw = understat.parse_page(html)
    df = understat.to_frame(raw, 2025)
    assert len(df) == 2  # unplayed match skipped
    assert df.iloc[1].home_team == "Man United" and df.iloc[1].away_team == "Wolves"
    assert df.iloc[0].home_xg == 2.71


FPL_RAW = {
    "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Man Utd"}],
    "elements": [
        {"team": 1, "web_name": "Saka", "element_type": 3, "status": "i", "chance_of_playing_next_round": 0,
         "news": "Hamstring injury", "minutes": 700, "expected_goal_involvements": "4.1"},
        {"team": 1, "web_name": "Raya", "element_type": 1, "status": "a", "chance_of_playing_next_round": None,
         "news": "", "minutes": 720, "expected_goal_involvements": "0.0"},
        {"team": 1, "web_name": "Fringe", "element_type": 4, "status": "i", "chance_of_playing_next_round": 0,
         "news": "Knee", "minutes": 30, "expected_goal_involvements": "0.1"},
        {"team": 2, "web_name": "Fernandes", "element_type": 3, "status": "d", "chance_of_playing_next_round": 75,
         "news": "Knock - 75% chance of playing", "minutes": 720, "expected_goal_involvements": "3.0"},
    ],
}


def test_fpl_key_absences_only_regulars():
    players = fpl.availability(FPL_RAW)
    assert set(players.team) == {"Arsenal", "Man United"}
    absent = fpl.key_absences(players)
    assert list(absent.player) == ["Saka", "Fernandes"]


def test_api_football_frames():
    fixtures = api_football.fixtures_frame([{
        "fixture": {"id": 9, "date": "2026-10-24T14:00:00+00:00", "status": {"short": "NS"}, "venue": {"name": "Anfield"}},
        "league": {"round": "Regular Season - 9"},
        "teams": {"home": {"name": "Liverpool"}, "away": {"name": "Manchester City"}},
        "goals": {"home": None, "away": None},
    }])
    assert fixtures.iloc[0].away_team == "Man City"
    odds = api_football.odds_frame([{"fixture": {"id": 9}, "bookmakers": [
        {"name": "A", "bets": [{"name": "Match Winner", "values": [
            {"value": "Home", "odd": "2.0"}, {"value": "Draw", "odd": "3.5"}, {"value": "Away", "odd": "3.6"}]}]},
        {"name": "B", "bets": [{"name": "Match Winner", "values": [
            {"value": "Home", "odd": "2.2"}, {"value": "Draw", "odd": "3.5"}, {"value": "Away", "odd": "3.4"}]}]},
    ]}])
    assert round(odds.iloc[0].odds_home, 2) == 2.1
    assert api_football.odds_frame([]).empty
