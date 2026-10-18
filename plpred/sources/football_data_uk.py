"""Recent Premier League results, match stats and odds from football-data.co.uk.

One CSV per season at https://www.football-data.co.uk/mmz4281/<code>/E0.csv.
Past seasons never change, so they are downloaded once and kept; only the current
season is re-downloaded.
"""
import io
import logging
from pathlib import Path

import pandas as pd

from plpred import seasons, teams
from plpred.http import session

log = logging.getLogger(__name__)

URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

STAT_COLUMNS = {
    "HTHG": "home_ht_goals", "HTAG": "away_ht_goals",
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_shots_on_target", "AST": "away_shots_on_target",
    "HC": "home_corners", "AC": "away_corners",
    "HF": "home_fouls", "AF": "away_fouls",
    "HY": "home_yellows", "AY": "away_yellows",
    "HR": "home_reds", "AR": "away_reds",
    "Referee": "referee",
}
# Market-average 1X2 odds changed column names over the years; take the first one present.
ODDS_PREFIXES = ["Avg", "BbAv", "B365", "WH"]


def raw_path(data_dir: Path, season: int) -> Path:
    return data_dir / "raw" / "football-data-uk" / f"E0_{seasons.fd_code(season)}.csv"


def download(data_dir: Path, wanted: list[int] | None = None) -> list[Path]:
    wanted = wanted or seasons.window()
    last = wanted[-1]
    s = session()
    paths = []
    for season in wanted:
        path = raw_path(data_dir, season)
        if path.exists() and season < last:
            paths.append(path)
            continue
        resp = s.get(URL.format(code=seasons.fd_code(season)), timeout=30)
        if resp.status_code == 404:
            log.warning("No football-data.co.uk file for %s yet", seasons.label(season))
            continue
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        paths.append(path)
        log.info("Downloaded %s", seasons.label(season))
    return paths


def parse(raw: bytes, season: int) -> pd.DataFrame:
    """Turn one season's CSV into tidy rows with canonical team names."""
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = pd.DataFrame({
        "season": season,
        "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed"),
        "home_team": df["HomeTeam"].map(teams.canonical),
        "away_team": df["AwayTeam"].map(teams.canonical),
        "home_goals": df["FTHG"].astype(int),
        "away_goals": df["FTAG"].astype(int),
        "result": df["FTR"],
    })
    for src, dst in STAT_COLUMNS.items():
        out[dst] = df[src] if src in df else pd.NA
    for prefix in ODDS_PREFIXES:
        cols = [f"{prefix}{x}" for x in "HDA"]
        if all(c in df for c in cols) and df[cols].notna().any().all():
            out["odds_home"], out["odds_draw"], out["odds_away"] = (df[c] for c in cols)
            out["odds_source"] = prefix
            break
    else:
        out["odds_home"] = out["odds_draw"] = out["odds_away"] = pd.NA
        out["odds_source"] = pd.NA
    return out.reset_index(drop=True)


def build_matches(data_dir: Path, wanted: list[int] | None = None) -> pd.DataFrame:
    frames = []
    for season in wanted or seasons.window():
        path = raw_path(data_dir, season)
        if path.exists():
            frames.append(parse(path.read_bytes(), season))
    matches = pd.concat(frames, ignore_index=True).sort_values(["date", "home_team"])
    return matches.reset_index(drop=True)
