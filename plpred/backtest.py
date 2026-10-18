"""Back-test the model: predict past matches using only what was known beforehand.

    python -m plpred.backtest            # tune on 2025/26, check on 2026/27 so far

For each gameweek, ratings are refitted on every match played before the round's first
kickoff and used to predict all of that round's games, just as the live site does. Results are compared with two yardsticks: the
bookmakers' favourite (from their average odds) and simply backing the home team.
"""
import argparse
import dataclasses
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from plpred import model, seasons

log = logging.getLogger("plpred.backtest")
DATA = Path(__file__).resolve().parent.parent / "data"

GRID = {"half_life_days": [90, 180, 365], "xg_weight": [0.0, 0.5, 1.0],
        "ridge": [0.05, 1.0], "rho": [-0.08, -0.13]}


def gameweeks(dates: pd.Series) -> pd.Series:
    """Number the rounds: a new round starts after a gap of two or more days, except on a
    Monday, which closes the weekend round rather than starting a new one."""
    days = pd.Series(pd.to_datetime(dates).dt.normalize().unique()).sort_values()
    starts = (days.diff().dt.days.fillna(99) >= 2) & (days.dt.dayofweek != 0)
    starts.iloc[0] = True
    return pd.to_datetime(dates).dt.normalize().map(dict(zip(days, starts.cumsum())))


def run(matches: pd.DataFrame, test_seasons: list[int], settings: model.Settings) -> pd.DataFrame:
    """One row per predicted match, with the model's and the bookmakers' probabilities."""
    matches = matches.assign(date=pd.to_datetime(matches.date))
    rows = []
    for season in test_seasons:
        test = matches[matches.season == season]
        for gw, games in test.groupby(gameweeks(test.date)):
            rows += _predict_round(matches, games, int(gw), settings)
    return pd.DataFrame(rows)


def _predict_round(matches: pd.DataFrame, games: pd.DataFrame, gw: int, settings: model.Settings) -> list[dict]:
    ratings = model.fit(matches, as_of=games.date.min().normalize(), settings=settings)
    rows = []
    for m in games.itertuples():
        p = model.predict(ratings, m.home_team, m.away_team)
        rows.append({
            "season": m.season, "gameweek": gw, "date": m.date, "home_team": m.home_team,
            "away_team": m.away_team, "result": m.result, "home_goals": m.home_goals,
            "away_goals": m.away_goals, "p_home": p.p_home, "p_draw": p.p_draw, "p_away": p.p_away,
            "call": p.call, "score": f"{p.score[0]}-{p.score[1]}", **_market(m),
        })
    return rows


def _market(m) -> dict:
    odds = np.array([m.odds_home, m.odds_draw, m.odds_away], dtype=float)
    if np.isnan(odds).any():
        return {"mkt_home": np.nan, "mkt_draw": np.nan, "mkt_away": np.nan}
    implied = (1 / odds) / (1 / odds).sum()  # strip the bookmakers' margin
    return dict(zip(["mkt_home", "mkt_draw", "mkt_away"], implied))


def score(preds: pd.DataFrame, prefix: str = "p") -> dict:
    """Accuracy, log loss and ranked probability score (lower is better for the last two)."""
    probs = preds[[f"{prefix}_home", f"{prefix}_draw", f"{prefix}_away"]].to_numpy()
    outcome = preds.result.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    actual = np.eye(3)[outcome]
    picked = probs.argmax(axis=1)
    rps = (((probs.cumsum(1) - actual.cumsum(1))[:, :2]) ** 2).sum(1) / 2
    return {
        "matches": len(preds),
        "accuracy": float((picked == outcome).mean()),
        "log_loss": float(-np.log(probs[np.arange(len(probs)), outcome]).mean()),
        "rps": float(rps.mean()),
    }


def report(preds: pd.DataFrame) -> pd.DataFrame:
    rows = {"Model": score(preds), "Bookmakers": score(preds.dropna(subset=["mkt_home"]), "mkt")}
    rows["Always home"] = {"matches": len(preds), "accuracy": float((preds.result == "H").mean()),
                           "log_loss": np.nan, "rps": np.nan}
    exact = (preds.score == preds.home_goals.astype(str) + "-" + preds.away_goals.astype(str)).mean()
    rows["Model"]["exact_score"] = float(exact)
    return pd.DataFrame(rows).T


def tune(matches: pd.DataFrame, season: int) -> tuple[model.Settings, pd.DataFrame]:
    results = []
    for values in itertools.product(*GRID.values()):
        settings = model.Settings(**dict(zip(GRID, values)))
        s = score(run(matches, [season], settings))
        results.append({**dataclasses.asdict(settings), **s})
        log.info("%s: accuracy %.3f, RPS %.4f", settings, s["accuracy"], s["rps"])
    table = pd.DataFrame(results)
    # Lowest RPS wins. Rounding to call near-ties used to hand the win to the shortest
    # memory even when it scored worse, so only exact ties fall back to preferring the
    # shorter memory and the gentler xG blend.
    table = table.sort_values(["rps", "half_life_days", "xg_weight"])
    best = table.iloc[0]
    return model.Settings(**{k: float(best[k]) for k in GRID}), table


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args(argv)

    matches = pd.read_csv(args.data / "matches.csv", parse_dates=["date"])
    current = seasons.current_season()
    tune_season, check_season = current - 1, current
    settings, grid = tune(matches, tune_season)
    log.info("Best settings on %s: %s", seasons.label(tune_season), settings)

    out = args.data / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "tuning.csv", index=False)
    (out / "settings.json").write_text(json.dumps(dataclasses.asdict(settings), indent=2) + "\n")
    for season in (tune_season, check_season):
        preds = run(matches, [season], settings)
        preds.to_csv(out / f"predictions_{season}.csv", index=False)
        summary = report(preds)
        summary.to_csv(out / f"summary_{season}.csv")
        print(f"\n{seasons.label(season)}" + (" (used for tuning)" if season == tune_season else " (unseen)"))
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
