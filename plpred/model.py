"""Team-strength model: Dixon-Coles style Poisson ratings with time decay and xG.

Each team gets an attack and a defence rating. Expected goals for a match are

    home: exp(home_adv + attack[home] + defence[away])
    away: exp(attack[away] + defence[home])

Ratings are fitted to recent matches, with older matches counting for less
(an exponential decay with a chosen half-life), so the current season matters most.
The fitting target blends actual goals with xG, because xG says more about how well
a team is really playing than a handful of results. A Dixon-Coles correction then
adjusts the chances of the low-scoring results (0-0, 1-0, 0-1, 1-1) that a plain
Poisson model gets wrong.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


@dataclass(frozen=True)
class Settings:
    half_life_days: float = 180.0   # a match this old counts half as much as one today
    xg_weight: float = 0.5          # 0 = goals only, 1 = xG only
    ridge: float = 1.0              # pulls teams with few matches towards average
    rho: float = -0.13              # Dixon-Coles low-score correction


@dataclass
class Ratings:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    settings: Settings
    as_of: pd.Timestamp
    matches_used: int = 0
    _index: dict = field(init=False, repr=False)

    def __post_init__(self):
        self._index = {t: i for i, t in enumerate(self.teams)}

    def knows(self, team: str) -> bool:
        return team in self._index

    def rating(self, team: str) -> tuple[float, float]:
        """(attack, defence) for a team. A team we have no matches for yet, usually a
        newly promoted one, is rated like the average of the three weakest teams."""
        if team in self._index:
            i = self._index[team]
            return float(self.attack[i]), float(self.defence[i])
        weakest = np.argsort(self.attack - self.defence)[:3]
        return float(self.attack[weakest].mean()), float(self.defence[weakest].mean())

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        h_att, h_def = self.rating(home)
        a_att, a_def = self.rating(away)
        lam = np.exp(self.home_adv + h_att + a_def)
        mu = np.exp(a_att + h_def)
        return float(lam), float(mu)

    def table(self) -> pd.DataFrame:
        """Ratings as a table, best overall first. High attack and low defence are good."""
        df = pd.DataFrame({"team": self.teams, "attack": self.attack, "defence": self.defence})
        df["overall"] = df.attack - df.defence
        return df.sort_values("overall", ascending=False).reset_index(drop=True)


def weights(dates: pd.Series, as_of: pd.Timestamp, half_life_days: float) -> np.ndarray:
    age = (as_of - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    return 0.5 ** (age / half_life_days)


def fit(matches: pd.DataFrame, as_of: pd.Timestamp | None = None,
        settings: Settings | None = None) -> Ratings:
    """Fit ratings on matches played before `as_of` (default: the day after the last match)."""
    settings = settings or Settings()
    dates = pd.to_datetime(matches.date)
    as_of = pd.Timestamp(as_of) if as_of is not None else dates.max() + pd.Timedelta(days=1)
    m = matches[dates < as_of]
    if m.empty:
        raise ValueError("No matches to fit on")

    teams = sorted(set(m.home_team) | set(m.away_team))
    idx = {t: i for i, t in enumerate(teams)}
    h = m.home_team.map(idx).to_numpy()
    a = m.away_team.map(idx).to_numpy()
    y_home = _target(m, "home", settings.xg_weight)
    y_away = _target(m, "away", settings.xg_weight)
    w = weights(m.date, as_of, settings.half_life_days)
    n = len(teams)

    def loss_and_grad(p):
        att, dfn, home = p[:n], p[n:2 * n], p[-1]
        lam = np.exp(home + att[h] + dfn[a])
        mu = np.exp(att[a] + dfn[h])
        # Poisson log-likelihood (also valid for non-integer xG targets), weighted by recency.
        ll = (w * (y_home * np.log(lam) - lam + y_away * np.log(mu) - mu)).sum()
        r_home, r_away = w * (y_home - lam), w * (y_away - mu)
        g_att = np.bincount(h, r_home, n) + np.bincount(a, r_away, n)
        g_def = np.bincount(a, r_home, n) + np.bincount(h, r_away, n)
        grad = -np.r_[g_att, g_def, r_home.sum()]
        grad[:2 * n] += 2 * settings.ridge * p[:2 * n]
        return -ll + settings.ridge * (att @ att + dfn @ dfn), grad

    # Attack and defence are only defined up to a shared constant, so fix mean attack at 0.
    mean_attack = {"type": "eq", "fun": lambda p: p[:n].mean(),
                   "jac": lambda p: np.r_[np.full(n, 1 / n), np.zeros(n + 1)]}
    res = minimize(loss_and_grad, np.r_[np.zeros(2 * n), 0.25], jac=True,
                   constraints=[mean_attack], method="SLSQP", options={"maxiter": 500})
    if not res.success:
        raise RuntimeError(f"Rating fit failed: {res.message}")
    p = res.x
    return Ratings(teams, p[:n], p[n:2 * n], float(p[-1]), settings, as_of, len(m))


def _target(m: pd.DataFrame, side: str, xg_weight: float) -> np.ndarray:
    goals = m[f"{side}_goals"].to_numpy(dtype=float)
    col = f"{side}_xg"
    if xg_weight == 0 or col not in m:
        return goals
    xg = m[col].to_numpy(dtype=float)
    return np.where(np.isnan(xg), goals, (1 - xg_weight) * goals + xg_weight * xg)


def score_matrix(lam: float, mu: float, rho: float) -> np.ndarray:
    """P(home scores i, away scores j) for i, j in 0..MAX_GOALS, with the Dixon-Coles correction."""
    goals = np.arange(MAX_GOALS + 1)
    probs = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    probs[0, 0] *= 1 - lam * mu * rho
    probs[0, 1] *= 1 + lam * rho
    probs[1, 0] *= 1 + mu * rho
    probs[1, 1] *= 1 - rho
    return probs / probs.sum()


@dataclass
class Prediction:
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    p_home: float
    p_draw: float
    p_away: float
    call: str           # "H", "D" or "A"
    score: tuple[int, int]
    score_prob: float


def predict(ratings: Ratings, home: str, away: str) -> Prediction:
    lam, mu = ratings.expected_goals(home, away)
    probs = score_matrix(lam, mu, ratings.settings.rho)
    p_home = float(np.tril(probs, -1).sum())
    p_away = float(np.triu(probs, 1).sum())
    p_draw = float(np.trace(probs))
    call = "HDA"[int(np.argmax([p_home, p_draw, p_away]))]
    # The predicted score is the most likely scoreline that agrees with the call.
    i, j = np.indices(probs.shape)
    agrees = {"H": i > j, "D": i == j, "A": i < j}[call]
    best = np.unravel_index(np.argmax(np.where(agrees, probs, -1)), probs.shape)
    return Prediction(home, away, lam, mu, p_home, p_draw, p_away, call,
                      (int(best[0]), int(best[1])), float(probs[best]))
