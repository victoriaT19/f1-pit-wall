"""Plackett-Luce (rank-ordered logit) model of the finishing order.

A race result is a ranking, not a set of independent positions. The Plackett-Luce model
gives every car a strength u and builds the order one place at a time: the winner is drawn
with probability exp(u_i) / sum_j exp(u_j), second place among the rest, and so on. This is
the model of van Kesteren & Bergkamp (2023, "Bayesian analysis of Formula One race results:
disentangling driver skill and constructor advantage"), adapted from explaining past seasons
to forecasting the next race:

    u = beta . x  +  driver effect  +  team-season effect

  * x: pre-race covariates (grid slot, qualifying gap, recent race pace), standardised;
  * driver and team-season effects with Gaussian priors (ridge, i.e. a MAP estimate of the
    Bayesian model), so a driver or team with few races is shrunk towards average;
  * older races weigh less (exponential decay with a half-life in days), because form and
    car performance change during a season.

As in the paper, non-finishers are left out of the likelihood: the order is learned among
the cars that finished, and retirements are handled separately by a per-driver DNF chance.

Probabilities of winning, a podium or a top-10 come from sampling: adding independent Gumbel
noise to u and sorting gives exactly a Plackett-Luce ranking. Cars drawn as DNF go to the back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# grid_log: the first slots matter more than the last ones (pole vs 2nd is worth more than
# 15th vs 16th), which a linear grid term cannot express.
COVARIATES = ["grid", "grid_log", "quali_gap_pct", "race_pace"]


@dataclass
class PLParams:
    half_life_days: float = 240.0   # weight of a race halves every this many days
    prior_driver: float = 1.0       # ridge strength on driver effects
    prior_team: float = 2.0         # ridge strength on team-season effects
    prior_beta: float = 0.01        # ridge strength on covariate weights
    dnf_prior_races: float = 10.0   # pseudo-races of the field's DNF rate in each driver's rate
    sharpness: float = 1.0          # multiplies u when sampling (1 = the fitted model)


def _design(df: pd.DataFrame, stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Standardised covariates; missing values get the mean (0 after standardising)."""
    df = df.assign(grid_log=np.log(df["grid"].clip(lower=1)))
    x = df[COVARIATES].astype(float).copy()
    x["quali_gap_pct"] = x["quali_gap_pct"].clip(upper=5.0)
    if stats is None:
        stats = {c: (float(x[c].mean()), float(x[c].std() or 1.0)) for c in COVARIATES}
        stats = {c: (m if np.isfinite(m) else 0.0, s if np.isfinite(s) and s > 0 else 1.0)
                 for c, (m, s) in stats.items()}
    for c in COVARIATES:
        m, s = stats[c]
        x[c] = ((x[c] - m) / s).fillna(0.0)
    return x.to_numpy(), stats


class PlackettLuce:
    def __init__(self, params: PLParams | None = None):
        self.p = params or PLParams()

    def fit(self, train: pd.DataFrame, as_of: str) -> "PlackettLuce":
        """train: one row per driver per race with COVARIATES, finished, position, date."""
        p = self.p
        fin = train[train["finished"]].copy()
        fin["team_season"] = fin["team"] + "|" + fin["year"].astype(str)
        self.drivers = {d: i for i, d in enumerate(sorted(fin["driver"].unique()))}
        self.teams = {t: i for i, t in enumerate(sorted(fin["team_season"].unique()))}
        X, self.stats = _design(fin)
        age = (pd.Timestamp(as_of) - pd.to_datetime(fin["date"])).dt.days.to_numpy()
        fin["_w"] = 0.5 ** (age / p.half_life_days)

        # Pad every race into one row of a [races, cars] matrix, finishers in finishing order.
        groups = [g.sort_values("position").index for _, g in fin.groupby(["year", "round"])]
        R, M = len(groups), max(len(g) for g in groups)
        pos = {ix: k for k, ix in enumerate(fin.index)}
        self._idx = np.full((R, M), -1)
        for r, g in enumerate(groups):
            self._idx[r, :len(g)] = [pos[i] for i in g]
        self._mask = self._idx >= 0
        self._w = np.array([fin.loc[g[0], "_w"] for g in groups])
        self._X = X
        self._d = fin["driver"].map(self.drivers).to_numpy()
        self._t = fin["team_season"].map(self.teams).to_numpy()
        k, nd, nt = X.shape[1], len(self.drivers), len(self.teams)
        res = minimize(self._nll, np.zeros(k + nd + nt), jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        self.beta, self.a, self.b = res.x[:k], res.x[k:k + nd], res.x[k + nd:]
        self.dnf = self._dnf_rates(train, as_of)
        return self

    def _nll(self, theta):
        p = self.p
        k, nd = self._X.shape[1], len(self.drivers)
        beta, a, b = theta[:k], theta[k:k + nd], theta[k + nd:]
        u_all = self._X @ beta + a[self._d] + b[self._t]
        U = np.where(self._mask, u_all[np.maximum(self._idx, 0)], -np.inf)
        shift = np.max(np.where(self._mask, U, -1e9), axis=1, keepdims=True)
        E = np.where(self._mask, np.exp(U - shift), 0.0)
        S = np.cumsum(E[:, ::-1], axis=1)[:, ::-1]           # sum over places i..end
        logS = np.log(np.where(self._mask, S, 1.0)) + shift
        ll = np.where(self._mask, U - logS, 0.0).sum(axis=1)
        nll = -(self._w * ll).sum()
        # d nll / d u_j = w * (-1 + exp(u_j) * sum_{i<=j} 1/S_i)
        inv = np.where(self._mask, 1.0 / np.where(self._mask, S, 1.0), 0.0)
        g = self._w[:, None] * (-1.0 + E * np.cumsum(inv, axis=1))
        g = np.where(self._mask, g, 0.0)
        gu = np.zeros(len(u_all))
        np.add.at(gu, self._idx[self._mask], g[self._mask])
        g_beta = self._X.T @ gu
        g_a = np.bincount(self._d, weights=gu, minlength=nd)
        g_b = np.bincount(self._t, weights=gu, minlength=len(self.teams))
        nll += 0.5 * (p.prior_beta * beta @ beta + p.prior_driver * a @ a + p.prior_team * b @ b)
        grad = np.concatenate([g_beta + p.prior_beta * beta, g_a + p.prior_driver * a,
                               g_b + p.prior_team * b])
        return nll, grad

    def _dnf_rates(self, train: pd.DataFrame, as_of: str) -> dict:
        rates, self.field_dnf = dnf_rates(train, as_of, self.p.half_life_days, self.p.dnf_prior_races)
        return rates

    def strength(self, race: pd.DataFrame) -> np.ndarray:
        X, _ = _design(race, self.stats)
        a = np.array([self.a[self.drivers[d]] if d in self.drivers else 0.0 for d in race["driver"]])
        ts = race["team"] + "|" + race["year"].astype(str)
        b = np.array([self.b[self.teams[t]] if t in self.teams else 0.0 for t in ts])
        return X @ self.beta + a + b

    def simulate(self, race: pd.DataFrame, n: int = 4000, seed: int = 0) -> np.ndarray:
        """Sampled finishing positions [n, cars] (1 = winner), DNFs at the back."""
        rng = np.random.default_rng(seed)
        u = self.strength(race)
        dnf_p = np.array([self.dnf.get(d, self.field_dnf) for d in race["driver"]])
        score = self.p.sharpness * u[None, :] + rng.gumbel(size=(n, len(u)))
        score[rng.random((n, len(u))) < dnf_p[None, :]] -= 1e3
        return np.argsort(np.argsort(-score, axis=1), axis=1) + 1


def dnf_rates(results: pd.DataFrame, as_of: str, half_life_days: float = 240.0,
              prior_races: float = 10.0) -> tuple[dict, float]:
    """Each driver's chance of not finishing a race: his recent DNF share (older races weigh
    less) shrunk towards the field's with `prior_races` pseudo-races. Returns (rates, field).

    `results`: earlier races, one row per driver, with date and status (or finished)."""
    from .data import finished

    past = results[results["date"].astype(str) < as_of]
    done = past["finished"] if "finished" in past else past["status"].map(finished)
    age = (pd.Timestamp(as_of) - pd.to_datetime(past["date"])).dt.days.to_numpy()
    w = 0.5 ** (age / half_life_days)
    dnf = (~done.astype(bool)).to_numpy(float)
    field = float((w * dnf).sum() / w.sum()) if len(w) else 0.08
    df = pd.DataFrame({"driver": past["driver"].to_numpy(), "w": w, "wd": w * dnf})
    g = df.groupby("driver")[["w", "wd"]].sum()
    return ((g["wd"] + prior_races * field) / (g["w"] + prior_races)).to_dict(), field


def outcome_probs(positions: np.ndarray) -> dict[str, np.ndarray]:
    return {"win": (positions == 1).mean(0), "podium": (positions <= 3).mean(0),
            "top10": (positions <= 10).mean(0)}
