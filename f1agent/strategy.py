"""Race strategy simulator: compare pit strategies with Monte Carlo races.

For one car, a strategy is a sequence of stints (compound, number of laps). Its race time,
apart from constants shared by every strategy, is

    sum over laps of [ offset(compound) + rate(compound) * tyre_age ]  +  pit losses

where offset is the compound's pace relative to the hard on new tyres, rate its
degradation per lap of age, and each pit stop costs the circuit's pit loss (less under a
safety car). Fuel burn is the same for every strategy, so it drops out of the comparison.

Uncertainty: each simulated race draws degradation rates and offsets from their estimated
spread, and safety-car periods from the circuit's history. The output is, per strategy,
the distribution of race time relative to the best, and how often it was the fastest.

The inputs are measured from the data (see `race_inputs`):
  * rates: the tyre model (prior) or practice long runs (tyres.py, practice.py)
  * offsets: stints of drivers who ran two compounds in the same race
  * pit loss: in-lap + out-lap minus two normal laps, at the circuit
  * safety-car rate: share of laps neutralised in the circuit's earlier races
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DRY = ["SOFT", "MEDIUM", "HARD"]


@dataclass
class RaceInputs:
    laps: int
    rate: dict           # compound -> mean degradation, s per lap of age
    rate_sd: dict        # compound -> uncertainty of the rate
    offset: dict         # compound -> pace vs HARD on new tyres, s per lap (negative = faster)
    offset_sd: dict
    pit_loss: float      # seconds lost by a stop under green flag
    sc_lap_prob: float   # probability that a given lap starts a safety-car period
    sc_pit_factor: float = 0.5  # share of the pit loss paid when stopping under SC
    notes: dict = field(default_factory=dict)


def strategies(n_laps: int, max_stops: int = 2, step: int = 2, min_stint: int = 8) -> list[tuple]:
    """All strategies with 1..max_stops stops using at least two dry compounds.

    A strategy is a tuple of (compound, laps) stints; pit laps move in `step`-lap increments.
    """
    out = []
    for stops in range(1, max_stops + 1):
        for comps in itertools.product(DRY, repeat=stops + 1):
            if len(set(comps)) < 2:  # the sporting rules require two dry compounds
                continue
            for cuts in itertools.combinations(range(min_stint, n_laps - min_stint + 1, step), stops):
                lengths = np.diff((0,) + cuts + (n_laps,))
                if lengths.min() < min_stint:
                    continue
                out.append(tuple(zip(comps, (int(x) for x in lengths))))
    return out


def stint_time(length: int, offset: float, rate: float, start_age: int = 0) -> float:
    """Sum over the stint's laps of offset + rate * age (age 1..length)."""
    ages = np.arange(start_age + 1, start_age + length + 1)
    return length * offset + rate * ages.sum()


def simulate(inputs: RaceInputs, strats: list[tuple], n_sims: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Monte Carlo race times for each strategy, all strategies sharing the same draws."""
    rng = np.random.default_rng(seed)
    comps = sorted({c for s in strats for c, _ in s})
    # One draw of the world per simulated race, shared by all strategies (common random numbers).
    rate = {c: rng.normal(inputs.rate[c], inputs.rate_sd[c], n_sims) for c in comps}
    off = {c: rng.normal(inputs.offset[c], inputs.offset_sd[c], n_sims) for c in comps}
    sc = rng.random((n_sims, inputs.laps)) < inputs.sc_lap_prob
    # a safety car lasts about 4 laps; a stop made in that window is cheaper
    sc_window = np.zeros_like(sc)
    for k in range(4):
        sc_window[:, k:] |= sc[:, : inputs.laps - k]

    times = np.zeros((len(strats), n_sims))
    for j, s in enumerate(strats):
        lap = 0
        total = np.zeros(n_sims)
        for i, (c, length) in enumerate(s):
            ages_sum = length * (length + 1) / 2
            total += length * off[c] + rate[c] * ages_sum
            lap += length
            if i < len(s) - 1:  # a stop at the end of this stint
                under_sc = sc_window[:, lap - 1]
                total += np.where(under_sc, inputs.sc_pit_factor * inputs.pit_loss, inputs.pit_loss)
        times[j] = total
    best = times.min(axis=0)
    rel = times - best
    res = pd.DataFrame({
        "strategy": [describe(s) for s in strats],
        "stops": [len(s) - 1 for s in strats],
        "pit_laps": [pit_laps(s) for s in strats],
        "compounds": ["-".join(c[0] for c, _ in s) for s in strats],
        "median_gap_s": np.median(rel, axis=1),
        "p10_gap_s": np.percentile(rel, 10, axis=1),
        "p90_gap_s": np.percentile(rel, 90, axis=1),
        "p_fastest": (rel == 0).mean(axis=1),
    })
    return res.sort_values(["median_gap_s", "p90_gap_s"]).reset_index(drop=True)


def describe(s: tuple) -> str:
    return " → ".join(f"{c[0]}{n}" for c, n in s)


def pit_laps(s: tuple) -> list[int]:
    return [int(x) for x in np.cumsum([n for _, n in s])[:-1]]


# ── Measuring the inputs ────────────────────────────────────────────────────


def compound_offsets(race_laps: pd.DataFrame, rates: dict, fuel: float) -> dict:
    """Pace of each compound vs HARD on new tyres, from drivers who ran both in a race.

    For each stint, the fuel- and age-corrected lap time at age 0 is the stint's intercept;
    the offset is the median difference between a driver's intercepts on two compounds.
    """
    from .tyres import corrected

    r = race_laps.copy()
    r["y"] = corrected(r, fuel) - r["compound"].map(rates).fillna(0) * r["tyre_age"]
    st = r.groupby(["session_key", "driver_number", "stint", "compound"])["y"].median().reset_index()
    diffs = {c: [] for c in DRY}
    for _, g in st.groupby(["session_key", "driver_number"]):
        base = g.loc[g["compound"] == "HARD", "y"]
        if base.empty:
            continue
        for c in ("SOFT", "MEDIUM"):
            v = g.loc[g["compound"] == c, "y"]
            if len(v):
                diffs[c].append(float(v.mean() - base.mean()))
    out = {"HARD": (0.0, 0.05)}
    for c in ("SOFT", "MEDIUM"):
        d = np.array(diffs[c])
        out[c] = (float(np.median(d)), float(d.std() / np.sqrt(len(d)) + 0.05)) if len(d) >= 3 else (np.nan, np.nan)
    return out


def pit_loss(race_laps_all: pd.DataFrame) -> float:
    """Seconds lost by a stop: in-lap + out-lap minus two of the driver's normal laps."""
    losses = []
    for _, g in race_laps_all.groupby(["session_key", "driver_number"]):
        g = g.sort_values("lap_number")
        normal = g.loc[g["clean"], "lap_duration"].median()
        ins = g[g["in_lap"] & ~g["neutralised"]]
        for _, row in ins.iterrows():
            out_lap = g[(g["lap_number"] == row["lap_number"] + 1) & g["is_pit_out_lap"].fillna(False).astype(bool)]
            if len(out_lap) and not pd.isna(normal):
                loss = row["lap_duration"] + out_lap["lap_duration"].iloc[0] - 2 * normal
                if 5 < loss < 60:  # drop drive-throughs, red flags and timing glitches
                    losses.append(loss)
    return float(np.median(losses)) if losses else np.nan
