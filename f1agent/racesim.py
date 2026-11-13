"""Lap-by-lap race simulation with all cars: traffic, overtaking and the undercut.

Each car has a base pace (s/lap relative to the field), a strategy (stints of compound and
length) and per-lap noise. Every lap, each car's free-air lap time is

    base + offset(compound) + rate(compound) * tyre_age + noise

Cars are then resolved front to back. A car that would close to within MIN_GAP of the car
ahead overtakes with probability

    p = 1 / (1 + exp(-(advantage - threshold) / PASS_SCALE))

where advantage is how much faster it would have been that lap and threshold is the
circuit's overtaking difficulty (calibrated, see `calibrate_threshold`). If it does not
pass, it crosses the line MIN_GAP behind. A car within DIRTY_AIR_GAP of the car ahead
loses DIRTY_AIR_LOSS seconds that lap. A pit stop adds the pit loss (less under a safety
car); the car rejoins wherever its total time puts it.

The undercut is not coded as a rule: a car that stops earlier runs on new tyres while its
rival is on old ones, and if it gains more than the gap it comes out ahead.

Rivals' pit stops: a car with stop_shift > 0 moves each of its stops by up to that many laps,
drawn again in every simulated race, so no rival carries the same early or late stop through
all of them (which would give some drivers a permanent handicap).

Retirements: each car retires with probability dnf_prob (its recent DNF rate) on a lap drawn
uniformly; a retired car drops to the back, behind every car that retired later. The first
simulated race, the one shown in the replay, has no retirements.

Everything is vectorised over simulated races: arrays are [sims, cars].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_GAP = 0.4          # s, closest a car can follow across the line without passing
DIRTY_AIR_GAP = 1.0    # s, following closer than this costs time
DIRTY_AIR_LOSS = 0.25  # s per lap lost in dirty air (assumption)
PASS_SCALE = 0.25      # s, how sharply pass probability rises with pace advantage
LAP_NOISE = 0.25       # s, lap-to-lap noise of a car
GRID_SPACING = 0.25    # s, time between grid slots at the start
SC_LENGTH = 4          # laps a safety-car period lasts
SC_SPACING = 0.6       # s between cars once the field has closed up behind a safety car


@dataclass
class Car:
    name: str
    base: float          # s/lap relative to the reference car (lower = faster)
    strategy: tuple      # ((compound, laps), ...)
    dnf_prob: float = 0.0  # chance of retiring during the race
    stop_shift: int = 0    # each simulated race moves every stop by up to this many laps


@dataclass
class Track:
    laps: int
    rate: dict
    rate_sd: dict
    offset: dict
    offset_sd: dict
    pit_loss: float
    sc_lap_prob: float
    pass_threshold: float      # s/lap of advantage at which a pass is a coin flip
    sc_pit_factor: float = 0.5
    form_sd: float = 0.0       # s/lap: each driver's race-day pace varies by this much
    start_sd: float = 0.0      # s: time spread added at the start (first-lap shuffle)




MIN_STINT = 6  # laps: a shifted stop never leaves a stint shorter than this


def _sim_plans(strategy: tuple, laps: int, comps: list, n_sims: int, shift: int,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per simulated race and lap: compound index, tyre age and whether the car pits at the end
    of the lap ([n_sims, laps] each). Stops move by up to `shift` laps in each race."""
    lengths = np.array([n for _, n in strategy])
    codes = np.array([comps.index(c) for c, _ in strategy])
    planned = np.cumsum(lengths)[:-1]              # laps completed before each stop
    k = len(planned)
    cuts = np.tile(planned, (n_sims, 1))
    if k and shift > 0:
        moved = np.sort(cuts + rng.integers(-shift, shift + 1, (n_sims, k)), axis=1)
        bounds = np.concatenate([np.zeros((n_sims, 1), int), moved, np.full((n_sims, 1), laps)], axis=1)
        ok = (np.diff(bounds, axis=1) >= MIN_STINT).all(axis=1)
        cuts[ok] = moved[ok]
    lap = np.arange(laps)
    stint = (lap[None, None, :] >= cuts[:, :, None]).sum(axis=1)           # [n_sims, laps]
    starts = np.concatenate([np.zeros((n_sims, 1), int), cuts], axis=1)
    age = lap[None, :] - np.take_along_axis(starts, stint, axis=1) + 1
    pit = np.zeros((n_sims, laps), dtype=bool)
    if k:
        np.put_along_axis(pit, cuts - 1, True, axis=1)
    return codes[stint], age.astype(float), pit


def simulate(cars: list[Car], track: Track, n_sims: int = 300, seed: int = 0,
             return_times: bool = False, return_trace: bool = False) -> dict:
    """Simulate `n_sims` races. Returns finishing positions [sims, cars] (1 = winner).

    With `return_trace`, also the gap to the leader of every car after every lap of the first
    simulated race ("trace", [laps, cars]), for a lap-by-lap replay.
    """
    rng = np.random.default_rng(seed)
    n, L = len(cars), track.laps
    rows = np.arange(n_sims)
    comps = sorted({c for car in cars for c, _ in car.strategy})
    rate = {c: rng.normal(track.rate[c], track.rate_sd[c], (n_sims, 1)) for c in comps}
    off = {c: rng.normal(track.offset[c], track.offset_sd[c], (n_sims, 1)) for c in comps}
    # Race-day form: one pace shift per driver per simulated race.
    base = np.array([car.base for car in cars])[None, :] + rng.normal(0, track.form_sd, (n_sims, len(cars)))
    comp_list = list(comps)
    plans = [_sim_plans(car.strategy, L, comp_list, n_sims, car.stop_shift, rng) for car in cars]
    OFF = np.hstack([off[c] for c in comps])      # [n_sims, compounds]
    RATE = np.hstack([rate[c] for c in comps])

    sc_start = rng.random((n_sims, L)) < track.sc_lap_prob
    sc = np.zeros((n_sims, L), dtype=bool)
    for k in range(SC_LENGTH):
        sc[:, k:] |= sc_start[:, : L - k]

    # Start: grid order is the order of `cars`.
    T = np.tile(np.arange(n) * GRID_SPACING, (n_sims, 1)).astype(float)
    # The start and first lap shuffle positions: a time spread applied before lap 1.
    T += np.abs(rng.normal(0, track.start_sd, (n_sims, n))) if track.start_sd > 0 else 0.0
    trace = np.zeros((L, n))
    dnf_p = np.array([car.dnf_prob for car in cars])
    retire_lap = np.where(rng.random((n_sims, n)) < dnf_p[None, :], rng.integers(0, L, (n_sims, n)), L)
    retire_lap[0] = L  # the replayed race has no retirements
    for lap in range(L):
        free = base + rng.normal(0, LAP_NOISE, (n_sims, n))
        for j, (comp, age, _) in enumerate(plans):
            ci = comp[:, lap]
            free[:, j] += OFF[rows, ci] + RATE[rows, ci] * age[:, lap]
        under_sc = sc[:, lap]
        # Under a safety car nobody passes and every car laps at the same speed; on its first
        # lap the field closes up behind the safety car (gaps shrink to SC_SPACING).
        free[under_sc] = free[under_sc].min(axis=1, keepdims=True)
        first_sc = under_sc & (sc_start[:, lap] if lap == 0 else (sc[:, lap] & ~sc[:, lap - 1]))
        if first_sc.any():
            ranks = np.argsort(np.argsort(T[first_sc], axis=1), axis=1)
            T[first_sc] = T[first_sc].min(axis=1, keepdims=True) + ranks * SC_SPACING
        new = T + free

        order = np.argsort(T, axis=1)  # positions at the start of the lap
        for k in range(1, n):
            car = order[:, k]
            ahead = order[:, k - 1]
            t_car, t_ahead = new[rows, car], new[rows, ahead]
            close = t_car < t_ahead + MIN_GAP
            adv = free[rows, ahead] - free[rows, car]  # how much faster this lap
            p = 1.0 / (1.0 + np.exp(-(adv - track.pass_threshold) / PASS_SCALE))
            passes = close & ~under_sc & (rng.random(n_sims) < p)
            stuck = close & ~passes
            new[rows[stuck], car[stuck]] = t_ahead[stuck] + MIN_GAP
            dirty = (~close) & (new[rows, car] - t_ahead < DIRTY_AIR_GAP) & ~under_sc
            new[rows[dirty], car[dirty]] += DIRTY_AIR_LOSS
        # Pit stops at the end of the lap.
        for j, (_, _, pit) in enumerate(plans):
            stop = pit[:, lap]
            if stop.any():
                loss = np.where(under_sc, track.sc_pit_factor * track.pit_loss, track.pit_loss)
                new[stop, j] += loss[stop]
        T = new
        # Retired cars leave the race: far behind, the earliest retirements last.
        gone = retire_lap <= lap
        if gone.any():
            T[gone] = 1e7 - retire_lap[gone] * 1e3
        trace[lap] = T[0] - T[0].min()
    positions = np.argsort(np.argsort(T, axis=1), axis=1) + 1
    out = {"positions": positions, "names": [c.name for c in cars]}
    if return_times:
        out["times"] = T
    if return_trace:
        out["trace"] = trace
    return out


def position_changes(positions: np.ndarray) -> float:
    """Mean |finish - grid| over cars and simulated races (grid = car order)."""
    grid = np.arange(1, positions.shape[1] + 1)[None, :]
    return float(np.abs(positions - grid).mean())


def calibrate_threshold(cars: list[Car], track: Track, target_changes: float,
                        grid: tuple = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
                        n_sims: int = 120, seed: int = 0) -> float:
    """Pick the overtaking threshold whose simulated position changes best match history."""
    best, best_err = grid[0], np.inf
    for th in grid:
        t = Track(**{**track.__dict__, "pass_threshold": th})
        err = abs(position_changes(simulate(cars, t, n_sims, seed)["positions"]) - target_changes)
        if err < best_err:
            best, best_err = th, err
    return best
