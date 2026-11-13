"""Data layer of the dashboard: strategy plans, race scenarios and real race traces.

A race is identified by a key: "past:<session_key>" for a race already run (inputs use only
what was known before it, as in the backtests) or "next:<meeting name>" for a race still to
come (data/upcoming.csv: lap count, date and the Pirelli nomination, provisional until
Pirelli announces it).

Degradation comes from the model fitted on earlier races only: blending in the weekend's
practice long runs made both the rate estimates and the strategy backtest worse
(docs/test-log.md), so the dashboard does not use them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .practice import practice_rates
from .racepace import decayed_pace, pace_pct_table, pace_table, pre_race_pace_decayed
from .racesim import Car, Track
from .racesim import simulate as simulate_field
from .ranking import dnf_rates
from .strategy import RaceInputs, describe, pit_laps, simulate, strategies
from .strategy_backtest import DRY, inputs_for
from .tyres import FUEL_S_PER_LAP, measure

ROOT = Path(__file__).resolve().parent.parent
# Guards for the dashboard (the backtests use the raw estimates): a softer compound is at most
# this much faster than the hard on new tyres, and never slower. Offsets estimated from few
# stints can be far outside this (e.g. -2.4 s/lap at Mexico 2026).
MAX_OFFSET = 1.2
USE_PRACTICE = False  # see the module docstring
COMPOUND_ORDER = ("HARD", "MEDIUM", "SOFT")  # order of the `compounds` override tuple
DEFAULT_PARAMS = {"form_sd": 0.1, "start_sd": 1.5, "pass_threshold": 3.0}


@dataclass
class Store:
    laps: pd.DataFrame
    races: pd.DataFrame
    upcoming: pd.DataFrame
    grids: pd.DataFrame
    rates_all: pd.DataFrame
    fp: pd.DataFrame
    paces: pd.DataFrame
    pct: pd.DataFrame
    colours: dict
    params: dict


@lru_cache(maxsize=1)
def store(root: str = str(ROOT)) -> Store:
    r = Path(root)
    laps = pd.read_csv(r / "data/laps_clean.csv.gz")
    races = pd.read_csv(r / "data/races.csv")
    upcoming = pd.read_csv(r / "data/upcoming.csv")
    # races already run leave the "next" list (they appear among past races once their laps exist)
    run = set(zip(races["year"], races["event"]))
    upcoming = upcoming[[(y, m) not in run for y, m in zip(upcoming["year"], upcoming["meeting_name"])]]
    grid_path = r / "data/upcoming_grid.csv"
    grids = pd.read_csv(grid_path) if grid_path.exists() else pd.DataFrame(
        columns=["year", "meeting_name", "driver", "grid", "quali_time_s"])
    colours = {}
    drv = r / "data/openf1/drivers.csv.gz"
    if drv.exists():
        d = pd.read_csv(drv).dropna(subset=["team_colour"]).drop_duplicates("name_acronym", keep="last")
        colours = {a: f"#{c}" for a, c in zip(d["name_acronym"], d["team_colour"])}
    summ = r / "results/racesim/summary.json"
    params = json.loads(summ.read_text(encoding="utf-8")).get("params", DEFAULT_PARAMS) if summ.exists() else DEFAULT_PARAMS
    return Store(laps, races, upcoming, grids, measure(laps, FUEL_S_PER_LAP), practice_rates(laps), pace_table(laps),
                 pace_pct_table(laps), colours, params)


# ── Races ───────────────────────────────────────────────────────────────────


def past_races(year: int = 2026) -> pd.DataFrame:
    l = store().laps
    s = (l[(l["session_name"] == "Race") & (l["year"] == year)]
         .drop_duplicates("session_key")[["session_key", "year", "meeting_name", "circuit_short_name",
                                          "session_date_start", "hard", "medium", "soft"]]
         .rename(columns={"circuit_short_name": "circuit", "session_date_start": "date"}))
    return s[s["soft"].notna()].sort_values("date")


def race_choices() -> list[tuple[str, str]]:
    """(label, key) for the dropdown: upcoming races first, then 2026 races, latest first."""
    out = [(f"▶ {u.meeting_name} (next)", f"next:{u.meeting_name}") for u in store().upcoming.itertuples()]
    out += [(f"{p.meeting_name} {p.year}", f"past:{p.session_key}") for p in past_races().iloc[::-1].itertuples()]
    return out


def race_row(key: str) -> tuple[pd.Series, int, dict]:
    """Race description in the form inputs_for expects, its lap count and display info."""
    kind, ident = key.split(":", 1)
    if kind == "next":
        u = store().upcoming.set_index("meeting_name").loc[ident]
        row = pd.Series({"year": int(u["year"]), "meeting_name": ident, "circuit": u["circuit"], "date": u["date"],
                         "hard": u["hard"], "medium": u["medium"], "soft": u["soft"], "session_key": None})
        info = {"upcoming": True, "compounds_status": u["compounds_status"], "source_url": u["source_url"]}
        n_laps = int(u["laps"])
    else:
        p = past_races().set_index("session_key").loc[int(ident)]
        row = pd.Series({**p.to_dict(), "session_key": int(ident)})
        info = {"upcoming": False, "compounds_status": "official", "source_url": None}
        l = store().laps
        n_laps = int(l.loc[l["session_key"] == int(ident), "lap_number"].max())
    for c in DRY:
        row[f"c_{c.lower()}"] = float(str(row[c.lower()]).lstrip("C"))
    return row, n_laps, info


# ── Strategy plan (single car) ──────────────────────────────────────────────


@lru_cache(maxsize=64)
def _inputs(key: str, compounds: tuple, track_temp: float | None) -> tuple[RaceInputs, dict]:
    s = store()
    row, n_laps, info = race_row(key)
    nominated = {c: float(row[f"c_{c.lower()}"]) for c in DRY}
    if compounds:
        # (hard, medium, soft), the order every caller uses (dashboard inputs, agent tool). The
        # first version zipped them with DRY = (SOFT, MEDIUM, HARD) and swapped soft and hard.
        for c, v in zip(COMPOUND_ORDER, compounds):
            row[f"c_{c.lower()}"] = float(v)
    inp, extra = inputs_for(row, s.laps, s.rates_all, s.fp, USE_PRACTICE, FUEL_S_PER_LAP, n_laps=n_laps,
                            track_temp=track_temp)
    if compounds:
        # What-if compounds: the rate model predicts the race's own nomination; a softer or
        # harder choice adds the within-race wear of each C step (tyres.wear_per_c_step).
        k = _wear_per_c_step()
        rate = {c: inp.rate[c] + k * (float(row[f"c_{c.lower()}"]) - nominated[c]) for c in DRY}
        inp = replace(inp, rate=rate)
        extra = {**extra, "rates": {c: round(v, 4) for c, v in rate.items()}, "wear_per_c_step": round(k, 4)}
    return inp, {**info, **extra}


@lru_cache(maxsize=1)
def _wear_per_c_step() -> float:
    from .tyres import wear_per_c_step

    return wear_per_c_step(store().rates_all)


def plan(key: str, compounds: tuple = (), track_temp: float | None = None, pit_loss: float | None = None,
         sc_lap_prob: float | None = None, n_sims: int = 1500, seed: int = 0) -> tuple[RaceInputs, dict, pd.DataFrame]:
    """Inputs (with overrides) and the ranked strategy table for a race.

    `compounds`: optional (hard, medium, soft) C numbers, e.g. (3, 4, 5)."""
    inp, info = _inputs(key, tuple(compounds), track_temp)
    # a negative rate (tyre getting faster with age) is a model artefact on cold tracks: floor at 0
    inp = replace(inp, rate={c: max(float(v), 0.0) for c, v in inp.rate.items()},
                  offset={c: float(np.clip(v, -MAX_OFFSET, 0.0)) for c, v in inp.offset.items()})
    if pit_loss is not None:
        inp = replace(inp, pit_loss=float(pit_loss))
    if sc_lap_prob is not None:
        inp = replace(inp, sc_lap_prob=float(sc_lap_prob))
    res = simulate(inp, strategies(inp.laps, max_stops=2, step=2), n_sims=n_sims, seed=seed)
    return inp, info, res


def degradation_curves(inp: RaceInputs, max_age: int = 35) -> pd.DataFrame:
    """Lap-time delta vs a new hard tyre, by compound and tyre age, with a ±1 sd band of the rate."""
    age = np.arange(0, max_age + 1)
    rows = []
    for c in DRY:
        mid = inp.offset[c] + inp.rate[c] * age
        sd = inp.rate_sd[c] * age
        rows += [{"compound": c, "age": a, "delta": m, "low": m - s, "high": m + s} for a, m, s in zip(age, mid, sd)]
    return pd.DataFrame(rows)


def parse_strategy(desc: str) -> tuple:
    names = {c[0]: c for c in DRY}
    return tuple((names[p.strip()[0]], int(p.strip()[1:])) for p in desc.split("→"))


# ── Real race ───────────────────────────────────────────────────────────────


def real_stints(session_key: int, top: int = 10) -> pd.DataFrame:
    """Stints of the top finishers: driver, compound, first and last lap."""
    s = store()
    l = s.laps[s.laps["session_key"] == session_key]
    order = _finish_order(session_key)[:top]
    st = (l[l["name_acronym"].isin(order)].groupby(["name_acronym", "stint"])
          .agg(compound=("compound", "first"), start=("lap_number", "min"), end=("lap_number", "max"))
          .reset_index().rename(columns={"name_acronym": "driver"}))
    st["rank"] = st["driver"].map({d: i for i, d in enumerate(order)})
    return st.sort_values(["rank", "start"])


def _finish_order(session_key: int) -> list[str]:
    s = store()
    meet = s.laps.loc[s.laps["session_key"] == session_key, ["meeting_name", "year"]].iloc[0]
    r = s.races[(s.races["event"] == meet["meeting_name"]) & (s.races["year"] == meet["year"])]
    return r.sort_values("position")["driver"].tolist()


def real_trace(session_key: int) -> pd.DataFrame:
    """Gap to the leader after every lap (s), from summed lap times; [lap, driver]."""
    l = store().laps
    r = l[(l["session_key"] == session_key)][["name_acronym", "lap_number", "lap_duration"]]
    wide = r.pivot_table(index="lap_number", columns="name_acronym", values="lap_duration")
    # a missing lap time (often lap 1) is filled with the driver's median; laps after retiring stay empty
    last = r.groupby("name_acronym")["lap_number"].max()
    filled = wide.fillna(wide.median())
    for d in filled.columns:
        filled.loc[filled.index > last[d], d] = np.nan
    cum = filled.cumsum()
    return cum.sub(cum.min(axis=1), axis=0)


# ── Whole-field scenario ────────────────────────────────────────────────────


def field(key: str) -> tuple[list[str], list[float], bool]:
    """Drivers in grid order, their pre-race base pace (s/lap) and whether the grid is real."""
    s = store()
    row, _, info = race_row(key)
    if not info["upcoming"]:
        res = s.races[(s.races["year"] == row["year"]) & (s.races["event"] == row["meeting_name"])]
        r = res.sort_values("grid")
        r = r[r["grid"].notna() & (r["grid"] > 0)]
        pole = r["quali_time_s"].min()
        gaps = (r["quali_time_s"] - pole).fillna((r["quali_time_s"] - pole).max() + 1.0).tolist()
        drivers = r["driver"].tolist()
        return drivers, pre_race_pace_decayed(s.pct, row["date"], drivers, gaps, float(pole)), True
    # Upcoming race after qualifying: the real starting grid (update.py saves it).
    g = s.grids[(s.grids["year"] == row["year"]) & (s.grids["meeting_name"] == row["meeting_name"])]
    if len(g):
        g = g.sort_values("grid")
        pole = g["quali_time_s"].min()
        gaps = (g["quali_time_s"] - pole).fillna((g["quali_time_s"] - pole).max() + 1.0).tolist()
        drivers = g["driver"].tolist()
        return drivers, pre_race_pace_decayed(s.pct, row["date"], drivers, gaps, float(pole)), True
    # Upcoming race before qualifying: the field of the latest race, grid ordered by recent race pace.
    last = s.races[s.races["date"] == s.races["date"].max()]
    drivers = last["driver"].tolist()
    pct = decayed_pace(s.pct, str(row["date"])[:10], drivers)
    lap_s = _lap_time(row)
    known = [p for p in pct if not np.isnan(p)]
    worst = max(known) if known else 1.0
    base = [(worst + 0.3 if np.isnan(p) else p) * lap_s / 100.0 for p in pct]
    order = np.argsort(base)
    m = min(base)
    return [drivers[i] for i in order], [base[i] - m for i in order], False


def _lap_time(row) -> float:
    """Typical race lap at this circuit (s), from earlier races there; 90 s if none."""
    l = store().laps
    l = l[(l["session_name"] == "Race") & l["clean"] & (l["meeting_name"] == row["meeting_name"])]
    return float(l["lap_duration"].median()) if len(l) else 90.0


def dnf_chances(key: str, drivers: list[str]) -> list[float]:
    """Each driver's chance of retiring, from his recent races (see ranking.dnf_rates)."""
    row, _, _ = race_row(key)
    rates, field_rate = dnf_rates(store().races, str(row["date"])[:10])
    return [float(rates.get(d, field_rate)) for d in drivers]


def scenario(key: str, driver: str, strat_a: str, strat_b: str, n_sims: int = 300, seed: int = 0,
             rival_strategy: str | None = None, **plan_kw) -> dict:
    """Simulate the whole field twice, `driver` on strategy A then B; rivals as given or the best plan."""
    inp, _, res = plan(key, n_sims=800, **plan_kw)
    params = store().params
    drivers, base, real_grid = field(key)
    dnf = dnf_chances(key, drivers)
    rival = parse_strategy(rival_strategy or res.iloc[0]["strategy"])
    track = Track(laps=inp.laps, rate=inp.rate, rate_sd=inp.rate_sd, offset=inp.offset, offset_sd=inp.offset_sd,
                  pit_loss=inp.pit_loss, sc_lap_prob=inp.sc_lap_prob, **params)
    j = drivers.index(driver)
    out = {"drivers": drivers, "real_grid": real_grid, "grid_pos": j + 1, "laps": inp.laps}
    for tag, desc in (("a", strat_a), ("b", strat_b)):
        # Rivals run the best plan with their stops moved by up to 3 laps, drawn again in every
        # simulated race; the chosen driver runs exactly the strategy asked for.
        cars = [Car(d, b, parse_strategy(desc), dnf[i]) if i == j else Car(d, b, rival, dnf[i], RIVAL_SHIFT)
                for i, (d, b) in enumerate(zip(drivers, base))]
        sim = simulate_field(cars, track, n_sims=n_sims, seed=seed, return_trace=True)
        out[tag] = {"strategy": desc, "positions": sim["positions"][:, j], "trace": sim["trace"],
                    "expected": sim["positions"].mean(axis=0)}
    return out


RIVAL_SHIFT = 3  # laps: how far rivals' stops move around the best plan




# ── Track map ───────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _maps() -> pd.DataFrame:
    f = ROOT / "data/track_maps.csv.gz"
    return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["circuit"])


def track_map(key: str) -> pd.DataFrame | None:
    """Telemetry lap of the race's circuit (t, x, y, speed, lap_duration, s1, s2), or None."""
    circuit = race_row(key)[0]["circuit"]
    m = _maps()
    m = m[m["circuit"] == circuit]
    return m.reset_index(drop=True) if len(m) else None


def on_track(gaps: pd.Series, tmap: pd.DataFrame) -> pd.DataFrame:
    """Place cars on the map from their gap to the leader (s): the leader at the line, a car
    g seconds behind where the reference lap was g seconds before the end (lapped cars wrap)."""
    T = float(tmap["lap_duration"].iloc[0])
    g = gaps.dropna()
    tau = T - np.mod(g.to_numpy(), T)
    idx = np.searchsorted(tmap["t"].to_numpy(), np.clip(tau, 0, tmap["t"].iloc[-1]))
    idx = np.clip(idx, 0, len(tmap) - 1)
    return pd.DataFrame({"driver": g.index, "gap": g.to_numpy(), "x": tmap["x"].to_numpy()[idx],
                         "y": tmap["y"].to_numpy()[idx]})


__all__ = ["track_map", "on_track", "store", "race_choices", "race_row", "plan", "degradation_curves", "parse_strategy", "real_stints",
           "real_trace", "field", "scenario", "describe", "pit_laps"]


# ── Smoothed track and replay data (dashboard drawing) ──────────────────────


@lru_cache(maxsize=32)
def smooth_track(key: str, n: int = 900) -> dict | None:
    """Track outline as a smooth closed curve, rotated so its long side is horizontal and scaled
    to fit [-1, 1]. Points are indexed by lap fraction f (0 = start/finish line), so a car's
    progress through the lap maps straight to a point. Telemetry samples (~4 per second, with
    noise and gaps) are fitted with a periodic smoothing spline in time."""
    from scipy.interpolate import splev, splprep

    m = track_map(key)
    if m is None:
        return None
    T = float(m["lap_duration"].iloc[0])
    f = (m["t"] / T).to_numpy()
    x, y = m["x"].to_numpy(float), m["y"].to_numpy(float)
    xs, ys, fs = np.r_[x, x[0]], np.r_[y, y[0]], np.r_[f, f[0] + 1.0]
    tck, _ = splprep([xs, ys], u=fs, s=len(xs) * 25.0 ** 2, per=1, quiet=2)
    uu = np.linspace(fs[0], fs[0] + 1.0, n, endpoint=False)
    px, py = splev(uu, tck)
    frac = np.mod(uu, 1.0)
    order = np.argsort(frac)
    frac, px, py = frac[order], np.asarray(px)[order], np.asarray(py)[order]
    # rotate (no mirroring) so the principal axis is horizontal
    c = np.c_[px - px.mean(), py - py.mean()]
    w, v = np.linalg.eigh(np.cov(c.T))
    ang = np.arctan2(v[1, -1], v[0, -1])
    rot = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
    r = c @ rot.T
    r /= np.abs(r).max()
    # speed along the curve: periodic interpolation of the samples, lightly smoothed
    sp = np.interp(frac, np.r_[f - 1, f, f + 1], np.tile(m["speed"].to_numpy(float), 3))
    k = 7
    sp = np.convolve(np.r_[sp[-k:], sp, sp[:k]], np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
    s1, s2 = m["s1"].iloc[0], m["s2"].iloc[0]
    return {"f": frac.round(5).tolist(), "x": r[:, 0].round(4).tolist(), "y": r[:, 1].round(4).tolist(),
            "speed": sp.round(0).tolist(), "lap_time": T, "driver": m["driver"].iloc[0],
            "sectors": [round(float(s1) / T, 4) if pd.notna(s1) else None,
                        round(float(s1 + s2) / T, 4) if pd.notna(s1) and pd.notna(s2) else None]}


def replay_data(key: str, gaps: pd.DataFrame, focus: str, title: str, grid: list[str] | None = None) -> dict | None:
    """Everything the on-track replay needs: the smooth track and, per car, the race time at the
    end of every lap (leader at a constant reference lap time plus the car's gap; None after a
    car stops). Cars start in grid order (`grid`, else their order after lap 1), one grid slot
    (0.25 s) apart; the start shuffle then shows during the first lap."""
    tr = smooth_track(key)
    if tr is None:
        return None
    T = tr["lap_time"]
    laps = np.asarray(gaps.index, dtype=int)
    order = [d for d in (grid or []) if d in gaps.columns]
    order += [d for d in gaps.iloc[0].sort_values().index if d not in order]
    slot = {d: i for i, d in enumerate(order)}
    colours = store().colours
    cars = []
    for d in gaps.columns:
        g = gaps[d].to_numpy(float)
        cum = [float(slot[d] * 0.25)] + [None if np.isnan(v) else round(float(l * T + v), 2)
                                                  for l, v in zip(laps, g)]
        cars.append({"name": d, "colour": colours.get(d, "#9AA3B2"), "cum": cum})
    return {"title": title, "track": tr, "laps": int(laps.max()), "lap_time": T, "cars": cars, "focus": focus}
