"""Calibrate the multi-car simulator on 2025 and backtest it on 2026.

Every race is set up with pre-race information only:
  * field: the real grid; each car's base pace from its race pace in the season's earlier
    races (racepace.py), blended with its qualifying gap; qualifying alone when there is
    no race history yet
  * track: degradation, offsets, pit loss and safety-car rate as in strategy_backtest
  * rivals: the single-car simulator's best strategy, stops moved by up to 3 laps

Three global parameters are calibrated on the 2025 races: race-day form spread, first-lap
spread and the overtaking threshold. The target is each race's real position changes among
finishers (mean |finish - grid|). They are then frozen for 2026.

On 2026, for each of the top 5 on the grid, the 40 best single-car strategies are simulated
against the rivals; the best expected finishing position is the recommendation. Compared
with the driver's real first stop and number of stops, the single-car recommendation and the
historical baseline. The whole field's expected order is compared with the real result,
the starting grid, and the finishing-order model of backtest.py.

Usage:
  python -m f1agent.racesim_backtest --out results/racesim
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import finished
from .practice import practice_rates
from .racepace import pace_table, pre_race_pace
from .racesim import Car, Track, position_changes, simulate
from .strategy import simulate as simulate_single
from .strategy import strategies
from .strategy_backtest import DRY, actual_strategies, inputs_for, rows_baseline
from .tyres import FUEL_S_PER_LAP, measure

N_CANDIDATES = 40
N_SIMS = 150
TOP_DRIVERS = 5
GRID = {"form_sd": (0.1, 0.25, 0.4), "start_sd": (0.3, 0.8, 1.5), "pass_threshold": (0.5, 1.5, 3.0)}


def jitter(strategy: tuple, rng: np.random.Generator, max_shift: int = 3, min_stint: int = 6) -> tuple:
    """Move each stop by up to max_shift laps, keeping stints at least min_stint long."""
    lengths = [n for _, n in strategy]
    for i in range(len(lengths) - 1):
        shift = int(rng.integers(-max_shift, max_shift + 1))
        a, b = lengths[i] + shift, lengths[i + 1] - shift
        if a >= min_stint and b >= min_stint:
            lengths[i], lengths[i + 1] = a, b
    return tuple((c, n) for (c, _), n in zip(strategy, lengths))


def quali_gaps(res: pd.DataFrame) -> tuple[list[str], list[float]]:
    r = res.sort_values("grid")
    pole = r["quali_time_s"].min()
    gap = (r["quali_time_s"] - pole).fillna((r["quali_time_s"] - pole).max() + 1.0)
    return r["driver"].tolist(), gap.astype(float).tolist()


def sessions_for(laps: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    s = (laps[laps["session_name"] == "Race"]
         .drop_duplicates("session_key")[["session_key", "year", "meeting_name", "circuit_short_name",
                                          "session_date_start", "hard", "medium", "soft"]]
         .rename(columns={"circuit_short_name": "circuit", "session_date_start": "date"}))
    s = s[s["year"].isin(years) & s["soft"].notna()].sort_values("date")
    for c in DRY:
        s[f"c_{c.lower()}"] = s[c.lower()].str.lstrip("C").astype(float)
    return s


def prepare(s: pd.Series, laps, races, rates_all, fp, paces, rng, fuel):
    """Everything needed to simulate one race, or None for wet or incomplete races."""
    res = races[(races["year"] == s["year"]) & (races["event"] == s["meeting_name"])]
    if res.empty:
        return None
    actual = actual_strategies(laps, s["session_key"], res["driver"])
    if actual.empty or actual["wet"].any():
        return None
    inp, info = inputs_for(s, laps, rates_all, fp, False, fuel)
    if any(np.isnan(v) for v in list(inp.offset.values()) + [inp.pit_loss]):
        return None
    single = simulate_single(inp, strategies(inp.laps, max_stops=2, step=2), n_sims=800)
    best = _parse(single.iloc[0]["strategy"])
    drivers, qgaps = quali_gaps(res)
    base = pre_race_pace(paces, int(s["year"]), s["date"], drivers, qgaps)
    cars = [Car(d, b, jitter(best, rng)) for d, b in zip(drivers, base)]
    track = Track(laps=inp.laps, rate=inp.rate, rate_sd=inp.rate_sd, offset=inp.offset,
                  offset_sd=inp.offset_sd, pit_loss=inp.pit_loss, sc_lap_prob=inp.sc_lap_prob,
                  pass_threshold=1.5)
    fin = res[res["status"].map(finished)]
    target = float((fin["position"] - fin["grid"]).abs().mean())
    return {"s": s, "res": res, "actual": actual, "inp": inp, "best": best, "cars": cars, "track": track,
            "candidates": [_parse(x) for x in single["strategy"].head(N_CANDIDATES)], "target": target}


def calibrate(prepared: list[dict], n_sims: int = 60, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    """Grid search of the three global parameters against real position changes."""
    rows = []
    for fs, ss, th in itertools.product(GRID["form_sd"], GRID["start_sd"], GRID["pass_threshold"]):
        errs, rhos = [], []
        for p in prepared:
            t = Track(**{**p["track"].__dict__, "form_sd": fs, "start_sd": ss, "pass_threshold": th})
            sim = simulate(p["cars"], t, n_sims=n_sims, seed=seed)
            errs.append(position_changes(sim["positions"]) - p["target"])
            rhos.append(_order_spearman(sim, p["res"]))
        rows.append({"form_sd": fs, "start_sd": ss, "pass_threshold": th,
                     "rmse_changes": float(np.sqrt(np.mean(np.square(errs)))),
                     "mean_bias": float(np.mean(errs)), "spearman": float(np.mean(rhos))})
    table = pd.DataFrame(rows).sort_values("rmse_changes")
    best = table.iloc[0]
    return {k: float(best[k]) for k in ("form_sd", "start_sd", "pass_threshold")}, table


def _order_spearman(sim: dict, res: pd.DataFrame) -> float:
    exp_pos = sim["positions"].mean(axis=0)
    actual = res.set_index("driver").loc[sim["names"], "position"].values
    return float(spearmanr(exp_pos, actual).statistic)


def validate(prepared: list[dict], params: dict, laps, races, model_order: pd.DataFrame, seed: int = 0):
    stops_rows, order_rows = [], []
    for p in prepared:
        s, res, actual, inp, best = p["s"], p["res"], p["actual"], p["inp"], p["best"]
        track = Track(**{**p["track"].__dict__, **params})
        cars = p["cars"]
        sim = simulate(cars, track, n_sims=N_SIMS, seed=seed)
        names = sim["names"]
        actual_pos = res.set_index("driver").loc[names, "position"].values
        m = model_order[(model_order["year"] == s["year"]) & (model_order["event"] == s["meeting_name"])
                        & (model_order["method"] == "model")]
        order_rows.append({"meeting_name": s["meeting_name"],
                           "sim_spearman": _order_spearman(sim, res),
                           "grid_spearman": float(spearmanr(np.arange(1, len(names) + 1), actual_pos).statistic),
                           "model_spearman": float(m["spearman"].iloc[0]) if len(m) else np.nan,
                           "sim_changes": position_changes(sim["positions"]), "real_changes": p["target"]})
        base = rows_baseline(laps, races, s["date"])
        single_first = _pit_laps(best)[0]
        for k, car in enumerate(cars[:TOP_DRIVERS]):
            a = actual[actual["driver"] == car.name]
            if a.empty or a["first_stop"].isna().all():
                continue
            scores = []
            for cand in p["candidates"]:
                trial = list(cars)
                trial[k] = Car(car.name, car.base, cand)
                pos = simulate(trial, track, n_sims=N_SIMS, seed=seed)["positions"][:, k]
                scores.append((float(pos.mean()), cand))
            scores.sort(key=lambda x: x[0])
            rec = scores[0][1]
            stops_rows.append({
                "meeting_name": s["meeting_name"], "driver": car.name, "grid": k + 1,
                "actual_first": float(a["first_stop"].iloc[0]), "actual_stops": int(a["stops"].iloc[0]),
                "multi_rec": _describe(rec), "multi_first": _pit_laps(rec)[0], "multi_stops": len(rec) - 1,
                "multi_expected_pos": round(scores[0][0], 2),
                "single_first": single_first, "single_stops": len(best) - 1,
                "baseline_first": round(base[1] * inp.laps) if base else np.nan,
                "baseline_stops": base[0] if base else np.nan,
            })
        print(f"  {s['meeting_name']}: Spearman sim {order_rows[-1]['sim_spearman']:.3f} | grid "
              f"{order_rows[-1]['grid_spearman']:.3f} | changes sim {order_rows[-1]['sim_changes']:.2f} "
              f"vs real {p['target']:.2f}", flush=True)
    per, order = pd.DataFrame(stops_rows), pd.DataFrame(order_rows)
    summary = {"races": len(order), "driver_cases": len(per), "params": params}
    for mth in ("multi", "single", "baseline"):
        err = (per[f"{mth}_first"] - per["actual_first"]).abs()
        summary[mth] = {"first_stop_error_median": round(float(err.median()), 1),
                        "first_stop_error_mean": round(float(err.mean()), 1),
                        "stops_match": round(float((per[f"{mth}_stops"] == per["actual_stops"]).mean()), 3)}
    d = (per["multi_first"] - per["actual_first"]).abs() - (per["single_first"] - per["actual_first"]).abs()
    summary["multi_vs_single_first_stop"] = {"multi_closer": int((d < 0).sum()), "single_closer": int((d > 0).sum()),
                                             "ties": int((d == 0).sum())}
    summary["finishing_order_spearman"] = {
        "simulation": round(float(order["sim_spearman"].mean()), 3),
        "grid": round(float(order["grid_spearman"].mean()), 3),
        "finishing_order_model": round(float(order["model_spearman"].mean()), 3),
        "sim_better_than_grid": int((order["sim_spearman"] > order["grid_spearman"]).sum()),
        "grid_better_than_sim": int((order["sim_spearman"] < order["grid_spearman"]).sum()),
    }
    summary["position_changes"] = {"simulated_mean": round(float(order["sim_changes"].mean()), 2),
                                   "real_mean": round(float(order["real_changes"].mean()), 2)}
    return per, order, summary


def _parse(desc: str) -> tuple:
    names = {"S": "SOFT", "M": "MEDIUM", "H": "HARD"}
    return tuple((names[p.strip()[0]], int(p.strip()[1:])) for p in desc.split("→"))


def _describe(s: tuple) -> str:
    return " → ".join(f"{c[0]}{n}" for c, n in s)


def _pit_laps(s: tuple) -> list[int]:
    return [int(x) for x in np.cumsum([n for _, n in s])[:-1]]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--laps", default="data/laps_clean.csv.gz")
    p.add_argument("--races", default="data/races.csv")
    p.add_argument("--model-order", default="results/backtest/per_race.csv")
    p.add_argument("--out", default="results/racesim")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    laps, races = pd.read_csv(args.laps), pd.read_csv(args.races)
    rates_all, fp, paces = measure(laps, FUEL_S_PER_LAP), practice_rates(laps), pace_table(laps)
    rng = np.random.default_rng(args.seed)
    prep = lambda years: [x for x in (prepare(s, laps, races, rates_all, fp, paces, rng, FUEL_S_PER_LAP)
                                      for _, s in sessions_for(laps, years).iterrows()) if x]
    cal_races = prep([2025])
    print(f"calibrating on {len(cal_races)} races of 2025...", flush=True)
    params, table = calibrate(cal_races, seed=args.seed)
    print("calibrated:", params, flush=True)
    val_races = prep([2026])
    model_order = pd.read_csv(args.model_order)
    per, order, summary = validate(val_races, params, laps, races, model_order, seed=args.seed)
    summary["calibration"] = {"races_2025": len(cal_races), "best": table.iloc[0].to_dict()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "calibration_2025.csv", index=False)
    per.to_csv(out / "drivers.csv", index=False)
    order.to_csv(out / "order.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
