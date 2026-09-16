"""Use the weekend's practice long runs to update the degradation forecast.

Teams run race-like stints in practice. The same within-stint method used for races
(tyres.measure) is applied to practice stints with at least MIN_FP_STINT clean laps,
pooling FP1-FP3 of a weekend. The weekend forecast then combines the prior (predicted
from earlier races) with the practice measurement, weighting practice by how many
clean long-run laps it has:

    rate = w * practice_rate + (1 - w) * prior_rate,   w = n_fp / (n_fp + n0)

n0 says how many practice laps count as much as the prior. Evaluated on every 2026 race
that has practice data, against the rate actually measured in the race.

Usage:
  python -m f1agent.practice --laps data/laps_clean.csv.gz --out results/practice
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .tyres import FUEL_S_PER_LAP, MIN_RATE_LAPS, corrected, fit_predict, measure, within_slope

MIN_FP_STINT = 5


LONG_RUN_TOLERANCE = 1.035  # a long-run lap is within 3.5% of the stint's fastest lap


def long_run_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Practice laps that belong to race-like long runs.

    The race filter (within 107% of the driver's median) does not work in practice: the
    median includes slow cool-down laps, so 80-85 s laps pass at a track where race pace is
    71 s. Here a lap counts only if it is within 3.5% of its own stint's fastest lap, and a
    stint counts only with at least MIN_FP_STINT such laps: this keeps long runs and drops
    single push laps and cool-down laps.
    """
    p = laps[laps["session_name"].str.startswith("Practice") & laps["compound_c"].notna()
             & laps["lap_duration"].notna() & ~laps["in_lap"]
             & ~laps["is_pit_out_lap"].fillna(False).astype(bool) & ~laps["neutralised"]].copy()
    p["stint_id"] = (p["session_key"].astype(str) + "_" + p["driver_number"].astype(str) + "_" + p["stint"].astype(str))
    fastest = p.groupby("stint_id")["lap_duration"].transform("min")
    p = p[p["lap_duration"] <= LONG_RUN_TOLERANCE * fastest]
    return p[p.groupby("stint_id")["lap_number"].transform("size") >= MIN_FP_STINT]


def practice_rates(laps: pd.DataFrame, fuel: float = FUEL_S_PER_LAP) -> pd.DataFrame:
    """Degradation per weekend and compound from practice long runs."""
    p = long_run_laps(laps)
    p["y"] = corrected(p, fuel)
    rows = []
    for (meeting, year, comp), g in p.groupby(["meeting_name", "year", "compound"]):
        slope, n = within_slope(g)
        rows.append({"meeting_name": meeting, "year": int(year), "compound": comp,
                     "fp_rate": slope, "fp_laps": n, "fp_stints": int(g["stint_id"].nunique())})
    return pd.DataFrame(rows)


def evaluate(laps: pd.DataFrame, n0_values=(10, 30, 100), fuel: float = FUEL_S_PER_LAP) -> tuple[pd.DataFrame, dict]:
    race = measure(laps, fuel)
    race = race[race["laps"] >= MIN_RATE_LAPS]
    fp = practice_rates(laps, fuel)
    rows = []
    for key, test in race[race["year"] >= 2026].groupby("session_key"):
        date = test["date"].iloc[0]
        past = race[race["date"] < date]
        test = test.assign(prior=fit_predict(past, test))
        test = test.merge(fp, on=["meeting_name", "year", "compound"], how="left")
        rows.append(test)
    per = pd.concat(rows, ignore_index=True)
    per = per[per["fp_rate"].notna() & (per["fp_laps"] >= 10)].copy()
    err = lambda col: float(np.average((per[col] - per["rate"]).abs(), weights=per["laps"]))
    summary = {"race_compounds": len(per), "races": int(per["session_key"].nunique()),
               "median_fp_laps": float(per["fp_laps"].median()),
               "mae_s_per_lap": {"prior_only": round(err("prior"), 4), "practice_only": round(err("fp_rate"), 4)}}
    for n0 in n0_values:
        w = per["fp_laps"] / (per["fp_laps"] + n0)
        per[f"blend_{n0}"] = w * per["fp_rate"] + (1 - w) * per["prior"]
        summary["mae_s_per_lap"][f"blend_n0_{n0}"] = round(err(f"blend_{n0}"), 4)
    d = (per["blend_30"] - per["rate"]).abs() - (per["prior"] - per["rate"]).abs()
    summary["blend30_vs_prior_rows"] = {"blend_better": int((d < 0).sum()), "prior_better": int((d > 0).sum())}
    return per, summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--laps", default="data/laps_clean.csv.gz")
    p.add_argument("--out", default="results/practice")
    args = p.parse_args(argv)
    per, summary = evaluate(pd.read_csv(args.laps))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_race_compound.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
