"""Tyre degradation: measure it per race, predict it for races that have not happened.

1. Measure. For each race and compound, the degradation rate (seconds lost per lap of
   tyre age) is the slope of fuel-corrected lap time against tyre age, within stints:
   each stint gets its own intercept, so driver and car pace cancel out.

   Fuel: inside a stint, tyre age and lap number grow together, so lap times alone cannot
   separate "lighter car" from "older tyre". A fixed fuel effect is added back
   (FUEL_S_PER_LAP, a commonly quoted F1 figure); `--fuel` tests other values.

2. Predict. For a race, a model trained only on earlier races predicts each compound's
   rate from the real Pirelli compound (C1-C6), track temperature, the circuit's past
   rates and whether it is a 2026-regulation car.

3. Validate. Walk-forward: each race's measured rates are predicted from earlier races
   only, and compared with three simple predictors: the average rate of the same label
   (SOFT/MEDIUM/HARD), of the same real compound (C1-C6), and of the same circuit.

Usage:
  python -m f1agent.tyres --laps data/laps_clean.csv.gz --out results/tyres
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FUEL_S_PER_LAP = 0.055  # lap time gained per lap as fuel burns; assumption, see --fuel
MIN_STINT_LAPS = 6


def corrected(df: pd.DataFrame, fuel: float = FUEL_S_PER_LAP) -> pd.Series:
    """Lap time with the fuel gain added back, so a flat line means no tyre wear."""
    return df["lap_duration"] + fuel * (df["lap_number"] - 1)


def race_laps(laps: pd.DataFrame) -> pd.DataFrame:
    r = laps[(laps["session_name"] == "Race") & laps["clean"] & laps["compound_c"].notna()].copy()
    r["stint_id"] = r["session_key"].astype(str) + "_" + r["driver_number"].astype(str) + "_" + r["stint"].astype(str)
    size = r.groupby("stint_id")["lap_number"].transform("size")
    return r[size >= MIN_STINT_LAPS]


def within_slope(g: pd.DataFrame, y: str = "y") -> tuple[float, int]:
    """Pooled within-stint OLS slope of y on tyre age (stint fixed effects)."""
    yd = g[y] - g.groupby("stint_id")[y].transform("mean")
    xd = g["tyre_age"] - g.groupby("stint_id")["tyre_age"].transform("mean")
    den = float((xd ** 2).sum())
    return (float((xd * yd).sum() / den) if den > 0 else np.nan), len(g)


def measure(laps: pd.DataFrame, fuel: float = FUEL_S_PER_LAP) -> pd.DataFrame:
    """Degradation rate per race and compound."""
    r = race_laps(laps)
    r["y"] = corrected(r, fuel)
    rows = []
    for (key, comp), g in r.groupby(["session_key", "compound"]):
        slope, n = within_slope(g)
        rows.append({
            "session_key": key, "year": int(g["year"].iloc[0]), "meeting_name": g["meeting_name"].iloc[0],
            "circuit": g["circuit_short_name"].iloc[0], "date": g["session_date_start"].iloc[0],
            "compound": comp, "compound_c": float(g["compound_c"].iloc[0]), "rate": slope, "laps": n,
            "stints": int(g["stint_id"].nunique()), "track_temp": float(g["track_temperature"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["date", "compound"]).reset_index(drop=True)


def rate_features(rates: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Features for predicting `target` rows from earlier `rates` (no peeking)."""
    out = target.copy()
    hist = []
    for _, t in target.iterrows():
        past = rates[(rates["date"] < t["date"]) & (rates["circuit"] == t["circuit"])]
        same_c = past[past["compound_c"] == t["compound_c"]]
        hist.append(np.average(same_c["rate"], weights=same_c["laps"]) if len(same_c) else
                    (np.average(past["rate"], weights=past["laps"]) if len(past) else np.nan))
    out["circuit_hist_rate"] = hist
    out["regs_2026"] = (out["year"] >= 2026).astype(int)
    return out


FEATURES = ["compound_c", "track_temp", "regs_2026"]
MIN_RATE_LAPS = 40  # a race-compound needs this many clean laps for its rate to be trusted


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Ridge regression on compound, track temperature and the 2026 flag.

    A gradient-boosted model with the circuit's history did worse than the simple label
    average on ~160 examples (it fitted noise); this linear model was the best of the
    simple predictors compared in docs/test-log.md.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr = rate_features(train, train)
    te = rate_features(train, test)
    tr = tr[tr["laps"] >= MIN_RATE_LAPS]
    # Standardise first: otherwise the penalty treats °C and compound steps differently.
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(tr[FEATURES], tr["rate"], ridge__sample_weight=tr["laps"])
    return model.predict(te[FEATURES].fillna(tr[FEATURES].mean()))


def stint_errors(laps_r: pd.DataFrame, rate_by_compound: dict, fuel: float) -> list[float]:
    """Mean absolute error (s/lap) predicting laps 3.. of each stint from its first two."""
    errs = []
    for _, g in laps_r.groupby("stint_id"):
        g = g.sort_values("lap_number")
        rate = rate_by_compound.get(g["compound"].iloc[0])
        if rate is None or np.isnan(rate) or len(g) < 4:
            continue
        y = corrected(g, fuel).values
        age = g["tyre_age"].values
        anchor = y[:2].mean() - rate * age[:2].mean()
        pred = anchor + rate * age[2:]
        errs.extend(np.abs(pred - y[2:]).tolist())
    return errs


def validate(laps: pd.DataFrame, fuel: float = FUEL_S_PER_LAP, min_history: int = 10) -> tuple[pd.DataFrame, dict]:
    """Walk-forward: each race's rates predicted only from earlier races."""
    rates = measure(laps, fuel)
    good = rates[rates["laps"] >= MIN_RATE_LAPS]
    rows = []
    for _, s in good.drop_duplicates("session_key").sort_values("date").iterrows():
        past = good[good["date"] < s["date"]]
        if past["session_key"].nunique() < min_history:
            continue
        test = good[good["session_key"] == s["session_key"]]
        model = dict(zip(test["compound"], fit_predict(past, test)))
        label = {c: np.average(g["rate"], weights=g["laps"]) for c, g in past.groupby("compound")}
        cnum = {c: np.average(g["rate"], weights=g["laps"]) for c, g in past.groupby("compound_c")}
        for _, t in test.iterrows():
            circ = past[(past["circuit"] == t["circuit"]) & (past["compound_c"] == t["compound_c"])]
            preds = {
                "model": model[t["compound"]],
                "label_average": label.get(t["compound"], np.nan),
                "compound_average": cnum.get(t["compound_c"], label.get(t["compound"], np.nan)),
                "circuit_history": (np.average(circ["rate"], weights=circ["laps"]) if len(circ)
                                    else cnum.get(t["compound_c"], label.get(t["compound"], np.nan))),
            }
            rows.append({"session_key": s["session_key"], "year": int(s["year"]), "meeting_name": s["meeting_name"],
                         "compound": t["compound"], "compound_c": t["compound_c"], "laps": int(t["laps"]),
                         "measured_rate": t["rate"], **{f"pred_{k}": v for k, v in preds.items()}})
    per = pd.DataFrame(rows)
    summary = {"fuel_s_per_lap": fuel, "races_evaluated": int(per["session_key"].nunique()),
               "rows": len(per), "rate_mae_s_per_lap": {}, "rate_mae_2026": {}}
    for k in ["model", "label_average", "compound_average", "circuit_history"]:
        err = (per[f"pred_{k}"] - per["measured_rate"]).abs()
        summary["rate_mae_s_per_lap"][k] = round(float(np.average(err, weights=per["laps"])), 4)
        m26 = per["year"] >= 2026
        if m26.any():
            summary["rate_mae_2026"][k] = round(float(np.average(err[m26], weights=per.loc[m26, "laps"])), 4)
    d = (per["pred_model"] - per["measured_rate"]).abs() - (per["pred_label_average"] - per["measured_rate"]).abs()
    summary["model_vs_label_average_rows"] = {"model_better": int((d < 0).sum()), "label_better": int((d > 0).sum())}
    return per, summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--laps", default="data/laps_clean.csv.gz")
    p.add_argument("--out", default="results/tyres")
    p.add_argument("--fuel", type=float, nargs="+", default=[FUEL_S_PER_LAP])
    args = p.parse_args(argv)
    laps = pd.read_csv(args.laps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    measure(laps, args.fuel[0]).to_csv(out / "rates.csv", index=False)
    summaries = []
    for fuel in args.fuel:
        per, summary = validate(laps, fuel)
        summaries.append(summary)
        if fuel == args.fuel[0]:
            per.to_csv(out / "validation.csv", index=False)
        print(json.dumps(summary, indent=2))
    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
