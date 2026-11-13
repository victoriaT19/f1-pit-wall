"""Tyre degradation: measure it per race, predict it for races that have not happened.

1. Measure. For each race and compound, the degradation rate (seconds lost per lap of
   tyre age) is the slope of fuel-corrected lap time against tyre age, within stints:
   each stint gets its own intercept, so driver and car pace cancel out.

   Fuel: inside a stint, tyre age and lap number grow together, so lap times alone cannot
   separate "lighter car" from "older tyre". A fixed fuel effect is added back
   (FUEL_S_PER_LAP, a commonly quoted F1 figure); `--fuel` tests other values.

2. Predict. A race's rate for a compound is the race's wear level plus a compound step,
   both learned from earlier races only:
     rate = level(race) + step(SOFT / MEDIUM / HARD),  step(HARD) = 0.
   The steps are estimated within races (rate = level_race + step_compound, by weighted
   least squares), so they say how much faster a softer tyre wears *in the same race*. The
   level of a new race comes from a ridge regression on track temperature and the 2026 flag.
   Blending in the levels of earlier editions at the same circuit made the validation worse
   (circuit history was already the weakest simple predictor), so it is not used.

   The first version regressed the rate directly on the real compound number (C1-C6). Across
   races that number is confounded with the track: Pirelli brings softer compounds to tracks
   that are easier on tyres, so "softer" looked like "less wear" and the model predicted the
   soft tyre as the most durable one in every race.

3. Validate. Walk-forward: each race's measured rates are predicted from earlier races
   only, and compared with three simple predictors: the average rate of the same label
   (SOFT/MEDIUM/HARD), of the same real compound (C1-C6), and of the same circuit. Besides
   the error on each rate, it checks what strategy depends on: the difference between two
   compounds of the same race, and whether the softer one is predicted to wear faster.

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


LEVEL_FEATURES = ["track_temp", "regs_2026"]
MIN_RATE_LAPS = 40  # a race-compound needs this many clean laps for its rate to be trusted


def fit_predict_cnumber(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The first version, kept only as a comparison in `validate`: ridge on the compound
    number (C1-C6), track temperature and the 2026 flag. See the module docstring."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = ["compound_c", "track_temp", "regs_2026"]
    tr = rate_features(train, train)
    te = rate_features(train, test)
    tr = tr[tr["laps"] >= MIN_RATE_LAPS]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(tr[feats], tr["rate"], ridge__sample_weight=tr["laps"])
    return model.predict(te[feats].fillna(tr[feats].mean()))


def levels_and_steps(rates: pd.DataFrame, iters: int = 50) -> tuple[pd.DataFrame, dict]:
    """Fit rate = level(race) + step(compound), step(HARD) = 0, weighted by laps.

    Returns one row per race (level, laps, track_temp, year, circuit, date) and the steps.
    Races with a single compound only inform their own level, so the steps come from
    comparisons inside the same race."""
    g = rates[rates["laps"] >= MIN_RATE_LAPS].copy()
    step = {c: 0.0 for c in g["compound"].unique()}
    for _ in range(iters):
        g["resid"] = g["rate"] - g["compound"].map(step)
        lvl = (g["resid"] * g["laps"]).groupby(g["session_key"]).sum() / g["laps"].groupby(g["session_key"]).sum()
        dev = g["rate"] - g["session_key"].map(lvl)
        new = {c: float(np.average(d, weights=w)) for c, (d, w) in
               ((c, (dev[g["compound"] == c], g.loc[g["compound"] == c, "laps"])) for c in step)}
        hard = new.get("HARD", 0.0)
        new = {c: v - hard for c, v in new.items()}
        if max(abs(new[c] - step[c]) for c in step) < 1e-9:
            step = new
            break
        step = new
    g["resid"] = g["rate"] - g["compound"].map(step)
    races = (g.groupby("session_key")
             .apply(lambda x: pd.Series({"level": np.average(x["resid"], weights=x["laps"]),
                                         "laps": x["laps"].sum(),
                                         "track_temp": np.average(x["track_temp"], weights=x["laps"]),
                                         "year": x["year"].iloc[0], "circuit": x["circuit"].iloc[0],
                                         "date": x["date"].iloc[0]}), include_groups=False)
             .reset_index())
    return races, step


def wear_per_c_step(rates: pd.DataFrame) -> float:
    """Extra wear (s/lap per lap of age) for each step softer in the C1-C6 scale, measured
    inside races (deviations from each race's mean), so the track's own wear cancels out.
    Used for what-if compound choices: the step between nominations of the same race."""
    g = rates[rates["laps"] >= MIN_RATE_LAPS]
    g = g[g.groupby("session_key")["compound"].transform("nunique") >= 2]
    w = g["laps"].astype(float)

    def centre(col):
        m = (g[col] * w).groupby(g["session_key"]).transform("sum") / w.groupby(g["session_key"]).transform("sum")
        return g[col] - m

    dr, dc = centre("rate"), centre("compound_c")
    den = float((w * dc * dc).sum())
    return float((w * dr * dc).sum() / den) if den > 0 else 0.0


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predict each `test` row's rate (race-compound) from earlier races in `train`.

    rate = level + step(compound): the level from a ridge regression on track temperature
    and the 2026 flag, the steps from within-race comparisons (levels_and_steps)."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    races, step = levels_and_steps(train)
    races["regs_2026"] = (races["year"].astype(int) >= 2026).astype(int)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(races[LEVEL_FEATURES], races["level"], ridge__sample_weight=races["laps"])
    te = test.copy()
    te["regs_2026"] = (te["year"].astype(int) >= 2026).astype(int)
    te["track_temp"] = te["track_temp"].fillna(races["track_temp"].mean())
    level = model.predict(te[LEVEL_FEATURES])
    return level + te["compound"].map(step).fillna(0.0).to_numpy()


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
        old = dict(zip(test["compound"], fit_predict_cnumber(past, test)))
        label = {c: np.average(g["rate"], weights=g["laps"]) for c, g in past.groupby("compound")}
        cnum = {c: np.average(g["rate"], weights=g["laps"]) for c, g in past.groupby("compound_c")}
        for _, t in test.iterrows():
            circ = past[(past["circuit"] == t["circuit"]) & (past["compound_c"] == t["compound_c"])]
            preds = {
                "model": model[t["compound"]],
                "first_version": old[t["compound"]],
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
    for k in ["model", "first_version", "label_average", "compound_average", "circuit_history"]:
        err = (per[f"pred_{k}"] - per["measured_rate"]).abs()
        summary["rate_mae_s_per_lap"][k] = round(float(np.average(err, weights=per["laps"])), 4)
        m26 = per["year"] >= 2026
        if m26.any():
            summary["rate_mae_2026"][k] = round(float(np.average(err[m26], weights=per.loc[m26, "laps"])), 4)
    d = (per["pred_model"] - per["measured_rate"]).abs() - (per["pred_label_average"] - per["measured_rate"]).abs()
    summary["model_vs_label_average_rows"] = {"model_better": int((d < 0).sum()), "label_better": int((d > 0).sum())}
    summary.update(pair_metrics(per))
    return per, summary


SOFTNESS = {"HARD": 0, "MEDIUM": 1, "SOFT": 2}


def pair_metrics(per: pd.DataFrame) -> dict:
    """What strategy depends on: within a race, the difference between two compounds' rates
    (softer minus harder) and whether the softer is predicted to wear faster."""
    out = {"pair_diff_mae": {}, "pairs_softer_predicted_faster": {}, "pairs": 0}
    preds = [c for c in per.columns if c.startswith("pred_")]
    rows = []
    for _, g in per.groupby("session_key"):
        g = g.assign(s=g["compound"].map(SOFTNESS)).sort_values("s")
        recs = g.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                hard, soft = recs[i], recs[j]
                row = {"measured": soft["measured_rate"] - hard["measured_rate"]}
                for p in preds:
                    row[p] = soft[p] - hard[p]
                rows.append(row)
    pairs = pd.DataFrame(rows)
    out["pairs"] = len(pairs)
    out["measured_softer_faster_share"] = round(float((pairs["measured"] > 0).mean()), 3) if len(pairs) else None
    for p in preds:
        k = p.replace("pred_", "")
        out["pair_diff_mae"][k] = round(float((pairs[p] - pairs["measured"]).abs().mean()), 4)
        out["pairs_softer_predicted_faster"][k] = round(float((pairs[p] > 0).mean()), 3)
    return out


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
