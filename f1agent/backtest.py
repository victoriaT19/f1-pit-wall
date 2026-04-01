"""Walk-forward backtest: predict each race using only the races before it.

For every Grand Prix from the first evaluated season on, the model is trained on all
earlier races and predicts the finishing order. It is compared with three baselines that
need no model: the starting grid, the championship order, and the driver's recent form.

Metrics per race:
  * spearman: rank correlation between predicted and actual order (1 = perfect)
  * mae: mean absolute error in positions
  * winner: 1 if the predicted winner won
  * podium: share of the actual top 3 that the prediction put in its top 3
  * top10: share of the actual top 10 predicted in the top 10

Usage:
  python -m f1agent.backtest --data data/races.csv --from-year 2024 --out results/backtest
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .features import FEATURES, build_features, race_order

METHODS = ["model", "model_delta", "grid", "championship", "form"]


def make_model(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Handles missing values natively (a rookie has no form yet), so nothing is imputed.
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
                                         min_samples_leaf=20, l2_regularization=1.0,
                                         random_state=seed)


def to_order(scores: pd.Series) -> pd.Series:
    """Lower score = better; returns predicted positions 1..n (ties broken by index)."""
    return scores.rank(method="first").astype(int)


def baseline_scores(race: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "grid": race["grid"],
        "championship": race["points_rank"].fillna(race["grid"] + 100),
        "form": race["drv_form5"].fillna(race["grid"] + 100),
    }


def race_metrics(pred_pos: pd.Series, actual: pd.Series) -> dict[str, float]:
    rho = spearmanr(pred_pos, actual).statistic
    top3_pred = set(pred_pos[pred_pos <= 3].index)
    top3_act = set(actual[actual <= 3].index)
    top10_pred = set(pred_pos[pred_pos <= 10].index)
    top10_act = set(actual[actual <= 10].index)
    return {
        "spearman": float(rho),
        "mae": float(np.mean(np.abs(pred_pos.values - actual.values))),
        "winner": float(pred_pos.idxmin() == actual.idxmin()),
        "podium": len(top3_pred & top3_act) / 3.0,
        "top10": len(top10_pred & top10_act) / 10.0,
    }


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def run(feats: pd.DataFrame, from_year: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-race metrics, per-driver predictions)."""
    rows, preds = [], []
    for year, rnd in race_order(feats):
        if year < from_year:
            continue
        train = feats[(feats["year"] < year) | ((feats["year"] == year) & (feats["round"] < rnd))]
        race = feats[(feats["year"] == year) & (feats["round"] == rnd)].set_index("driver")
        # A feature with no data at all yet (no earlier edition of any circuit in the
        # first season) cannot be binned; leave it out until it has values.
        cols = [f for f in FEATURES if train[f].notna().any()]
        model = make_model(seed).fit(train[cols], train["position"])
        scores = {"model": pd.Series(model.predict(race[cols]), index=race.index)}
        # Same features, but the target is the change from the grid slot (gained or lost
        # places); the prediction is grid + predicted change. A regressor on the raw
        # position pulls everyone toward the middle of the field; this keeps the grid as
        # the anchor and only learns the corrections.
        delta = make_model(seed).fit(train[cols], train["position"] - train["grid"])
        scores["model_delta"] = race["grid"] + pd.Series(delta.predict(race[cols]), index=race.index)
        scores.update(baseline_scores(race))
        actual = race["position"]
        for method in METHODS:
            pos = to_order(scores[method])
            m = race_metrics(pos, actual)
            rows.append({"year": year, "round": rnd, "event": race["event"].iloc[0],
                         "method": method, **m})
            if method == "model":
                for drv in race.index:
                    preds.append({"year": year, "round": rnd, "event": race["event"].iloc[0],
                                  "driver": drv, "grid": race.loc[drv, "grid"],
                                  "predicted": int(pos[drv]), "actual": int(actual[drv]),
                                  "score": float(scores["model"][drv])})
    return pd.DataFrame(rows), pd.DataFrame(preds)


def summarize(per_race: pd.DataFrame) -> dict:
    metrics = ["spearman", "mae", "winner", "podium", "top10"]
    summary = {"races": int(per_race.groupby(["year", "round"]).ngroups), "overall": {}, "by_year": {}}
    for method in METHODS:
        sub = per_race[per_race["method"] == method]
        summary["overall"][method] = {k: round(float(sub[k].mean()), 4) for k in metrics}
        for year, g in sub.groupby("year"):
            summary["by_year"].setdefault(str(year), {})[method] = {
                k: round(float(g[k].mean()), 4) for k in metrics}
    pivot = per_race.pivot_table(index=["year", "round"], columns="method", values="spearman")
    comp = {}
    for model_name, base in [("model", "grid"), ("model_delta", "grid"),
                             ("model", "championship"), ("model", "form")]:
        diff = pivot[model_name] - pivot[base]
        wins, losses = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
        comp[f"{model_name} vs {base}"] = {"model_better": wins, "baseline_better": losses,
                      "ties": int(len(diff) - wins - losses),
                      "sign_test_p": round(sign_test(wins, losses), 4)}
    summary["model_vs_baselines_spearman"] = comp
    return summary


def markdown(summary: dict) -> str:
    names = {"model": "Model (position)", "model_delta": "Model (change from grid)", "grid": "Starting grid", "championship": "Championship order",
             "form": "Recent form (last 5)"}
    lines = [f"Walk-forward backtest over {summary['races']} races.", "",
             "| Method | Spearman | MAE (positions) | Winner | Podium | Top 10 |",
             "|---|---|---|---|---|---|"]
    for m in METHODS:
        o = summary["overall"][m]
        lines.append(f"| {names[m]} | {o['spearman']:.3f} | {o['mae']:.2f} | {o['winner']:.0%} | "
                     f"{o['podium']:.0%} | {o['top10']:.0%} |")
    lines += ["", "| Spearman per year | " + " | ".join(names[m] for m in METHODS) + " |",
              "|---|" + "---|" * len(METHODS)]
    for year, d in summary["by_year"].items():
        lines.append(f"| {year} | " + " | ".join(f"{d[m]['spearman']:.3f}" for m in METHODS) + " |")
    lines += ["", "| Spearman, per race | model better | baseline better | ties | sign test p |",
              "|---|---|---|---|---|"]
    for key, c in summary["model_vs_baselines_spearman"].items():
        m, b = key.split(" vs ")
        lines.append(f"| {names[m]} vs {names[b].lower()} | {c['model_better']} | {c['baseline_better']} | {c['ties']} | {c['sign_test_p']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/races.csv")
    p.add_argument("--from-year", type=int, default=2024)
    p.add_argument("--to-date", default=None, help="only races on or before this date (YYYY-MM-DD)")
    p.add_argument("--out", default="results/backtest")
    args = p.parse_args(argv)

    df = pd.read_csv(args.data)
    if args.to_date:
        df = df[df["date"] <= args.to_date]
    feats = build_features(df)
    per_race, preds = run(feats, args.from_year)
    summary = summarize(per_race)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_race.to_csv(out / "per_race.csv", index=False)
    preds.to_csv(out / "predictions.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = markdown(summary)
    (out / "table.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
