"""Walk-forward backtest: predict each race using only the races before it.

For every Grand Prix from the first evaluated season on, the models are trained on all
earlier races and predict the finishing order:
  * model: gradient boosting on the pre-race features (features.py), target = position;
  * pl: Plackett-Luce ranking model with driver and team-season strengths (ranking.py);
  * ensemble: the mean of the model's and the Plackett-Luce ranks.
They are compared with three baselines that need no model: the starting grid, the
championship order, and the driver's recent form.

The Plackett-Luce hyperparameters were chosen on 2024 only; 2025 and 2026 are held out.

Probabilities (win, podium, top 10) are scored with the Brier score (mean squared error of
the probability, lower is better) against a grid baseline: how often each grid slot won,
reached the podium or the top 10 in earlier races. The ensemble's probabilities are the
mean of the Plackett-Luce and the grid-slot probabilities.

Metrics per race (orders):
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

from .data import finished
from .features import FEATURES, add_race_pace, build_features, race_order
from .ranking import PlackettLuce, outcome_probs

METHODS = ["ensemble", "pl", "model", "model_delta", "grid", "championship", "form"]
PROB_METHODS = ["ensemble", "pl", "grid"]
OUTCOMES = (("win", 1), ("podium", 3), ("top10", 10))


def make_model(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Handles missing values natively (a rookie has no form yet), so nothing is imputed.
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
                                         min_samples_leaf=20, l2_regularization=1.0,
                                         random_state=seed)


def to_order(scores: pd.Series) -> pd.Series:
    """Lower score = better; returns predicted positions 1..n (ties broken by row order, so
    rows must not be in finishing order: run() puts them in grid order)."""
    return scores.rank(method="first").astype(int)


def grid_probs(train: pd.DataFrame, race: pd.DataFrame) -> dict[str, np.ndarray]:
    """How often each grid slot won / reached the podium / the top 10 in earlier races
    (smoothed towards k/20 with two pseudo-races)."""
    out = {}
    for name, k in OUTCOMES:
        hit = (train["position"] <= k).groupby(train["grid"]).agg(["sum", "count"])
        prior = k / 20
        p = (hit["sum"] + 2 * prior) / (hit["count"] + 2)
        out[name] = race["grid"].map(p).fillna(prior).to_numpy(float)
    return out


def predict(train: pd.DataFrame, race: pd.DataFrame, seed: int = 0, n_samples: int = 4000) -> dict:
    """Scores (lower = better) and outcome probabilities for one race, from earlier races only.

    `train` and `race` are feature rows (build_features + add_race_pace); the race rows keep
    their index (the drivers, in the tools)."""
    train = train.assign(finished=train["status"].map(finished))
    cols = [f for f in FEATURES if train[f].notna().any()]
    model = make_model(seed).fit(train[cols], train["position"])
    scores = {"model": pd.Series(model.predict(race[cols]), index=race.index)}
    rows = race.reset_index() if "driver" not in race.columns else race
    pl = PlackettLuce().fit(train, str(race["date"].iloc[0]) or pd.Timestamp.today().strftime("%Y-%m-%d"))
    scores["pl"] = pd.Series(-pl.strength(rows), index=race.index)
    # Mean rank of the two models; ties (frequent with summed ranks) go to the better grid slot.
    scores["ensemble"] = to_order(scores["model"]) + to_order(scores["pl"]) + 1e-3 * race["grid"]
    probs = {"pl": outcome_probs(pl.simulate(rows, n_samples, seed)), "grid": grid_probs(train, race)}
    probs["ensemble"] = {o: 0.5 * probs["pl"][o] + 0.5 * probs["grid"][o] for o in probs["pl"]}
    return {"scores": scores, "probs": probs, "cols": cols}


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
        # Rows in grid order: ties in a score are broken by row order (to_order), and the
        # feature table is sorted by finishing position, which would leak the result.
        race = (feats[(feats["year"] == year) & (feats["round"] == rnd)]
                .sort_values(["grid", "driver"]).set_index("driver"))
        out = predict(train, race, seed)
        scores, probs = dict(out["scores"]), out["probs"]
        # A feature with no data at all yet (no earlier edition of any circuit in the
        # first season) cannot be binned; leave it out until it has values.
        cols = out["cols"]
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
            if method in probs:
                for o, k in OUTCOMES:
                    m[f"brier_{o}"] = float(np.mean((probs[method][o] - (actual.to_numpy() <= k)) ** 2))
            rows.append({"year": year, "round": rnd, "event": race["event"].iloc[0],
                         "method": method, **m})
            if method == "ensemble":
                for i, drv in enumerate(race.index):
                    preds.append({"year": year, "round": rnd, "event": race["event"].iloc[0],
                                  "driver": drv, "grid": race.loc[drv, "grid"],
                                  "predicted": int(pos[drv]), "actual": int(actual[drv]),
                                  "p_win": round(float(probs["ensemble"]["win"][i]), 4),
                                  "p_podium": round(float(probs["ensemble"]["podium"][i]), 4),
                                  "p_top10": round(float(probs["ensemble"]["top10"][i]), 4)})
    return pd.DataFrame(rows), pd.DataFrame(preds)


def summarize(per_race: pd.DataFrame) -> dict:
    metrics = ["spearman", "mae", "winner", "podium", "top10"]
    briers = [f"brier_{o}" for o, _ in OUTCOMES]
    summary = {"races": int(per_race.groupby(["year", "round"]).ngroups), "overall": {}, "by_year": {},
               "held_out_2025_2026": {}, "brier": {}, "brier_held_out_2025_2026": {}}
    held = per_race[per_race["year"] >= 2025]
    present = [m for m in METHODS if (per_race["method"] == m).any()]
    for method in present:
        sub = per_race[per_race["method"] == method]
        summary["overall"][method] = {k: round(float(sub[k].mean()), 4) for k in metrics}
        h = held[held["method"] == method]
        if len(h):
            summary["held_out_2025_2026"][method] = {k: round(float(h[k].mean()), 4) for k in metrics}
        for year, g in sub.groupby("year"):
            summary["by_year"].setdefault(str(year), {})[method] = {
                k: round(float(g[k].mean()), 4) for k in metrics}
        if method in PROB_METHODS and briers[0] in sub and sub[briers[0]].notna().any():
            summary["brier"][method] = {k: round(float(sub[k].mean()), 4) for k in briers}
            if len(h):
                summary["brier_held_out_2025_2026"][method] = {k: round(float(h[k].mean()), 4) for k in briers}
    pivot = per_race.pivot_table(index=["year", "round"], columns="method", values="spearman")
    comp = {}
    for model_name, base in [("ensemble", "grid"), ("ensemble", "model"), ("pl", "grid"),
                             ("model", "grid"), ("model_delta", "grid"),
                             ("model", "championship"), ("model", "form")]:
        if model_name not in present or base not in present:
            continue
        diff = pivot[model_name] - pivot[base]
        wins, losses = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
        comp[f"{model_name} vs {base}"] = {"model_better": wins, "baseline_better": losses,
                      "ties": int(len(diff) - wins - losses),
                      "sign_test_p": round(sign_test(wins, losses), 4)}
    summary["model_vs_baselines_spearman"] = comp
    return summary


def markdown(summary: dict) -> str:
    names = {"ensemble": "Ensemble (model + Plackett-Luce)", "pl": "Plackett-Luce",
             "model": "Model (position)", "model_delta": "Model (change from grid)",
             "grid": "Starting grid", "championship": "Championship order",
             "form": "Recent form (last 5)"}
    lines = [f"Walk-forward backtest over {summary['races']} races.", "",
             "| Method | Spearman | MAE (positions) | Winner | Podium | Top 10 |",
             "|---|---|---|---|---|---|"]
    for m in summary["overall"]:
        o = summary["overall"][m]
        lines.append(f"| {names[m]} | {o['spearman']:.3f} | {o['mae']:.2f} | {o['winner']:.0%} | "
                     f"{o['podium']:.0%} | {o['top10']:.0%} |")
    ms = list(summary["overall"])
    lines += ["", "| Spearman per year | " + " | ".join(names[m] for m in ms) + " |",
              "|---|" + "---|" * len(ms)]
    for year, d in summary["by_year"].items():
        lines.append(f"| {year} | " + " | ".join(f"{d[m]['spearman']:.3f}" for m in ms) + " |")
    lines += ["", "| Spearman, per race | model better | baseline better | ties | sign test p |",
              "|---|---|---|---|---|"]
    for key, c in summary["model_vs_baselines_spearman"].items():
        m, b = key.split(" vs ")
        lines.append(f"| {names[m]} vs {names[b].lower()} | {c['model_better']} | {c['baseline_better']} | {c['ties']} | {c['sign_test_p']} |")
    if summary.get("held_out_2025_2026"):
        lines += ["", "Held out (2025-2026, not used to choose any hyperparameter):", "",
                  "| Method | Spearman | MAE | Winner | Podium | Top 10 |", "|---|---|---|---|---|---|"]
        for m, o in summary["held_out_2025_2026"].items():
            lines.append(f"| {names[m]} | {o['spearman']:.3f} | {o['mae']:.2f} | {o['winner']:.0%} | "
                         f"{o['podium']:.0%} | {o['top10']:.0%} |")
    if summary.get("brier"):
        lines += ["", "Probabilities, Brier score (lower is better):", "",
                  "| Method | win | podium | top 10 | win (2025-26) | podium (2025-26) | top 10 (2025-26) |",
                  "|---|---|---|---|---|---|---|"]
        for m, o in summary["brier"].items():
            h = summary["brier_held_out_2025_2026"].get(m, {})
            lines.append(f"| {names[m]} | {o['brier_win']:.4f} | {o['brier_podium']:.4f} | {o['brier_top10']:.4f} | "
                         f"{h.get('brier_win', float('nan')):.4f} | {h.get('brier_podium', float('nan')):.4f} | "
                         f"{h.get('brier_top10', float('nan')):.4f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/races.csv")
    p.add_argument("--from-year", type=int, default=2024)
    p.add_argument("--to-date", default=None, help="only races on or before this date (YYYY-MM-DD)")
    p.add_argument("--laps", default="data/laps_clean.csv.gz")
    p.add_argument("--out", default="results/backtest")
    args = p.parse_args(argv)

    df = pd.read_csv(args.data)
    if args.to_date:
        df = df[df["date"] <= args.to_date]
    feats = add_race_pace(build_features(df), pd.read_csv(args.laps))
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
