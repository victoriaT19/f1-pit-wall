import numpy as np
import pandas as pd
import pytest

from f1agent.backtest import race_metrics, sign_test, summarize, to_order
from f1agent.data import finished
from f1agent.features import FEATURES, build_features

DRIVERS = ["AAA", "BBB", "CCC", "DDD"]
TEAMS = {"AAA": "Red", "BBB": "Red", "CCC": "Blue", "DDD": "Blue"}


def toy_races(n_rounds=6, years=(2024, 2025), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        for rnd in range(1, n_rounds + 1):
            order = rng.permutation(DRIVERS)
            grid = rng.permutation(len(DRIVERS)) + 1
            for pos, (drv, g) in enumerate(zip(order, grid), start=1):
                rows.append({
                    "year": year, "round": rnd, "event": f"GP{rnd}", "date": f"{year}-0{min(rnd,9)}-01",
                    "driver": drv, "team": TEAMS[drv], "grid": float(g),
                    "quali_time_s": 80 + rng.random(), "position": float(pos),
                    "status": "Finished" if pos < 4 else "Engine", "points": float(10 - 2 * pos),
                    "laps": 50.0,
                })
    return pd.DataFrame(rows)


def test_finished_status():
    assert finished("Finished") and finished("+1 Lap") and finished("Lapped")
    assert not finished("Engine") and not finished(None)


def test_features_do_not_use_the_race_itself_or_the_future():
    df = toy_races()
    base = build_features(df)
    # Change the result of 2025 round 3: its own features and every earlier race must not move.
    changed = df.copy()
    mask = (changed["year"] == 2025) & (changed["round"] == 3)
    changed.loc[mask, "position"] = changed.loc[mask, "position"].values[::-1]
    changed.loc[mask, "points"] = changed.loc[mask, "points"].values[::-1]
    after = build_features(changed)
    key = ["year", "round", "driver"]
    feat_cols = [c for c in FEATURES if c not in ("grid", "quali_gap_pct", "season_round")]
    a = base.set_index(key)[feat_cols].sort_index()
    b = after.set_index(key)[feat_cols].sort_index()
    upto = [(y, r) for y, r in zip(a.index.get_level_values(0), a.index.get_level_values(1))
            if (y, r) <= (2025, 3)]
    sel = [i for i in a.index if (i[0], i[1]) in set(upto)]
    pd.testing.assert_frame_equal(a.loc[sel], b.loc[sel])
    # ...while a later race does see the change.
    later = [i for i in a.index if (i[0], i[1]) == (2025, 4)]
    assert not a.loc[later].equals(b.loc[later])


def test_first_race_has_no_history():
    feats = build_features(toy_races())
    first = feats[(feats["year"] == 2024) & (feats["round"] == 1)]
    assert first["drv_form5"].isna().all()
    assert first["points_rank"].isna().all()


def test_season_form_resets_each_year():
    feats = build_features(toy_races())
    r1 = feats[(feats["year"] == 2025) & (feats["round"] == 1)]
    assert r1["drv_season_form"].isna().all()
    assert r1["drv_form5"].notna().all()


def test_points_rank_matches_cumulative_points():
    df = toy_races()
    feats = build_features(df)
    season = df[(df["year"] == 2024) & (df["round"] < 4)]
    leader = season.groupby("driver")["points"].sum().idxmax()
    r4 = feats[(feats["year"] == 2024) & (feats["round"] == 4)].set_index("driver")
    assert r4.loc[leader, "points_rank"] == 1


def test_race_metrics_perfect_and_reversed():
    actual = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], index=list("abcdefghijkl"), dtype=float)
    perfect = race_metrics(actual.astype(int), actual)
    assert perfect["spearman"] == pytest.approx(1.0)
    assert perfect["mae"] == 0 and perfect["winner"] == 1 and perfect["podium"] == 1 and perfect["top10"] == 1
    rev = race_metrics((13 - actual).astype(int), actual)
    assert rev["spearman"] == pytest.approx(-1.0) and rev["winner"] == 0 and rev["podium"] == 0


def test_to_order_lower_score_is_better():
    s = pd.Series([3.2, 1.1, 2.5], index=["x", "y", "z"])
    assert to_order(s).to_dict() == {"x": 3, "y": 1, "z": 2}


def test_sign_test():
    assert sign_test(0, 0) == 1.0
    assert sign_test(10, 0) < 0.01
    assert sign_test(5, 5) == 1.0


def test_summarize_counts_wins():
    rows = []
    for rnd in range(1, 4):
        for method, rho in [("model", 0.8), ("model_delta", 0.7), ("grid", 0.5), ("championship", 0.9), ("form", 0.8)]:
            rows.append({"year": 2024, "round": rnd, "event": "x", "method": method, "spearman": rho,
                         "mae": 1.0, "winner": 1.0, "podium": 1.0, "top10": 1.0})
    s = summarize(pd.DataFrame(rows))
    c = s["model_vs_baselines_spearman"]
    assert c["model vs grid"]["model_better"] == 3 and c["model vs championship"]["baseline_better"] == 3
    assert c["model vs form"]["ties"] == 3
