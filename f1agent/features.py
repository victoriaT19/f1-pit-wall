"""Pre-race features, computed only from races that happened before.

The original project computed each driver's averages over every race, including the one
being predicted, so the model saw the answer during training. Here each race's features
come from strictly earlier races, which `test_features.py` checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import finished

FEATURES = [
    "grid",             # starting slot (known before the start)
    "quali_gap_pct",    # best qualifying time vs pole, in %
    "drv_form5",        # driver's mean finish in the last 5 races (any season)
    "drv_season_form",  # driver's mean finish this season so far
    "team_form5",       # team's mean finish in its last 5 races
    "team_season_form", # team's mean finish this season so far
    "points_rank",      # championship position before the race (1 = leader)
    "dnf_rate10",       # share of the driver's last 10 races not finished
    "season_round",     # how far into the season (form is noisier early on)
    "track_overtaking", # mean |finish - grid| at this circuit in earlier editions
    "drv_track_hist",   # driver's mean finish at this circuit in earlier editions
]


def race_order(df: pd.DataFrame) -> list[tuple[int, int]]:
    return sorted(set(zip(df["year"], df["round"])))


def _quali_gap(race: pd.DataFrame) -> pd.Series:
    t = race["quali_time_s"]
    pole = t.min(skipna=True)
    gap = (t / pole - 1.0) * 100.0
    # No time (crash, no lap set): just behind the slowest driver who set one.
    worst = gap.max(skipna=True)
    return gap.fillna((0.0 if pd.isna(worst) else worst) + 0.5)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per driver per race with FEATURES, the target and identifiers."""
    # Chronological order, so that tail(n) means "the last n races".
    df = df.sort_values(["year", "round", "position"]).reset_index(drop=True)
    df["finished"] = df["status"].map(finished)
    out = []
    order = race_order(df)
    for year, rnd in order:
        race = df[(df["year"] == year) & (df["round"] == rnd)].copy()
        past = df[(df["year"] < year) | ((df["year"] == year) & (df["round"] < rnd))]
        season = past[past["year"] == year]
        race["quali_gap_pct"] = _quali_gap(race)
        race["season_round"] = rnd

        pts = season.groupby("driver")["points"].sum().sort_values(ascending=False)
        ranks = {d: i + 1 for i, d in enumerate(pts.index)}

        # Earlier editions of the same Grand Prix (matched by event name).
        same_track = past[past["event"] == race["event"].iloc[0]]
        race["track_overtaking"] = ((same_track["position"] - same_track["grid"]).abs().mean()
                                    if len(same_track) else np.nan)

        rows = []
        for _, r in race.iterrows():
            d_hist = past[past["driver"] == r["driver"]]
            t_hist = past[past["team"] == r["team"]]
            d_season = season[season["driver"] == r["driver"]]
            t_season = season[season["team"] == r["team"]]
            last_t_races = race_order(t_hist)[-5:]
            t_last = t_hist[[(y, k) in set(last_t_races) for y, k in zip(t_hist["year"], t_hist["round"])]]
            rows.append({
                "drv_form5": d_hist["position"].tail(5).mean() if len(d_hist) else np.nan,
                "drv_season_form": d_season["position"].mean() if len(d_season) else np.nan,
                "team_form5": t_last["position"].mean() if len(t_last) else np.nan,
                "team_season_form": t_season["position"].mean() if len(t_season) else np.nan,
                "points_rank": ranks.get(r["driver"], len(ranks) + 1) if ranks else np.nan,
                "dnf_rate10": 1.0 - d_hist["finished"].tail(10).mean() if len(d_hist) else np.nan,
                "drv_track_hist": (same_track.loc[same_track["driver"] == r["driver"], "position"].mean()
                                   if (same_track["driver"] == r["driver"]).any() else np.nan),
            })
        race = pd.concat([race.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        out.append(race)
    feats = pd.concat(out, ignore_index=True)
    keep = ["year", "round", "event", "date", "driver", "team", "position", "status"] + FEATURES
    return feats[keep]
