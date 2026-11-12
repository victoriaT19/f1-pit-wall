"""Race pace of each driver, measured in earlier races.

Qualifying shows one-lap pace; races are about long-run pace on heavy fuel, which ranks
cars differently. For every race, each driver's pace is the median, over his clean laps,
of (fuel-corrected lap time minus the field's median fuel-corrected time on the same lap).
Comparing within the same lap cancels fuel load and track evolution, which every car shares.

Before a race, a driver's expected race pace is the mean of his last RACES_BACK values in
the same season; with none (first race, or a new driver), his qualifying gap is used.

Usage in code: `pace_table(laps)` then `pre_race_pace(table, year, date, drivers, quali_gaps)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .tyres import FUEL_S_PER_LAP, corrected

RACES_BACK = 3
QUALI_WEIGHT = 0.3  # weight of the qualifying gap when race history exists


def pace_table(laps: pd.DataFrame, fuel: float = FUEL_S_PER_LAP) -> pd.DataFrame:
    """One row per driver per race: pace relative to the field (s/lap, lower = faster)."""
    r = laps[(laps["session_name"] == "Race") & laps["clean"]].copy()
    r["y"] = corrected(r, fuel)
    r["rel"] = r["y"] - r.groupby(["session_key", "lap_number"])["y"].transform("median")
    out = (r.groupby(["session_key", "year", "session_date_start", "meeting_name", "name_acronym"])["rel"]
           .agg(["median", "size"]).reset_index()
           .rename(columns={"name_acronym": "driver", "median": "pace", "size": "laps",
                            "session_date_start": "date"}))
    out = out[out["laps"] >= 10]
    # Relative to the fastest driver of that race, like a gap to pole.
    out["pace"] = out["pace"] - out.groupby("session_key")["pace"].transform("min")
    return out


def pre_race_pace(table: pd.DataFrame, year: int, date: str, drivers: list[str],
                  quali_gaps: list[float]) -> list[float]:
    """Expected race pace (s/lap relative to the fastest) for each driver, before `date`."""
    past = table[(table["year"] == year) & (table["date"] < date)].sort_values("date")
    out = []
    for d, q in zip(drivers, quali_gaps):
        h = past[past["driver"] == d].tail(RACES_BACK)
        if len(h):
            out.append(float((1 - QUALI_WEIGHT) * h["pace"].mean() + QUALI_WEIGHT * q))
        else:
            out.append(float(q))
    m = min(out)
    return [x - m for x in out]


def pace_pct_table(laps: pd.DataFrame, fuel: float = FUEL_S_PER_LAP) -> pd.DataFrame:
    """Like pace_table, but in % of the field's median lap, so circuits of different lengths
    are comparable and races from earlier seasons can be pooled. One row per driver per race."""
    r = laps[(laps["session_name"] == "Race") & laps["clean"]].copy()
    r["y"] = corrected(r, fuel)
    med = r.groupby(["session_key", "lap_number"])["y"].transform("median")
    r["rel"] = (r["y"] - med) / med * 100.0
    out = (r.groupby(["session_key", "session_date_start", "name_acronym"])["rel"]
           .agg(["median", "size"]).reset_index()
           .rename(columns={"name_acronym": "driver", "median": "pace_pct", "size": "laps",
                            "session_date_start": "date"}))
    out = out[out["laps"] >= 10]
    out["date"] = out["date"].str[:10]
    out["pace_pct"] = out["pace_pct"] - out.groupby("session_key")["pace_pct"].transform("min")
    return out[["date", "driver", "pace_pct"]]


def pre_race_pace_decayed(pct_table: pd.DataFrame, date: str, drivers: list[str], quali_gaps: list[float],
                          lap_s: float, half_life_days: float = 90.0) -> list[float]:
    """Like pre_race_pace, but the race history pools earlier seasons too (recent races weigh
    more, see decayed_pace) and is converted from % to s/lap with this race's lap time."""
    pct = decayed_pace(pct_table, date[:10], drivers, half_life_days)
    out = [float(q) if np.isnan(p) else float((1 - QUALI_WEIGHT) * p * lap_s / 100.0 + QUALI_WEIGHT * q)
           for p, q in zip(pct, quali_gaps)]
    m = min(out)
    return [x - m for x in out]


def decayed_pace(table: pd.DataFrame, date: str, drivers, half_life_days: float = 90.0) -> list[float]:
    """Each driver's race pace (% off the fastest) before `date`, recent races weighing more;
    NaN for a driver with no earlier race laps."""
    past = table[table["date"] < date]
    age = (pd.Timestamp(date) - pd.to_datetime(past["date"])).dt.days.to_numpy()
    w = pd.Series(0.5 ** (age / half_life_days), index=past.index)
    num = (past["pace_pct"] * w).groupby(past["driver"]).sum()
    den = w.groupby(past["driver"]).sum()
    pace = num / den
    return [float(pace[d]) if d in pace.index else np.nan for d in drivers]
