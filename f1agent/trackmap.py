"""Track maps from OpenF1 telemetry: one fast race lap per circuit, with speed.

For each circuit, the latest race in the data is taken and, in it, the fastest clean lap.
OpenF1's `location` endpoint gives the car's x, y position (about 4 samples per second) and
`car_data` its speed; both are fetched for the lap's time window and joined on time. The
result is the track outline with the speed at each point, the time since the start of the
lap (used to place cars on the map from their gap to the leader) and the sector boundaries
(from the lap's sector times).

Usage:
  python -m f1agent.trackmap --out data/track_maps.csv.gz
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .openf1 import get


def fastest_laps(laps_clean: pd.DataFrame, openf1_laps: pd.DataFrame) -> pd.DataFrame:
    """Per circuit: the fastest clean lap of its latest race, with the lap's start time."""
    r = laps_clean[(laps_clean["session_name"] == "Race") & laps_clean["clean"]]
    latest = r.sort_values("session_date_start").groupby("circuit_short_name")["session_key"].last()
    rows = []
    for circuit, sk in latest.items():
        best = r[r["session_key"] == sk].nsmallest(1, "lap_duration").iloc[0]
        o = openf1_laps[(openf1_laps["session_key"] == sk) & (openf1_laps["driver_number"] == best["driver_number"])
                        & (openf1_laps["lap_number"] == best["lap_number"])]
        if o.empty or pd.isna(o["date_start"].iloc[0]):
            continue
        o = o.iloc[0]
        rows.append({"circuit": circuit, "session_key": int(sk), "driver_number": int(best["driver_number"]),
                     "driver": best["name_acronym"], "lap_number": int(best["lap_number"]),
                     "date_start": o["date_start"], "lap_duration": float(best["lap_duration"]),
                     "s1": o["duration_sector_1"], "s2": o["duration_sector_2"]})
    return pd.DataFrame(rows)


def lap_trace(lap: pd.Series) -> pd.DataFrame:
    """x, y, speed and time since lap start for one lap."""
    start = pd.Timestamp(lap["date_start"])
    end = start + timedelta(seconds=float(lap["lap_duration"]))
    fmt = lambda t: t.strftime("%Y-%m-%dT%H:%M:%S.%f")
    raw = f"date>={fmt(start)}&date<={fmt(end)}"
    loc = pd.DataFrame(get("location", session_key=lap["session_key"], driver_number=lap["driver_number"], raw=raw))
    car = pd.DataFrame(get("car_data", session_key=lap["session_key"], driver_number=lap["driver_number"], raw=raw))
    if loc.empty:
        return pd.DataFrame()
    loc["date"] = pd.to_datetime(loc["date"], format="ISO8601")
    loc = loc.sort_values("date")
    loc = loc[(loc["x"] != 0) | (loc["y"] != 0)]  # (0, 0) marks a missing sample
    if not car.empty:
        car["date"] = pd.to_datetime(car["date"], format="ISO8601")
        loc = pd.merge_asof(loc, car.sort_values("date")[["date", "speed"]], on="date", direction="nearest")
    else:
        loc["speed"] = np.nan
    loc["t"] = (loc["date"] - start).dt.total_seconds()
    return loc[["t", "x", "y", "speed"]].reset_index(drop=True)


def build(laps_clean: pd.DataFrame, openf1_laps: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, lap in fastest_laps(laps_clean, openf1_laps).iterrows():
        tr = lap_trace(lap)
        if len(tr) < 50:
            print(f"  {lap['circuit']}: no telemetry", flush=True)
            continue
        for k in ("circuit", "session_key", "driver", "lap_duration", "s1", "s2"):
            tr[k] = lap[k]
        out.append(tr)
        print(f"  {lap['circuit']}: {len(tr)} points ({lap['driver']}, {lap['lap_duration']:.3f} s)", flush=True)
    return pd.concat(out, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--laps", default="data/laps_clean.csv.gz")
    p.add_argument("--openf1-laps", default="data/openf1/laps.csv.gz")
    p.add_argument("--out", default="data/track_maps.csv.gz")
    args = p.parse_args(argv)
    maps = build(pd.read_csv(args.laps), pd.read_csv(args.openf1_laps))
    tmp = Path(args.out).with_suffix(".tmp")
    maps.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(args.out)
    print(f"{maps['circuit'].nunique()} circuits -> {args.out}")


if __name__ == "__main__":
    main()
