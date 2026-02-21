"""Download race and qualifying results with FastF1 into one table.

One row per driver per Grand Prix (sprints are not included), with what was known before
the start (grid slot, best qualifying time) and what happened (finishing order, status,
points). Only events whose date has passed are downloaded.

Usage:
  python -m f1agent.data --years 2023 2024 2025 2026 --out data/races.csv
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "year", "round", "event", "date", "driver", "team", "grid", "quali_time_s",
    "position", "status", "points", "laps",
]

# Status strings that mean the car took the chequered flag (on the lead lap or lapped).
FINISHED_PREFIXES = ("Finished", "Lapped", "+")


def finished(status: str) -> bool:
    return isinstance(status, str) and status.startswith(FINISHED_PREFIXES)


def _best_quali_time(row: pd.Series) -> float:
    times = [row.get(q) for q in ("Q1", "Q2", "Q3")]
    secs = [t.total_seconds() for t in times if isinstance(t, pd.Timedelta) and not pd.isna(t)]
    return min(secs) if secs else np.nan


def load_event(year: int, rnd: int) -> pd.DataFrame:
    """Results of one Grand Prix (race plus qualifying), in the table's columns."""
    import fastf1

    race = fastf1.get_session(year, rnd, "R")
    race.load(laps=False, telemetry=False, weather=False, messages=False)
    res = race.results.copy()

    quali_times: dict[str, float] = {}
    try:
        quali = fastf1.get_session(year, rnd, "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
        for _, r in quali.results.iterrows():
            quali_times[str(r["Abbreviation"])] = _best_quali_time(r)
    except Exception as exc:  # a missing qualifying session should not drop the race
        logging.warning("qualifying %s round %s unavailable: %s", year, rnd, exc)

    n = len(res)
    rows = []
    for _, r in res.iterrows():
        grid = r.get("GridPosition")
        # 0 means a pit-lane start: behind everyone on the grid.
        grid = float(n) if (pd.isna(grid) or grid == 0) else float(grid)
        pos = r.get("Position")
        code = r.get("Abbreviation")
        if not isinstance(code, str) or not code.strip() or code.lower() == "nan":
            # Some stand-in drivers have no code in the source; use the surname's first
            # three letters, as F1 does.
            code = str(r.get("LastName") or r.get("FullName") or r.get("DriverNumber"))[:3].upper()
        rows.append({
            "year": year,
            "round": rnd,
            "event": race.event["EventName"],
            "date": pd.Timestamp(race.event["EventDate"]).date().isoformat(),
            "driver": code,
            "team": r["TeamName"],
            "grid": grid,
            "quali_time_s": quali_times.get(str(r["Abbreviation"]), np.nan),
            "position": float(pos) if not pd.isna(pos) else np.nan,
            "status": r.get("Status"),
            "points": float(r.get("Points") or 0.0),
            "laps": float(r.get("Laps")) if not pd.isna(r.get("Laps")) else np.nan,
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    # Drivers without a classified position (disqualified, did not start) go to the back,
    # in their original order.
    missing = df["position"].isna()
    if missing.any():
        start = df["position"].max(skipna=True)
        start = 0 if pd.isna(start) else start
        df.loc[missing, "position"] = start + np.arange(1, missing.sum() + 1)
    return df


def completed_rounds(year: int, today: date | None = None) -> list[int]:
    import fastf1

    today = today or date.today()
    sched = fastf1.get_event_schedule(year, include_testing=False)
    done = sched[pd.to_datetime(sched["EventDate"]).dt.date < today]
    return [int(r) for r in done["RoundNumber"]]


def build(years: list[int], out: Path, cache: Path) -> pd.DataFrame:
    import fastf1

    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    logging.getLogger("fastf1").setLevel(logging.WARNING)

    existing = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=COLUMNS)
    have = set(zip(existing["year"], existing["round"]))
    frames = [existing] if len(existing) else []
    out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> pd.DataFrame:
        table = pd.concat(frames, ignore_index=True)
        table = table.sort_values(["year", "round", "position"]).reset_index(drop=True)
        table.to_csv(out, index=False)
        return table

    # The F1 data APIs allow about 500 calls per hour. The table is saved after every
    # event, so a run that hits the limit can simply be repeated later and resumes.
    try:
        for year in years:
            for rnd in completed_rounds(year):
                if (year, rnd) in have:
                    continue
                try:
                    df = load_event(year, rnd)
                except fastf1_rate_limit_error():
                    raise
                except Exception as exc:
                    print(f"  {year} round {rnd}: failed ({type(exc).__name__}: {exc})", flush=True)
                    continue
                frames.append(df)
                save()
                print(f"  {year} round {rnd:2d} {df['event'].iloc[0]}: {len(df)} drivers", flush=True)
    except fastf1_rate_limit_error() as exc:
        print(f"Stopped at the API rate limit ({exc}); run again later to continue.", flush=True)
    return save() if frames else existing


def fastf1_rate_limit_error():
    try:
        from fastf1.exceptions import RateLimitExceededError
        return RateLimitExceededError
    except ImportError:  # older FastF1
        return RuntimeError


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    p.add_argument("--out", default="data/races.csv")
    p.add_argument("--cache", default="cache")
    args = p.parse_args(argv)
    table = build(args.years, Path(args.out), Path(args.cache))
    races = table.groupby(["year", "round"]).ngroups
    print(f"{len(table)} rows, {races} races -> {args.out}")


if __name__ == "__main__":
    main()
