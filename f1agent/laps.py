"""Build the clean-lap table the tyre model learns from.

One row per lap with: driver, lap time, compound label (SOFT/MEDIUM/HARD) and the real
Pirelli compound (C1-C6) for that weekend, tyre age, stint, lap number (a proxy for fuel
load) and track temperature. Laps that say nothing about tyre wear are flagged and left
out of the fit: the first lap, pit in- and out-laps, laps under safety car, virtual safety
car or red flag, and laps far off the driver's own pace (traffic, mistakes, rain).

Usage:
  python -m f1agent.laps --openf1 data/openf1 --out data/laps_clean.csv.gz
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .openf1 import get

SC_START = re.compile(r"(?<!VIRTUAL )SAFETY CAR DEPLOYED|VIRTUAL SAFETY CAR DEPLOYED|RED FLAG", re.I)
SC_END = re.compile(r"SAFETY CAR IN THIS LAP|VIRTUAL SAFETY CAR ENDING|SAFETY CAR ENDING|RESUME", re.I)
LABELS = ["HARD", "MEDIUM", "SOFT"]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = s.replace("grand prix", "").replace("-", " ")
    return " ".join(s.split())


# Names that differ between Wikipedia (the Pirelli table) and OpenF1's meetings.
NAME_ALIASES = {"barcelona catalunya": "barcelona"}


def meetings(openf1_dir: Path, years: list[int]) -> pd.DataFrame:
    path = openf1_dir / "meetings.csv.gz"
    if path.exists():
        m = pd.read_csv(path)
        if set(years) <= set(m["year"]):
            return m
    m = pd.DataFrame([r for y in years for r in get("meetings", year=y)])
    m.to_csv(path, index=False, compression="gzip")
    return m


def attach_pirelli(sessions: pd.DataFrame, pirelli: pd.DataFrame) -> pd.DataFrame:
    """Add hard/medium/soft C-numbers to each session through its meeting name and year."""
    p = pirelli.copy()
    p["key"] = p["event"].map(_norm).map(lambda k: NAME_ALIASES.get(k, k))
    s = sessions.copy()
    s["key"] = s["meeting_name"].map(_norm).map(lambda k: NAME_ALIASES.get(k, k))
    out = s.merge(p[["year", "key", "hard", "medium", "soft"]], on=["year", "key"], how="left")
    return out.drop(columns="key")


def compound_number(row: pd.Series) -> float:
    label = str(row["compound"]).upper()
    if label not in LABELS or pd.isna(row.get(label.lower())):
        return np.nan
    return float(str(row[label.lower()]).lstrip("C"))


def sc_laps(rc: pd.DataFrame, max_lap: int) -> set[int]:
    """Laps run under safety car, virtual safety car or red flag in one session."""
    laps, start = set(), None
    for _, r in rc.sort_values("date").iterrows():
        msg, lap = str(r.get("message", "")), r.get("lap_number")
        if pd.isna(lap):
            continue
        lap = int(lap)
        if SC_START.search(msg) and start is None:
            start = lap
        elif SC_END.search(msg) and start is not None:
            laps.update(range(start, lap + 1))
            start = None
    if start is not None:
        laps.update(range(start, max_lap + 1))
    return laps


def build(openf1_dir: Path, pirelli_csv: Path) -> pd.DataFrame:
    sessions = pd.read_csv(openf1_dir / "sessions.csv.gz")
    laps = pd.read_csv(openf1_dir / "laps.csv.gz")
    stints = pd.read_csv(openf1_dir / "stints.csv.gz")
    pits = pd.read_csv(openf1_dir / "pit.csv.gz")
    weather = pd.read_csv(openf1_dir / "weather.csv.gz")
    rc = pd.read_csv(openf1_dir / "race_control.csv.gz")
    drivers = pd.read_csv(openf1_dir / "drivers.csv.gz")

    meet = meetings(openf1_dir, sorted(sessions["year"].unique().tolist()))
    sessions = sessions.merge(meet[["meeting_key", "meeting_name"]], on="meeting_key", how="left")
    sessions = attach_pirelli(sessions, pd.read_csv(pirelli_csv))

    out = []
    for _, s in sessions.iterrows():
        key = s["session_key"]
        L = laps[laps["session_key"] == key].copy()
        if L.empty:
            continue
        St = stints[stints["session_key"] == key]
        # Tyre compound and age for every lap, from the stint it belongs to.
        L["compound"], L["tyre_age"], L["stint"] = None, np.nan, np.nan
        for _, st in St.iterrows():
            m = (L["driver_number"] == st["driver_number"]) & L["lap_number"].between(st["lap_start"], st["lap_end"])
            L.loc[m, "compound"] = st["compound"]
            L.loc[m, "stint"] = st["stint_number"]
            L.loc[m, "tyre_age"] = st["tyre_age_at_start"] + (L.loc[m, "lap_number"] - st["lap_start"])
        P = pits[pits["session_key"] == key]
        in_laps = set(zip(P["driver_number"], P["lap_number"]))
        L["in_lap"] = [(d, n) in in_laps for d, n in zip(L["driver_number"], L["lap_number"])]
        neutral = sc_laps(rc[rc["session_key"] == key], int(L["lap_number"].max()))
        L["neutralised"] = L["lap_number"].isin(neutral)
        W = weather[weather["session_key"] == key].sort_values("date")
        if len(W):
            L = L.sort_values("date_start")
            L["date_dt"] = pd.to_datetime(L["date_start"], utc=True, format="mixed")
            W = W.assign(date_dt=pd.to_datetime(W["date"], utc=True, format="mixed"))
            L = pd.merge_asof(L.dropna(subset=["date_dt"]), W[["date_dt", "track_temperature", "rainfall"]],
                              on="date_dt", direction="nearest")
        else:
            L["track_temperature"], L["rainfall"] = np.nan, np.nan
        D = drivers[drivers["session_key"] == key][["driver_number", "name_acronym", "team_name"]].drop_duplicates("driver_number")
        L = L.merge(D, on="driver_number", how="left")
        for col in ["year", "session_name", "meeting_name", "circuit_short_name", "date_start", "hard", "medium", "soft"]:
            L["session_" + col if col == "date_start" else col] = s[col]
        out.append(L)
    df = pd.concat(out, ignore_index=True)
    df["compound_c"] = df.apply(compound_number, axis=1)

    # A lap is "clean" when it can say something about tyre wear.
    pace = df.groupby(["session_key", "driver_number"])["lap_duration"].transform("median")
    df["clean"] = (
        df["lap_duration"].notna()
        & (df["lap_number"] > 1)
        & ~df["is_pit_out_lap"].fillna(False).astype(bool)
        & ~df["in_lap"]
        & ~df["neutralised"]
        & (df["rainfall"].fillna(0) == 0)
        & df["compound"].isin(LABELS)
        & (df["lap_duration"] < 1.07 * pace)
    )
    keep = ["year", "session_key", "session_name", "meeting_name", "circuit_short_name", "session_date_start",
            "driver_number", "name_acronym", "team_name", "lap_number", "lap_duration", "compound",
            "compound_c", "hard", "medium", "soft", "tyre_age", "stint", "track_temperature",
            "in_lap", "is_pit_out_lap", "neutralised", "clean"]
    return df[keep].sort_values(["session_date_start", "driver_number", "lap_number"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--openf1", default="data/openf1")
    p.add_argument("--pirelli", default="data/pirelli_compounds.csv")
    p.add_argument("--out", default="data/laps_clean.csv.gz")
    args = p.parse_args(argv)
    df = build(Path(args.openf1), Path(args.pirelli))
    df.to_csv(args.out, index=False, compression="gzip")
    races = df[df["session_name"] == "Race"]
    print(f"{len(df)} laps, {df['session_key'].nunique()} sessions; races: {races['session_key'].nunique()}, "
          f"clean race laps: {int(races['clean'].sum())}, race laps without a C-number: "
          f"{int(races['compound_c'].isna().sum())}")


if __name__ == "__main__":
    main()
