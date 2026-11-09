"""One command to bring the dashboard up to date after every session.

  1. OpenF1: downloads any finished practice and race session not yet in data/openf1.
  2. Results: adds finished races to data/races.csv (FastF1).
  3. Qualifying: for upcoming races whose qualifying is over, saves the starting grid
     (OpenF1 `starting_grid`, penalties applied) to data/upcoming_grid.csv.
  4. Compounds: `--compounds` records Pirelli's nomination for an upcoming race (it is then
     marked official); a race that has been run is added to data/pirelli_compounds.csv when
     its nomination is official.
  5. Rebuilds data/laps_clean.csv.gz.
  6. Prediction log: for every upcoming race, appends the current plan to
     results/live/predictions.jsonl with the time and the sessions it had seen, so the
     forecasts made before each race can be compared with what happened.
  7. With --publish: runs the tests, commits data and log, pushes, and redeploys the Space
     (HF_TOKEN in the environment or in .env).

Usage:
  python -m f1agent.update
  python -m f1agent.update --compounds "Las Vegas Grand Prix=C3,C4,C5" --publish
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOG = ROOT / "results/live/predictions.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def qualifying_grid(year: int, circuit: str) -> pd.DataFrame:
    """Starting grid of the year's race at `circuit`, once its qualifying has finished."""
    from .openf1 import get

    sessions = [s for s in get("sessions", year=year, session_name="Qualifying")
                if s.get("circuit_short_name") == circuit and not s.get("is_cancelled")
                and s.get("date_end", "") < _now()]
    if not sessions:
        return pd.DataFrame()
    key = sessions[-1]["session_key"]
    grid = pd.DataFrame(get("starting_grid", session_key=key))
    if grid.empty:
        return grid
    names = {d["driver_number"]: d["name_acronym"] for d in get("drivers", session_key=key)}
    grid["driver"] = grid["driver_number"].map(names)
    return (grid.rename(columns={"position": "grid", "lap_duration": "quali_time_s"})
            [["driver", "grid", "quali_time_s"]].dropna(subset=["driver"]).sort_values("grid"))


def set_compounds(upcoming: pd.DataFrame, spec: list[str]) -> pd.DataFrame:
    """'Las Vegas Grand Prix=C3,C4,C5' (hard, medium, soft) -> official nomination."""
    for item in spec:
        name, comps = item.split("=", 1)
        hard, medium, soft = [c.strip().upper() for c in comps.split(",")]
        mask = upcoming["meeting_name"] == name.strip()
        if not mask.any():
            sys.exit(f"unknown upcoming race: {name!r}; known: {', '.join(upcoming['meeting_name'])}")
        upcoming.loc[mask, ["hard", "medium", "soft", "compounds_status"]] = [hard, medium, soft, "official"]
        print(f"compounds: {name.strip()} = {hard}/{medium}/{soft} (official)")
    return upcoming


def run_races_to_pirelli(upcoming: pd.DataFrame, races: pd.DataFrame) -> None:
    """Upcoming races that have now been run join the Pirelli table (official nominations only)."""
    path = DATA / "pirelli_compounds.csv"
    pir = pd.read_csv(path)
    have = set(zip(pir["year"], pir["event"]))
    run = set(zip(races["year"], races["event"]))
    rows = []
    for u in upcoming.itertuples():
        if (u.year, u.meeting_name) in run and (u.year, u.meeting_name) not in have:
            if u.compounds_status != "official":
                print(f"WARNING: {u.meeting_name} has been run but its compounds are still provisional; "
                      "pass --compounds to record the real nomination")
                continue
            rows.append({"year": u.year, "event": u.meeting_name, "hard": u.hard, "medium": u.medium,
                         "soft": u.soft, "source_url": u.source_url})
    if rows:
        pd.concat([pir, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
        print(f"pirelli: added {', '.join(r['event'] for r in rows)}")


def sessions_seen(laps: pd.DataFrame, year: int, meeting: str, has_grid: bool) -> list[str]:
    seen = sorted(laps.loc[(laps["year"] == year) & (laps["meeting_name"] == meeting), "session_name"].unique())
    return seen + (["Qualifying"] if has_grid else [])


def log_predictions() -> list[dict]:
    """Append the current plan of every upcoming race to the prediction log."""
    from . import planner as P

    P.store.cache_clear()
    P._inputs.cache_clear()
    s = P.store()
    entries = []
    for label, key in P.race_choices():
        if not key.startswith("next:"):
            continue
        row, n_laps, info = P.race_row(key)
        inp, extra, res = P.plan(key)
        drivers, base, real_grid = P.field(key)
        best = res.iloc[0]
        near = res[(res["stops"] == best["stops"]) & (res["median_gap_s"] <= best["median_gap_s"] + 2.0)]
        first = sorted({p[0] for p in near["pit_laps"]})
        entries.append({
            "logged_at": _now(), "year": int(row["year"]), "race": row["meeting_name"],
            "sessions_seen": sessions_seen(s.laps, int(row["year"]), row["meeting_name"], real_grid),
            "compounds": {"hard": row["hard"], "medium": row["medium"], "soft": row["soft"],
                          "status": info["compounds_status"]},
            "track_temp": extra["track_temp"], "pit_loss": round(inp.pit_loss, 1),
            "recommended": best["strategy"], "stops": int(best["stops"]),
            "first_stop_window": [int(first[0]), int(first[-1])],
            "top5": res["strategy"].head(5).tolist(),
            "grid": "real" if real_grid else "estimated", "front_row": drivers[:2],
        })
    LOG.parent.mkdir(parents=True, exist_ok=True)
    # skip a forecast identical to the race's previous one (nothing new since the last update)
    previous = {}
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                previous[(r["year"], r["race"])] = {k: v for k, v in r.items() if k != "logged_at"}
    same = lambda e: previous.get((e["year"], e["race"])) == {k: v for k, v in e.items() if k != "logged_at"}
    with LOG.open("a", encoding="utf-8") as f:
        for e in entries:
            if not same(e):
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    for e in entries:
        print(f"{'unchanged' if same(e) else 'logged'}: {e['race']} · seen "
              f"{', '.join(e['sessions_seen']) or 'nothing yet'} · {e['recommended']} · grid {e['grid']}")
    return entries


def load_env(path: Path) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def publish(message: str) -> None:
    def run(*cmd):
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)

    run(sys.executable, "-m", "pytest", "-q", "-p", "no:warnings")
    run("git", "add", "data", "results/live")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        print("nothing new to commit")
    else:
        run("git", "commit", "-q", "-m", message)
        run("git", "push", "-q", "origin", "HEAD:main")
    load_env(ROOT / ".env")
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN not set: skipping the Space deploy")
        return
    run(sys.executable, "scripts/deploy_space.py", "--repo", os.environ.get("SPACE_REPO", "vtaua/f1-pit-wall"))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, default=datetime.now().year)
    p.add_argument("--compounds", nargs="*", default=[], metavar="RACE=HARD,MEDIUM,SOFT")
    p.add_argument("--skip-download", action="store_true", help="only rebuild and log from local data")
    p.add_argument("--publish", action="store_true", help="test, commit, push and redeploy the Space")
    args = p.parse_args(argv)

    from . import data as results
    from . import laps as lapsmod
    from . import openf1

    upcoming = pd.read_csv(DATA / "upcoming.csv")
    if not args.skip_download:
        print("== OpenF1 sessions", flush=True)
        openf1.download([args.year], DATA / "openf1", include_practice=True)
        print("== race results", flush=True)
        results.build([args.year], DATA / "races.csv", ROOT / "cache")
    races = pd.read_csv(DATA / "races.csv")

    if args.compounds:
        upcoming = set_compounds(upcoming, args.compounds)
        upcoming.to_csv(DATA / "upcoming.csv", index=False)

    if not args.skip_download:
        print("== qualifying grids", flush=True)
        run = set(zip(races["year"], races["event"]))
        grids = []
        for u in upcoming.itertuples():
            if (u.year, u.meeting_name) in run:
                continue
            g = qualifying_grid(int(u.year), u.circuit)
            if len(g):
                g.insert(0, "meeting_name", u.meeting_name)
                g.insert(0, "year", int(u.year))
                grids.append(g)
                print(f"  {u.meeting_name}: grid saved (pole {g['driver'].iloc[0]})")
            else:
                print(f"  {u.meeting_name}: qualifying not run yet")
        path = DATA / "upcoming_grid.csv"
        if grids:
            pd.concat(grids, ignore_index=True).to_csv(path, index=False)
        elif path.exists():
            path.unlink()  # every upcoming race has been run, or none has qualified yet

    run_races_to_pirelli(upcoming, races)
    print("== laps", flush=True)
    df = lapsmod.build(DATA / "openf1", DATA / "pirelli_compounds.csv")
    path = DATA / "laps_clean.csv.gz"
    import gzip

    text = df.to_csv(index=False)
    old = gzip.open(path, "rt", encoding="utf-8", newline="").read() if path.exists() else None
    if old == text:
        print(f"  unchanged ({len(df)} laps)")
    else:  # fixed gzip timestamp: the same data always gives the same file
        with gzip.GzipFile(path, "wb", mtime=0) as f:
            f.write(text.encode("utf-8"))
        print(f"  {len(df)} laps, latest session {df['session_date_start'].max()[:10]}")

    print("== prediction log", flush=True)
    entries = log_predictions()

    if args.publish:
        latest = entries[0]["sessions_seen"][-1] if entries and entries[0]["sessions_seen"] else "dados"
        race = entries[0]["race"] if entries else ""
        publish(f"Atualiza dados: {race} {latest}".strip())


if __name__ == "__main__":
    main()
