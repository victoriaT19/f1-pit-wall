"""Download race data from OpenF1 (https://openf1.org), free historical API since 2023.

OpenF1 is a community project, not affiliated with Formula 1; its data is meant for
non-commercial analysis and fan projects. The free tier allows 30 requests per minute,
so calls are spaced and retried on HTTP 429.

For every race session (and, with --practice, the practice sessions) it saves the
endpoints the strategy model needs, one gzip CSV per endpoint under data/openf1/:
  laps, stints, pit, weather, race_control, drivers, sessions

Usage:
  python -m f1agent.openf1 --years 2024 2025 2026
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

API = "https://api.openf1.org/v1/"
ENDPOINTS = ["laps", "stints", "pit", "weather", "race_control", "drivers"]
MIN_INTERVAL = 2.2  # seconds between calls: stays under 30 per minute

_last_call = 0.0


def get(endpoint: str, raw: str = "", **params) -> list[dict]:
    """GET one endpoint, spaced and retried. Returns a list of records.

    `raw` is appended to the query as is, for OpenF1's comparison filters such as
    "date>=2024-03-02T15:03:42&date<=2024-03-02T15:05:20" (urlencode would escape them).
    """
    global _last_call
    query = "&".join(x for x in (urllib.parse.urlencode(params), raw) if x)
    url = API + endpoint + ("?" + query if query else "")
    delay = 10.0
    for attempt in range(6):
        wait = MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={"User-Agent": "f1agent (fan project)"}), timeout=60) as r:
                data = json.load(r)
            return data if isinstance(data, list) else []
        except urllib.error.HTTPError as e:
            if e.code == 404:  # OpenF1 answers 404 when a filter matches nothing
                return []
            if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
        time.sleep(delay)
        delay = min(delay * 2, 120)
    return []


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    """Write to a temporary file and rename, so an interrupted run never leaves a broken file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)


def race_sessions(year: int, include_practice: bool = False) -> list[dict]:
    """Finished sessions of a year: races (not sprints), plus practice if asked."""
    now = datetime.now(timezone.utc).isoformat()
    names = ["Race"] + (["Practice 1", "Practice 2", "Practice 3"] if include_practice else [])
    out = []
    for name in names:
        for s in get("sessions", year=year, session_name=name):
            if not s.get("is_cancelled") and s.get("date_end", "") < now:
                out.append(s)
    return sorted(out, key=lambda s: s["date_start"])


def download(years: list[int], out: Path, include_practice: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sessions_path = out / "sessions.csv.gz"
    known = pd.read_csv(sessions_path) if sessions_path.exists() else pd.DataFrame()
    done = set(known["session_key"]) if len(known) else set()
    # Each new session goes to its own small files under parts/; the combined tables are
    # rebuilt once at the end (rewriting them after every session made runs slow).
    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    new_sessions = []
    for year in years:
        for s in race_sessions(year, include_practice):
            key = s["session_key"]
            if key in done:
                continue
            counts = {}
            for e in ENDPOINTS:
                rows = get(e, session_key=key)
                counts[e] = len(rows)
                if rows:
                    _atomic_csv(pd.DataFrame(rows), parts / f"{key}_{e}.csv.gz")
            new_sessions.append(s)
            _atomic_csv(pd.concat([known, pd.DataFrame(new_sessions)], ignore_index=True), sessions_path)
            print(f"  {year} {s['session_name']:<10} {s['circuit_short_name']:<20} "
                  f"laps={counts['laps']} stints={counts['stints']} pit={counts['pit']}", flush=True)
    combine(out)


def combine(out: Path) -> None:
    """Merge parts/ into the combined tables and remove the parts."""
    parts = out / "parts"
    if not parts.exists():
        return
    for e in ENDPOINTS:
        files = sorted(parts.glob(f"*_{e}.csv.gz"))
        if not files:
            continue
        frames = [pd.read_csv(out / f"{e}.csv.gz")] if (out / f"{e}.csv.gz").exists() else []
        frames += [pd.read_csv(f) for f in files]
        _atomic_csv(pd.concat(frames, ignore_index=True), out / f"{e}.csv.gz")
        for f in files:
            f.unlink()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    p.add_argument("--out", default="data/openf1")
    p.add_argument("--practice", action="store_true", help="also download practice sessions")
    args = p.parse_args(argv)
    download(args.years, Path(args.out), args.practice)


if __name__ == "__main__":
    main()
