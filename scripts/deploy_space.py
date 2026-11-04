"""Publish the dashboard as a Hugging Face Space (free CPU hardware).

Uploads app.py, the f1agent package, the data files the dashboard reads and a Space
README. Nothing secret is uploaded (.env stays local).

Usage (HF_TOKEN with write access in the environment):
  python scripts/deploy_space.py --repo <user>/f1-pit-wall [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "app.py",
    "docs/methodology.md",
    "docs/test-log.md",
    "data/laps_clean.csv.gz",
    "data/races.csv",
    "data/upcoming.csv",
    "data/pirelli_compounds.csv",
    "data/track_maps.csv.gz",
    "data/openf1/drivers.csv.gz",
    "results/backtest/per_race.csv",
    "results/racesim/summary.json",
]

SPACE_README = """---
title: Pit wall — F1 race strategy
emoji: 🏁
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
license: mit
short_description: F1 pit strategy and race simulation from public timing data
---

# Pit wall

Race strategy for the rest of the 2026 Formula 1 season, built from public timing data
([OpenF1](https://openf1.org)) and Pirelli's compound nominations.

- **Strategy**: tyre degradation modelled from earlier races, Monte Carlo comparison of
  one- and two-stop strategies, track map coloured by speed.
- **Race simulation**: the whole field lap by lap, with traffic, overtaking, safety cars
  and the undercut; finishing-position distribution for two strategies and an on-track
  replay.
- **Scoreboard**: every backtest on the 2026 races, including where the model is no
  better than simple baselines.

Code, data pipeline, backtests and methodology:
[github.com/victoriaT19/f1_pre-abu_dhabi](https://github.com/victoriaT19/f1_pre-abu_dhabi).

Unofficial fan project, not associated with Formula 1 or Pirelli. Data from OpenF1
(non-commercial use).
"""

SPACE_REQUIREMENTS = """numpy>=1.26
pandas>=2.2
scipy>=1.12
scikit-learn>=1.4
plotly>=5.20
"""


def build(out: Path) -> None:
    for f in FILES:
        dst = out / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f, dst)
    shutil.copytree(ROOT / "f1agent", out / "f1agent", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (out / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
    import gradio

    (out / "README.md").write_text(SPACE_README.format(gradio_version=gradio.__version__), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True)
    p.add_argument("--dry-run", metavar="DIR", help="only write the Space folder to DIR")
    args = p.parse_args()

    if args.dry_run:
        out = Path(args.dry_run)
        out.mkdir(parents=True, exist_ok=True)
        build(out)
        print(f"Space folder written to {out}")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    # Creating a Space through the API can be refused for free accounts (HTTP 402) while the
    # website allows it: create it on the website first in that case.
    if not api.repo_exists(args.repo, repo_type="space"):
        api.create_repo(args.repo, repo_type="space", space_sdk="gradio")
    with tempfile.TemporaryDirectory() as tmp:
        build(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=args.repo, repo_type="space",
                          commit_message="Deploy the Pit wall dashboard")
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
