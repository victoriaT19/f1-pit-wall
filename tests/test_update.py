"""Session update helpers (no network)."""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "data/laps_clean.csv.gz").exists(), reason="data not built")

from f1agent import planner as P  # noqa: E402
from f1agent.update import sessions_seen, set_compounds  # noqa: E402

NEXT = "next:Las Vegas Grand Prix"


def test_set_compounds_marks_the_nomination_official():
    up = pd.DataFrame([{"meeting_name": "Las Vegas Grand Prix", "hard": "C3", "medium": "C4", "soft": "C5",
                        "compounds_status": "provisional: 2025 nomination"}])
    out = set_compounds(up, ["Las Vegas Grand Prix=c2,c3,c4"])
    assert out.iloc[0][["hard", "medium", "soft", "compounds_status"]].tolist() == ["C2", "C3", "C4", "official"]
    with pytest.raises(SystemExit):
        set_compounds(up, ["Nowhere Grand Prix=C1,C2,C3"])


def test_sessions_seen_lists_practice_and_qualifying():
    laps = pd.DataFrame({"year": [2026, 2026, 2026], "meeting_name": ["X GP"] * 3,
                         "session_name": ["Practice 2", "Practice 1", "Practice 1"]})
    assert sessions_seen(laps, 2026, "X GP", True) == ["Practice 1", "Practice 2", "Qualifying"]
    assert sessions_seen(laps, 2026, "Y GP", False) == []


def test_real_qualifying_grid_replaces_the_estimate():
    s = P.store()
    drivers, _, real = P.field(NEXT)
    assert not real
    saved = s.grids
    try:
        order = list(reversed(drivers))  # a grid clearly different from the estimate
        s.grids = pd.DataFrame({"year": 2026, "meeting_name": "Las Vegas Grand Prix", "driver": order,
                                "grid": range(1, len(order) + 1),
                                "quali_time_s": [93.0 + 0.1 * i for i in range(len(order))]})
        got, base, real = P.field(NEXT)
        assert real and got == order and len(base) == len(order)
    finally:
        s.grids = saved
