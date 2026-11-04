"""Dashboard data layer, on the real data files (skipped if they are missing)."""

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "data/laps_clean.csv.gz").exists(), reason="data not built")

from f1agent import planner as P  # noqa: E402

NEXT = "next:Las Vegas Grand Prix"


def past_key():
    return next(k for _, k in P.race_choices() if k.startswith("past"))


def test_upcoming_races_come_first_and_use_upcoming_csv():
    keys = [k for _, k in P.race_choices()]
    n_next = len(P.store().upcoming)
    assert all(k.startswith("next:") for k in keys[:n_next])
    assert all(k.startswith("past:") for k in keys[n_next:])


def test_plan_respects_guards_and_overrides():
    inp, info, res = P.plan(NEXT, pit_loss=30.0, sc_lap_prob=0.0, n_sims=200)
    assert inp.pit_loss == 30.0 and inp.sc_lap_prob == 0.0
    assert all(v >= 0 for v in inp.rate.values())
    assert all(-P.MAX_OFFSET <= v <= 0 for v in inp.offset.values())
    assert res["median_gap_s"].is_monotonic_increasing
    assert info["upcoming"] and inp.laps == 50


def test_compound_override_changes_the_inputs():
    soft_range, _, _ = P.plan(NEXT, compounds=(3.0, 4.0, 5.0), n_sims=50)
    hard_range, _, _ = P.plan(NEXT, compounds=(1.0, 2.0, 3.0), n_sims=50)
    assert soft_range.rate != hard_range.rate


def test_strategy_text_round_trip():
    s = (("MEDIUM", 20), ("HARD", 30))
    assert P.parse_strategy(P.describe(s)) == s


def test_upcoming_grid_is_estimated_and_ordered_by_pace():
    drivers, base, real = P.field(NEXT)
    assert not real
    assert base == sorted(base) and base[0] == 0.0
    assert len(set(drivers)) == len(drivers)


def test_real_trace_has_a_leader_on_every_lap():
    trace = P.real_trace(int(past_key().split(":")[1]))
    assert np.allclose(trace.min(axis=1), 0.0)


def test_same_strategy_for_a_and_b_gives_identical_races():
    key = past_key()
    _, _, res = P.plan(key, n_sims=50)
    drv = P.field(key)[0][0]
    s = res.iloc[0]["strategy"]
    out = P.scenario(key, drv, s, s, n_sims=40)
    assert np.array_equal(out["a"]["positions"], out["b"]["positions"])


def test_track_map_exists_for_the_upcoming_races():
    for _, key in P.race_choices()[: len(P.store().upcoming)]:
        m = P.track_map(key)
        assert m is not None and len(m) > 100
        assert m["t"].is_monotonic_increasing and m["speed"].between(40, 380).all()


def test_on_track_puts_the_leader_at_the_line_and_wraps_lapped_cars():
    import pandas as pd
    m = P.track_map(NEXT)
    T = float(m["lap_duration"].iloc[0])
    pos = P.on_track(pd.Series({"AAA": 0.0, "BBB": T / 2, "CCC": T + T / 2, "DDD": float("nan")}), m)
    assert list(pos["driver"]) == ["AAA", "BBB", "CCC"]  # retired car dropped
    last = m.iloc[-1]
    assert (pos.iloc[0]["x"], pos.iloc[0]["y"]) == (last["x"], last["y"])  # leader at the end of the lap
    assert (pos.iloc[1]["x"], pos.iloc[1]["y"]) == (pos.iloc[2]["x"], pos.iloc[2]["y"])  # one lap down: same spot
