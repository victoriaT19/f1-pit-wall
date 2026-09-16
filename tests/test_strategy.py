import numpy as np
import pytest

from f1agent.strategy import RaceInputs, pit_laps, simulate, stint_time, strategies


def inputs(**kw):
    base = dict(laps=50, rate={"SOFT": 0.10, "MEDIUM": 0.06, "HARD": 0.03},
                rate_sd={c: 0.0 for c in ("SOFT", "MEDIUM", "HARD")},
                offset={"SOFT": -0.6, "MEDIUM": -0.3, "HARD": 0.0},
                offset_sd={c: 0.0 for c in ("SOFT", "MEDIUM", "HARD")},
                pit_loss=22.0, sc_lap_prob=0.0)
    base.update(kw)
    return RaceInputs(**base)


def test_strategies_follow_the_rules():
    s = strategies(50, max_stops=2, step=2, min_stint=8)
    assert s and all(sum(n for _, n in x) == 50 for x in s)
    assert all(len({c for c, _ in x}) >= 2 for x in s)  # two dry compounds
    assert all(min(n for _, n in x) >= 8 for x in s)
    assert {len(x) - 1 for x in s} == {1, 2}


def test_stint_time_closed_form():
    assert stint_time(3, offset=-0.5, rate=0.1) == pytest.approx(3 * -0.5 + 0.1 * (1 + 2 + 3))


def test_simulation_matches_deterministic_arithmetic():
    inp = inputs()
    s = [(("MEDIUM", 25), ("HARD", 25)), (("SOFT", 15), ("MEDIUM", 20), ("HARD", 15))]
    res = simulate(inp, s, n_sims=10).set_index("strategy")
    one = stint_time(25, -0.3, 0.06) + stint_time(25, 0, 0.03) + 22
    two = stint_time(15, -0.6, 0.10) + stint_time(20, -0.3, 0.06) + stint_time(15, 0, 0.03) + 44
    gap = abs(two - one)
    assert res["median_gap_s"].max() == pytest.approx(gap, abs=1e-9)


def test_high_degradation_favours_more_stops():
    s = strategies(50, max_stops=2, step=2)
    low = simulate(inputs(), s, n_sims=200).iloc[0]
    high = simulate(inputs(rate={"SOFT": 0.40, "MEDIUM": 0.30, "HARD": 0.22}), s, n_sims=200).iloc[0]
    assert low["stops"] == 1 and high["stops"] == 2


def test_cheap_pit_stops_favour_more_stops():
    s = strategies(50, max_stops=2, step=2)
    expensive = simulate(inputs(rate={"SOFT": 0.2, "MEDIUM": 0.14, "HARD": 0.1}, pit_loss=35), s, n_sims=200).iloc[0]
    cheap = simulate(inputs(rate={"SOFT": 0.2, "MEDIUM": 0.14, "HARD": 0.1}, pit_loss=10), s, n_sims=200).iloc[0]
    assert expensive["stops"] <= cheap["stops"]


def test_safety_car_makes_stops_cheaper_on_average():
    s = [(("MEDIUM", 25), ("HARD", 25)), (("SOFT", 15), ("MEDIUM", 20), ("HARD", 15))]
    calm = simulate(inputs(), s, n_sims=2000, seed=1).set_index("strategy")
    busy = simulate(inputs(sc_lap_prob=0.05), s, n_sims=2000, seed=1).set_index("strategy")
    two = "S15 → M20 → H15"
    assert busy.loc[two, "median_gap_s"] <= calm.loc[two, "median_gap_s"] + 1e-9


def test_uncertainty_spreads_the_gaps():
    s = strategies(50, max_stops=1, step=5)
    sure = simulate(inputs(), s, n_sims=500).iloc[1]
    unsure = simulate(inputs(rate_sd={"SOFT": 0.03, "MEDIUM": 0.03, "HARD": 0.03}), s, n_sims=500).iloc[1]
    assert (unsure["p90_gap_s"] - unsure["p10_gap_s"]) > (sure["p90_gap_s"] - sure["p10_gap_s"])


def test_pit_laps():
    assert pit_laps((("MEDIUM", 20), ("HARD", 30))) == [20]
    assert pit_laps((("SOFT", 12), ("MEDIUM", 18), ("HARD", 20))) == [12, 30]
