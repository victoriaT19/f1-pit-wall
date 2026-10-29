import numpy as np
import pytest

from f1agent.racesim import Car, Track, calibrate_threshold, position_changes, simulate


def track(**kw):
    base = dict(laps=50, rate={"SOFT": 0.10, "MEDIUM": 0.07, "HARD": 0.04},
                rate_sd={c: 0.0 for c in ("SOFT", "MEDIUM", "HARD")},
                offset={"SOFT": -0.6, "MEDIUM": -0.3, "HARD": 0.0},
                offset_sd={c: 0.0 for c in ("SOFT", "MEDIUM", "HARD")},
                pit_loss=20.0, sc_lap_prob=0.0, pass_threshold=1.0)
    base.update(kw)
    return Track(**base)


ONE_STOP = (("MEDIUM", 25), ("HARD", 25))


def test_identical_cars_keep_grid_order_when_passing_is_hard():
    cars = [Car(f"C{i}", 0.0, ONE_STOP) for i in range(10)]
    pos = simulate(cars, track(pass_threshold=3.0), n_sims=50)["positions"]
    assert position_changes(pos) < 0.5


def test_faster_car_moves_forward_and_easier_track_means_more_passes():
    cars = [Car(f"C{i}", 0.0, ONE_STOP) for i in range(9)] + [Car("FAST", -1.0, ONE_STOP)]
    hard = simulate(cars, track(pass_threshold=2.0), n_sims=100)["positions"][:, -1].mean()
    easy = simulate(cars, track(pass_threshold=0.0), n_sims=100)["positions"][:, -1].mean()
    assert easy < hard < 10  # starts last; moves up, and more where passing is easy


def test_undercut_beats_a_rival_on_old_tyres():
    """Two equal cars nose to tail; the one behind stops earlier and should come out ahead."""
    t = track(laps=40, rate={"SOFT": 0.25, "MEDIUM": 0.20, "HARD": 0.15}, pit_loss=20.0,
              pass_threshold=3.0)  # no overtaking on track: only the pit stop can change order
    leader = Car("LEAD", 0.0, (("MEDIUM", 22), ("HARD", 18)))
    chaser_late = Car("CHASE", 0.0, (("MEDIUM", 22), ("HARD", 18)))
    chaser_early = Car("CHASE", 0.0, (("MEDIUM", 16), ("HARD", 24)))
    same = simulate([leader, chaser_late], t, n_sims=200)["positions"][:, 1].mean()
    under = simulate([leader, chaser_early], t, n_sims=200)["positions"][:, 1].mean()
    assert same > 1.9  # same strategy, no passing: stays second
    assert under < 1.5  # the earlier stop gets it ahead most of the time


def test_pit_stop_costs_positions_without_degradation():
    t = track(rate={c: 0.0 for c in ("SOFT", "MEDIUM", "HARD")}, offset={c: 0.0 for c in ("SOFT", "MEDIUM", "HARD")})
    cars = [Car(f"C{i}", 0.0, ONE_STOP) for i in range(4)] + [Car("TWO", 0.0, (("MEDIUM", 15), ("HARD", 15), ("MEDIUM", 20)))]
    two_stop_pos = simulate(cars, t, n_sims=100)["positions"][:, -1].mean()
    assert two_stop_pos == pytest.approx(5.0, abs=0.2)


def test_safety_car_bunches_the_field():
    cars = [Car("FAST", -1.5, ONE_STOP)] + [Car(f"C{i}", 0.0, ONE_STOP) for i in range(5)]
    calm = simulate(cars, track(), n_sims=200, return_times=True)["times"]
    busy = simulate(cars, track(sc_lap_prob=0.03), n_sims=200, return_times=True)["times"]
    gap = lambda T: np.median(np.sort(T, axis=1)[:, 1] - np.sort(T, axis=1)[:, 0])
    assert gap(busy) < gap(calm)


def test_calibration_picks_lower_threshold_for_more_changes():
    rng = np.random.default_rng(0)
    cars = [Car(f"C{i}", float(b), ONE_STOP) for i, b in enumerate(rng.normal(0, 0.4, 12))]
    t = track()
    few = calibrate_threshold(cars, t, target_changes=0.5, n_sims=40)
    many = calibrate_threshold(cars, t, target_changes=3.0, n_sims=40)
    assert many < few


def test_form_and_start_spread_increase_position_changes():
    cars = [Car(f"C{i}", 0.1 * i, ONE_STOP) for i in range(10)]
    calm = position_changes(simulate(cars, track(pass_threshold=1.0), n_sims=100)["positions"])
    lively = position_changes(simulate(cars, track(pass_threshold=1.0, form_sd=0.4, start_sd=1.0), n_sims=100)["positions"])
    assert lively > calm + 0.3


def test_pre_race_pace_uses_only_earlier_races_of_the_season():
    import pandas as pd
    from f1agent.racepace import QUALI_WEIGHT, pre_race_pace

    table = pd.DataFrame([
        {"year": 2026, "date": "2026-03-01", "driver": "AAA", "pace": 0.0},
        {"year": 2026, "date": "2026-03-01", "driver": "BBB", "pace": 1.0},
        {"year": 2026, "date": "2026-05-01", "driver": "AAA", "pace": 9.9},   # after the race
        {"year": 2025, "date": "2025-11-01", "driver": "BBB", "pace": 0.0},   # other season
    ])
    out = pre_race_pace(table, 2026, "2026-04-01", ["AAA", "BBB", "CCC"], [0.5, 0.2, 0.8])
    a = (1 - QUALI_WEIGHT) * 0.0 + QUALI_WEIGHT * 0.5
    b = (1 - QUALI_WEIGHT) * 1.0 + QUALI_WEIGHT * 0.2
    c = 0.8  # no race history: qualifying gap
    m = min(a, b, c)
    assert out == pytest.approx([a - m, b - m, c - m])
