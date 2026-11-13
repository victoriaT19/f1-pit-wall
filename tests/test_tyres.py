import numpy as np
import pandas as pd
import pytest

from f1agent.laps import attach_pirelli, compound_number, sc_laps
from f1agent.tyres import FUEL_S_PER_LAP, corrected, measure, stint_errors, within_slope


def synthetic_race(rates={"SOFT": 0.12, "HARD": 0.05}, fuel=FUEL_S_PER_LAP, noise=0.0, seed=0, key=1, date="2025-01-01"):
    """Stints whose fuel-corrected lap time grows exactly at `rates`."""
    rng = np.random.default_rng(seed)
    rows = []
    for drv in range(1, 9):
        base = 90 + drv * 0.2
        lap = 2
        for stint, comp in enumerate(["SOFT", "HARD"], start=1):
            for age in range(1, 16):
                t = base + (0.5 if comp == "SOFT" else 0) * -1 + rates[comp] * age - fuel * (lap - 1)
                rows.append({"session_key": key, "session_name": "Race", "year": 2025, "meeting_name": "Test GP",
                             "circuit_short_name": "Test", "session_date_start": date, "driver_number": drv,
                             "stint": stint, "compound": comp, "compound_c": 4.0 if comp == "SOFT" else 2.0,
                             "tyre_age": age, "lap_number": lap, "lap_duration": t + rng.normal(0, noise),
                             "track_temperature": 35.0, "clean": True})
                lap += 1
    return pd.DataFrame(rows)


def test_measure_recovers_known_rates():
    rates = measure(synthetic_race()).set_index("compound")["rate"]
    assert rates["SOFT"] == pytest.approx(0.12, abs=1e-9)
    assert rates["HARD"] == pytest.approx(0.05, abs=1e-9)


def test_measure_with_noise_is_close():
    rates = measure(synthetic_race(noise=0.3, seed=1)).set_index("compound")["rate"]
    assert rates["SOFT"] == pytest.approx(0.12, abs=0.02)


def test_wrong_fuel_assumption_shifts_rates_equally():
    df = synthetic_race(fuel=0.055)
    r = measure(df, fuel=0.035).set_index("compound")["rate"]
    # assuming 0.02 s/lap less fuel effect lowers every measured rate by 0.02
    assert r["SOFT"] == pytest.approx(0.10, abs=1e-9) and r["HARD"] == pytest.approx(0.03, abs=1e-9)


def test_stint_errors_zero_with_true_rate_and_positive_without():
    df = synthetic_race()
    df["stint_id"] = df["driver_number"].astype(str) + "_" + df["stint"].astype(str)
    assert max(stint_errors(df, {"SOFT": 0.12, "HARD": 0.05}, FUEL_S_PER_LAP)) == pytest.approx(0, abs=1e-9)
    assert np.mean(stint_errors(df, {"SOFT": 0.0, "HARD": 0.0}, FUEL_S_PER_LAP)) > 0.3


def test_within_slope_ignores_stint_offsets():
    g = pd.DataFrame({"stint_id": ["a"] * 5 + ["b"] * 5,
                      "tyre_age": list(range(5)) * 2,
                      "y": [10 + 0.1 * a for a in range(5)] + [50 + 0.1 * a for a in range(5)]})
    slope, n = within_slope(g)
    assert slope == pytest.approx(0.1) and n == 10


def test_sc_laps():
    rc = pd.DataFrame([
        {"date": "1", "lap_number": 10, "message": "SAFETY CAR DEPLOYED"},
        {"date": "2", "lap_number": 13, "message": "SAFETY CAR IN THIS LAP"},
        {"date": "3", "lap_number": 30, "message": "VIRTUAL SAFETY CAR DEPLOYED"},
        {"date": "4", "lap_number": 31, "message": "VIRTUAL SAFETY CAR ENDING"},
        {"date": "5", "lap_number": 40, "message": "BLUE FLAG FOR CAR 1"},
    ])
    assert sc_laps(rc, 57) == {10, 11, 12, 13, 30, 31}


def test_compound_number_and_pirelli_join():
    sessions = pd.DataFrame([{"year": 2026, "meeting_name": "Barcelona Grand Prix"},
                             {"year": 2025, "meeting_name": "São Paulo Grand Prix"}])
    pirelli = pd.DataFrame([{"year": 2026, "event": "Barcelona-Catalunya Grand Prix", "hard": "C2", "medium": "C3", "soft": "C4"},
                            {"year": 2025, "event": "São Paulo Grand Prix", "hard": "C2", "medium": "C3", "soft": "C4"}])
    out = attach_pirelli(sessions, pirelli)
    assert out["soft"].tolist() == ["C4", "C4"]
    assert compound_number(pd.Series({"compound": "MEDIUM", "hard": "C2", "medium": "C3", "soft": "C4"})) == 3.0
    assert np.isnan(compound_number(pd.Series({"compound": "INTERMEDIATE", "hard": "C2", "medium": "C3", "soft": "C4"})))


def _confounded_rates(n_races=24, seed=0):
    """Low-wear tracks get softer nominations (as Pirelli does), and inside every race the
    softer compound wears faster: +0.01 (medium) and +0.02 (soft) over hard."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_races):
        level = rng.uniform(0.0, 0.08)                 # the track's wear, hard-equivalent
        hard_c = 1 if level > 0.05 else (2 if level > 0.025 else 3)   # gentle tracks get softer tyres
        for comp, step, dc in (("HARD", 0.0, 0), ("MEDIUM", 0.01, 1), ("SOFT", 0.02, 2)):
            rows.append({"session_key": i, "year": 2025, "meeting_name": f"R{i}", "circuit": f"C{i}",
                         "date": f"2025-{1 + i // 4:02d}-{1 + i % 4 * 7:02d}", "compound": comp,
                         "compound_c": float(hard_c + dc), "rate": level + step + rng.normal(0, 0.002),
                         "laps": 200, "stints": 10, "track_temp": 35.0})
    return pd.DataFrame(rows)


def test_rate_model_keeps_soft_above_hard_despite_confounding():
    from f1agent.tyres import fit_predict, fit_predict_cnumber, levels_and_steps

    rates = _confounded_rates()
    races, step = levels_and_steps(rates)
    assert abs(step["MEDIUM"] - 0.01) < 0.003 and abs(step["SOFT"] - 0.02) < 0.003
    test = rates[rates["session_key"] == rates["session_key"].max()]
    train = rates[rates["session_key"] != rates["session_key"].max()]
    pred = dict(zip(test["compound"], fit_predict(train, test)))
    assert pred["SOFT"] > pred["MEDIUM"] > pred["HARD"]
    old = dict(zip(test["compound"], fit_predict_cnumber(train, test)))
    assert not (old["SOFT"] > old["MEDIUM"] > old["HARD"])   # the first version's inversion


def test_wear_per_c_step_is_measured_inside_races():
    from f1agent.tyres import wear_per_c_step

    k = wear_per_c_step(_confounded_rates())
    assert 0.008 < k < 0.012   # steps of 0.01 per C inside races, whatever the tracks do
