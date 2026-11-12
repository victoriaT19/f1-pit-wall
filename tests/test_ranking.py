from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

from f1agent.racesim import Car, Track, simulate
from f1agent.ranking import PlackettLuce, PLParams, dnf_rates, outcome_probs


def synthetic_races(n_races=30, seed=0):
    """Six drivers with known strengths; results drawn from a Plackett-Luce model."""
    rng = np.random.default_rng(seed)
    strength = {"AAA": 2.0, "BBB": 1.2, "CCC": 0.6, "DDD": 0.0, "EEE": -0.6, "FFF": -1.2}
    teams = {"AAA": "T1", "BBB": "T1", "CCC": "T2", "DDD": "T2", "EEE": "T3", "FFF": "T3"}
    rows = []
    for r in range(n_races):
        u = np.array(list(strength.values())) + rng.gumbel(size=6)
        order = np.argsort(-u)
        for pos, i in enumerate(order, start=1):
            d = list(strength)[i]
            rows.append({"year": 2025, "round": r + 1, "date": (date(2025, 3, 1) + timedelta(days=7 * r)).isoformat(),
                         "driver": d, "team": teams[d], "position": pos, "finished": True, "status": "Finished",
                         "grid": 3.5, "quali_gap_pct": 1.0, "race_pace": np.nan})
    return pd.DataFrame(rows), strength


def test_plackett_luce_recovers_the_strength_order():
    df, strength = synthetic_races()
    m = PlackettLuce(PLParams(half_life_days=10_000)).fit(df, "2026-01-01")
    race = df[df["round"] == 1].copy()
    u = pd.Series(m.strength(race), index=race["driver"])
    assert list(u.sort_values(ascending=False).index[:2]) == ["AAA", "BBB"]
    assert u["AAA"] > u["DDD"] > u["FFF"]


def test_gradient_matches_finite_differences():
    df, _ = synthetic_races(n_races=6)
    m = PlackettLuce()
    m.fit(df, "2026-01-01")  # builds the internal arrays
    theta = np.random.default_rng(1).normal(0, 0.3, len(m.beta) + len(m.a) + len(m.b))
    _, grad = m._nll(theta)
    num = approx_fprime(theta, lambda t: m._nll(t)[0], 1e-6)
    assert np.allclose(grad, num, atol=1e-4)


def test_probabilities_are_consistent_and_dnfs_go_to_the_back():
    df, _ = synthetic_races()
    m = PlackettLuce().fit(df, "2026-01-01")
    race = df[df["round"] == 1].copy()
    m.dnf = {d: 0.0 for d in race["driver"]}
    m.dnf["AAA"] = 1.0
    pos = m.simulate(race, n=2000)
    p = outcome_probs(pos)
    assert abs(p["win"].sum() - 1) < 1e-9 and abs(p["podium"].sum() - 3) < 1e-9
    a = list(race["driver"]).index("AAA")
    assert (pos[:, a] == 6).all()


def test_dnf_rates_are_shrunk_towards_the_field():
    res = pd.DataFrame({"driver": ["X"] * 2 + ["Y"] * 20, "date": ["2025-01-01"] * 22,
                        "status": ["Retired", "Retired"] + ["Finished"] * 20})
    rates, field = dnf_rates(res, "2025-02-01", prior_races=10)
    assert 0 < field < 1
    assert field < rates["X"] < 1.0        # two DNFs in two races, but shrunk
    assert rates["Y"] < field


def test_simulated_retirements_drop_to_the_back_except_in_the_replayed_race():
    track = Track(laps=20, rate={"MEDIUM": 0.05}, rate_sd={"MEDIUM": 0.0}, offset={"MEDIUM": 0.0},
                  offset_sd={"MEDIUM": 0.0}, pit_loss=20.0, sc_lap_prob=0.0, pass_threshold=1.0)
    cars = [Car("FAST", -1.0, (("MEDIUM", 20),), dnf_prob=1.0)] + \
           [Car(f"C{i}", 0.0, (("MEDIUM", 20),)) for i in range(4)]
    pos = simulate(cars, track, n_sims=50, seed=0)["positions"]
    assert pos[0, 0] == 1                  # first simulated race (the replay): no retirements
    assert (pos[1:, 0] == 5).all()
