"""Strategy tools of the agent, on the real data (no model API calls)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "data/laps_clean.csv.gz").exists(), reason="data not built")

from f1agent import strategy_tools as S  # noqa: E402


@pytest.mark.parametrize("name,expected", [("Las Vegas", "next:Las Vegas Grand Prix"), ("vegas", "next:Las Vegas Grand Prix"),
                                           ("Catar", "next:Qatar Grand Prix"), ("Abu Dhabi", "next:Abu Dhabi Grand Prix")])
def test_resolve_race_understands_common_names(name, expected):
    assert S.resolve_race(name) == expected


def test_resolve_past_race_and_unknown():
    assert S.resolve_race("Brasil", 2026).startswith("past:")
    assert S.resolve_race("Atlantis") is None


def test_parse_strategy_accepts_several_spellings_and_checks_laps():
    assert S._parse_strategy("M25 → H25", 50) == (("MEDIUM", 25), ("HARD", 25))
    assert S._parse_strategy("medium 25, hard 25", 50) == (("MEDIUM", 25), ("HARD", 25))
    assert S._parse_strategy("M20-H20", 50) is None  # laps must add up to the race distance


def test_strategy_plan_applies_what_if_inputs():
    base = S.strategy_plan("Las Vegas")
    alt = S.strategy_plan("Las Vegas", pit_loss=30, safety_car_chance=0.0)
    assert base["inputs"]["laps"] == 50 and len(base["top5"]) == 5
    assert alt["inputs"]["pit_loss_s"] == 30.0 and alt["inputs"]["safety_car_start_chance_per_lap"] == 0.0


def test_simulate_race_reports_errors_the_model_can_fix():
    assert "error" in S.simulate_race("Las Vegas", "XXX", "M25 H25")
    assert "error" in S.simulate_race("Las Vegas", "HAM", "M10 H10")
    out = S.simulate_race("Las Vegas", "HAM", "M25 → H25", runs=60)
    pct = out["a_vs_b_same_conditions_pct"]
    assert abs(pct["a_ahead"] + pct["level"] + pct["a_behind"] - 100) < 0.5
