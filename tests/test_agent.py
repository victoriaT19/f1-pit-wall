"""Agent and tools without calling the API (real data in data/races.csv)."""

import math

import pytest

from f1agent import agent
from f1agent import tools as T
from f1agent.agent_eval import mentions_driver, mentions_number


def test_find_event_aliases():
    assert T.find_event("Brasil", 2026) == "São Paulo Grand Prix"
    assert T.find_event("Interlagos") == "São Paulo Grand Prix"
    assert T.find_event("Mônaco") == "Monaco Grand Prix"
    assert T.find_event("Las Vegas", 2026) is None  # not raced yet


def test_tools_return_consistent_numbers():
    r = T.race_result(2026, "São Paulo")
    assert r["result"][0]["position"] == 1
    season = T.driver_season("nor", 2025)
    assert season["driver"] == "NOR" and season["races"] == len(season["results"])
    df = T.races()
    assert season["points"] == df[(df["year"] == 2025) & (df["driver"] == "NOR")]["points"].sum()
    st = T.standings(2025)["standings"]
    assert next(s for s in st if s["driver"] == "NOR")["points"] == season["points"]
    assert st == sorted(st, key=lambda s: -s["points"])


def test_predict_past_race_uses_only_earlier_races_and_reports_actual():
    p = T.predict_race(2026, "São Paulo")
    assert "anteriores" in p["note"]
    assert sorted(x["predicted"] for x in p["prediction"]) == list(range(1, len(p["prediction"]) + 1))
    assert all("actual" in x for x in p["prediction"])


def test_predict_future_race_with_and_without_grid():
    prov = T.predict_race(2026, "Las Vegas")
    assert "provisório" in prov["note"]
    grid = [{"driver": x["driver"], "position": i + 1}
            for i, x in enumerate(reversed(prov["prediction"]))]
    given = T.predict_race(2026, "Las Vegas", grid=grid)
    assert "informado" in given["note"]
    assert given["prediction"] != prov["prediction"]


def test_tool_errors_are_returned_not_raised():
    assert "error" in T.race_result(2026, "Atlantis")
    assert "error" in agent.run_tool("race_result", {"year": 2026})
    assert "error" in agent.run_tool("nope", {})


def test_json_safe():
    out = agent.json_safe({"a": float("nan"), "b": [1.0, float("inf")], "c": "x"})
    assert out == {"a": None, "b": [1.0, None], "c": "x"}


def test_gemini_schema_conversion():
    tools = agent.gemini_tools()[0]["functionDeclarations"]
    names = {t["name"] for t in tools}
    assert names == set(agent.TOOLS)
    assert {"strategy_plan", "simulate_race", "upcoming_races", "prediction_log"} <= names
    pr = next(t for t in tools if t["name"] == "predict_race")
    assert pr["parameters"]["type"] == "OBJECT"
    assert pr["parameters"]["properties"]["grid"]["items"]["properties"]["position"]["type"] == "INTEGER"
    assert "parameters" not in next(t for t in tools if t["name"] == "backtest_summary")


def test_agent_loop_with_fake_model(monkeypatch):
    replies = iter([
        {"candidates": [{"content": {"role": "model", "parts": [
            {"functionCall": {"name": "race_result", "args": {"year": 2026, "event": "São Paulo"}, "id": "c1"}}]}}],
         "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10}},
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "Norris venceu."}]}}],
         "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 5}},
    ])
    sent = []

    def fake_post(model, body, retries=6, failures=None):
        sent.append(body)
        return next(replies)

    monkeypatch.setattr(agent, "_post", fake_post)
    t = agent.ask("Quem venceu em São Paulo 2026?")
    assert t.answer == "Norris venceu."
    assert [c["name"] for c in t.tool_calls] == ["race_result"]
    assert t.llm_calls == 2 and t.prompt_tokens == 400 and t.output_tokens == 15
    # the second request carries the tool result, with the call id
    fr = sent[1]["contents"][-1]["parts"][0]["functionResponse"]
    assert fr["id"] == "c1" and fr["response"]["result"]["result"][0]["position"] == 1


def test_agent_stops_after_max_steps(monkeypatch):
    call = {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": "standings", "args": {"year": 2026}}}]}}]}
    monkeypatch.setattr(agent, "_post", lambda *a, **k: call)
    t = agent.ask("loop", max_steps=3)
    assert t.error and len(t.tool_calls) == 3


def test_answer_checkers():
    assert mentions_driver("Lando Norris venceu", "NOR")
    assert mentions_driver("O NOR venceu", "NOR")
    assert not mentions_driver("ao norte", "NOR")
    assert mentions_number("fez 394 pontos", 394)
    assert mentions_number("média de 4,25", 4.25)
    assert not mentions_number("média de 4,4", 4.25)


def test_auto_provider_falls_back_to_gemini_and_keeps_text_history(monkeypatch):
    monkeypatch.setattr(agent, "available_providers", lambda: ["openrouter", "gemini"])

    def broken(*a, **k):
        raise RuntimeError("OpenRouter 429: rate-limited upstream")

    def gemini(question, history, trace, *a, **k):
        assert history == [{"role": "user", "text": "oi"}, {"role": "assistant", "text": "olá"}]
        trace.answer = "S42 → M8"

    monkeypatch.setattr(agent, "_openrouter_answer", broken)
    monkeypatch.setattr(agent, "_gemini_answer", gemini)
    t = agent.ask("estratégia?", history=[{"role": "user", "text": "oi"}, {"role": "assistant", "text": "olá"}])
    assert t.provider == "gemini" and t.answer == "S42 → M8" and t.error is None
    assert t.history[-1] == {"role": "assistant", "text": "S42 → M8"}


def test_no_provider_is_reported(monkeypatch):
    monkeypatch.setattr(agent, "available_providers", lambda: [])
    t = agent.ask("oi")
    assert t.answer == "" and "no provider" in t.error


def test_openrouter_refuses_paid_models_before_any_request(monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("no request should be made for a paid model")

    monkeypatch.setattr(agent, "_openrouter_post", no_network)
    monkeypatch.setattr(agent, "available_providers", lambda: ["openrouter"])
    t = agent.ask("oi", provider="openrouter", model="google/gemma-4-31b-it")
    assert t.answer == "" and "non-free" in t.error
    assert agent.OPENROUTER_MODEL.endswith(":free")
