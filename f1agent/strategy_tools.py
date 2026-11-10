"""Agent tools over the strategy side of the project: the same calculations the dashboard shows.

  upcoming_races   next races, compounds (official or provisional), grid real or estimated
  strategy_plan    single-car Monte Carlo plan for a race, with optional what-if inputs
  simulate_race    whole-field simulation: a driver on strategy A vs B
  prediction_log   forecasts saved after each session before a race

Races are named as people say them ("Las Vegas", "Brasil", "Mexico 2026"); `resolve_race`
maps them to the dashboard's keys. Every tool returns a JSON-serialisable dict.
"""

from __future__ import annotations

import json
import re

import numpy as np

from .tools import ALIASES, _norm

COMPOUND = {"S": "SOFT", "M": "MEDIUM", "H": "HARD"}
STRATEGY_ACCURACY = ("Backtest 2026: a primeira parada recomendada erra em média ~8 voltas (igual a uma "
                     "referência histórica simples) e 1/4 das paradas reais acontece sob safety car, que "
                     "nenhum plano prevê. Serve para comparar opções, não como previsão exata.")


def _planner():
    from . import planner  # heavy import (loads the lap data), only when a strategy tool runs
    return planner


def resolve_race(event: str, year: int | None = None) -> str | None:
    P = _planner()
    q = _norm(event)
    q = re.sub(r"\b(19|20)\d{2}\b", "", q).strip() or q
    q = ALIASES.get(q, q)
    for _, key in P.race_choices():
        row = P.race_row(key)[0]
        if year is not None and int(row["year"]) != int(year):
            continue
        names = [_norm(row["meeting_name"]), _norm(row["circuit"])]
        if any(q and (q in n or n in q) for n in names):
            return key
    return None


def _parse_strategy(text: str, laps: int) -> tuple | None:
    """'M20 → H30', 'M20-H30', 'medium 20, hard 30' -> ((MEDIUM, 20), (HARD, 30))."""
    t = text.upper().replace("SOFT", "S").replace("MEDIUM", "M").replace("HARD", "H")
    parts = re.findall(r"([SMH])\s*(\d+)", t)
    if not parts:
        return None
    s = tuple((COMPOUND[c], int(n)) for c, n in parts)
    if sum(n for _, n in s) != laps:
        return None
    return s


def _describe(s: tuple) -> str:
    return " → ".join(f"{c[0]}{n}" for c, n in s)


def _overrides(hard, medium, soft, track_temp, pit_loss, safety_car_chance) -> dict:
    kw = {}
    if hard and medium and soft:
        kw["compounds"] = tuple(float(str(c).upper().lstrip("C")) for c in (hard, medium, soft))
    if track_temp is not None:
        kw["track_temp"] = float(track_temp)
    if pit_loss is not None:
        kw["pit_loss"] = float(pit_loss)
    if safety_car_chance is not None:
        kw["sc_lap_prob"] = float(safety_car_chance)
    return kw


def upcoming_races() -> dict:
    """Next races with their tyre nomination, whether the grid is real and the sessions already run."""
    P = _planner()
    s = P.store()
    out = []
    for _, key in P.race_choices():
        if not key.startswith("next:"):
            continue
        row, laps, info = P.race_row(key)
        drivers, _, real = P.field(key)
        seen = sorted(s.laps.loc[(s.laps["year"] == row["year"]) & (s.laps["meeting_name"] == row["meeting_name"]),
                                 "session_name"].unique())
        out.append({"race": row["meeting_name"], "date_utc": row["date"], "laps": laps,
                    "compounds": {"hard": row["hard"], "medium": row["medium"], "soft": row["soft"]},
                    "compounds_status": info["compounds_status"],
                    "grid": "real (após a classificação)" if real else "estimado pelo ritmo recente",
                    "front_of_grid": drivers[:3], "sessions_seen": seen})
    return {"upcoming": out, "latest_race_in_data": P.past_races().iloc[-1]["meeting_name"]}


def strategy_plan(event: str, year: int | None = None, hard: str | None = None, medium: str | None = None,
                  soft: str | None = None, track_temp: float | None = None, pit_loss: float | None = None,
                  safety_car_chance: float | None = None) -> dict:
    """Best pit strategies for a race (Monte Carlo, one car), optionally with changed conditions."""
    P = _planner()
    key = resolve_race(event, year)
    if key is None:
        return {"error": f"corrida não encontrada: {event!r}. Use upcoming_races para ver as próximas."}
    kw = _overrides(hard, medium, soft, track_temp, pit_loss, safety_car_chance)
    inp, info, res = P.plan(key, **kw)
    best = res.iloc[0]
    near = res[(res["stops"] == best["stops"]) & (res["median_gap_s"] <= best["median_gap_s"] + 2.0)]
    first = sorted({p[0] for p in near["pit_laps"]})
    alt = res[res["stops"] != best["stops"]].iloc[0]
    row = P.race_row(key)[0]
    top = [{"strategy": r.strategy, "stops": int(r.stops), "pit_laps": list(r.pit_laps),
            "median_loss_vs_best_s": round(float(r.median_gap_s), 1),
            "loss_10_90_pct_s": [round(float(r.p10_gap_s), 1), round(float(r.p90_gap_s), 1)],
            "fastest_in_pct_of_races": round(float(r.p_fastest) * 100, 1)} for r in res.head(5).itertuples()]
    return {
        "race": f"{row['meeting_name']} {int(row['year'])}", "already_run": key.startswith("past:"),
        "recommended": best["strategy"], "first_stop_window_laps": [int(first[0]), int(first[-1])],
        "plans_within_2s_of_best": int(len(near)), "top5": top,
        "best_other_stop_count": {"strategy": alt["strategy"], "stops": int(alt["stops"]),
                                  "median_extra_s": round(float(alt["median_gap_s"] - best["median_gap_s"]), 1)},
        "inputs": {"laps": inp.laps, "compounds": {"hard": row["hard"], "medium": row["medium"], "soft": row["soft"]}
                   if "compounds" not in kw else dict(zip(("hard", "medium", "soft"), (hard, medium, soft))),
                   "compounds_status": info["compounds_status"], "track_temp_c": info["track_temp"],
                   "pit_loss_s": round(inp.pit_loss, 1), "safety_car_start_chance_per_lap": round(inp.sc_lap_prob, 3),
                   "degradation_s_per_lap": {c: round(v, 3) for c, v in inp.rate.items()},
                   "pace_vs_hard_new_s_per_lap": {c: round(v, 2) for c, v in inp.offset.items()}},
        "how_to_read": ("median_loss_vs_best_s: quanto o plano perde para o melhor plano de cada corrida simulada "
                        "(mediana). loss_10_90_pct_s: essa perda nas corridas simuladas mais favoráveis e mais "
                        "desfavoráveis; a faixa larga vem sobretudo do safety car sorteado (uma parada sob safety car "
                        "custa metade) e da incerteza do desgaste. first_stop_window_laps: primeiras paradas entre os "
                        "planos com o mesmo número de paradas a até 2 s do melhor; janela larga = o momento importa pouco."),
        "how_reliable": STRATEGY_ACCURACY,
        "strategy_format": "Letra do composto + voltas do stint, ex.: 'M20 → H30' (S=macio, M=médio, H=duro).",
    }


def simulate_race(event: str, driver: str, strategy_a: str, strategy_b: str | None = None,
                  year: int | None = None, runs: int = 300) -> dict:
    """Simulate the whole field: `driver` on strategy A vs B (default B: the best plan)."""
    P = _planner()
    key = resolve_race(event, year)
    if key is None:
        return {"error": f"corrida não encontrada: {event!r}"}
    inp, _, res = P.plan(key, n_sims=800)
    drivers, _, real = P.field(key)
    d = str(driver).strip().upper()[:3]
    if d not in drivers:
        return {"error": f"piloto {driver!r} não está no grid; pilotos: {', '.join(drivers)}"}
    a = _parse_strategy(strategy_a, inp.laps)
    b = _parse_strategy(strategy_b, inp.laps) if strategy_b else P.parse_strategy(res.iloc[0]["strategy"])
    for name, s, txt in (("A", a, strategy_a), ("B", b, strategy_b)):
        if s is None:
            return {"error": f"estratégia {name} inválida: {txt!r}. Use o formato 'M20 → H30' e some "
                             f"{inp.laps} voltas no total."}
    out = P.scenario(key, d, _describe(a), _describe(b), n_sims=int(min(max(runs, 50), 500)))
    pa, pb = out["a"]["positions"], out["b"]["positions"]
    summary = lambda p: {"expected_position": round(float(p.mean()), 1), "median": int(np.median(p)),
                         "podium_pct": round(float((p <= 3).mean()) * 100, 1),
                         "top10_pct": round(float((p <= 10).mean()) * 100, 1),
                         "best_10pct": int(np.percentile(p, 10)), "worst_10pct": int(np.percentile(p, 90))}
    return {
        "race": P.race_row(key)[0]["meeting_name"], "driver": d, "grid_position": out["grid_pos"],
        "grid": "real" if real else "estimado (classificação ainda não aconteceu)",
        "strategy_a": {"strategy": _describe(a), **summary(pa)},
        "strategy_b": {"strategy": _describe(b), **summary(pb)},
        "a_vs_b_same_conditions_pct": {"a_ahead": round(float((pa < pb).mean()) * 100, 1),
                                       "level": round(float((pa == pb).mean()) * 100, 1),
                                       "a_behind": round(float((pa > pb).mean()) * 100, 1)},
        "rivals": "Rivais usam o melhor plano com as paradas espalhadas em ±3 voltas.",
        "model_limits": "Ultrapassagem calibrada em 2025 é difícil (a largada mexe mais nas posições que a pista); "
                        "safety car sorteado pela taxa histórica do circuito.",
    }


def prediction_log(event: str | None = None) -> dict:
    """Forecasts saved after each session before a race (python -m f1agent.update)."""
    P = _planner()
    path = P.ROOT / "results/live/predictions.jsonl"
    if not path.exists():
        return {"entries": [], "note": "Nenhuma previsão registrada ainda."}
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if event:
        q = ALIASES.get(_norm(event), _norm(event))
        rows = [r for r in rows if q in _norm(r["race"])]
    return {"entries": rows[-12:]}


TOOLS = {"upcoming_races": upcoming_races, "strategy_plan": strategy_plan,
         "simulate_race": simulate_race, "prediction_log": prediction_log}

_EVENT = {"type": "string", "description": "Nome do GP ou circuito, ex.: 'Las Vegas', 'Brasil', 'Abu Dhabi'"}
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "upcoming_races",
        "description": "Próximas corridas: data, voltas, compostos da Pirelli (oficiais ou provisórios), "
                       "se o grid já é o real e quais sessões já aconteceram.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "strategy_plan",
        "description": "Melhores estratégias de pit stop para uma corrida (simulação Monte Carlo de um carro): "
                       "recomendada, janela da primeira parada, top 5 com incerteza e os dados usados. Aceita "
                       "cenários: outros compostos (C1-C6), temperatura da pista, perda no pit, chance de safety car.",
        "parameters": {"type": "object", "properties": {
            "event": _EVENT, "year": {"type": "integer"},
            "hard": {"type": "string", "description": "Composto duro, ex.: 'C3' (informe os três ou nenhum)"},
            "medium": {"type": "string"}, "soft": {"type": "string"},
            "track_temp": {"type": "number", "description": "Temperatura da pista em °C"},
            "pit_loss": {"type": "number", "description": "Tempo perdido num pit stop, em segundos"},
            "safety_car_chance": {"type": "number", "description": "Chance de safety car começar em cada volta (0 a 0.1)"}},
            "required": ["event"]}}},
    {"type": "function", "function": {
        "name": "simulate_race",
        "description": "Simula a corrida com todos os carros (tráfego, ultrapassagem, safety car, undercut) "
                       "e compara um piloto em duas estratégias: posição esperada, chance de pódio e com que "
                       "frequência A termina à frente de B. Sem strategy_b, compara com o melhor plano.",
        "parameters": {"type": "object", "properties": {
            "event": _EVENT, "driver": {"type": "string", "description": "Código de 3 letras, ex.: HAM"},
            "strategy_a": {"type": "string", "description": "Ex.: 'M20 → H30' (as voltas somam o total da corrida)"},
            "strategy_b": {"type": "string"}, "year": {"type": "integer"},
            "runs": {"type": "integer", "description": "Número de corridas simuladas (50 a 500, padrão 300)"}},
            "required": ["event", "driver", "strategy_a"]}}},
    {"type": "function", "function": {
        "name": "prediction_log",
        "description": "Previsões registradas depois de cada sessão (treinos, classificação) antes de uma corrida.",
        "parameters": {"type": "object", "properties": {"event": _EVENT}}}},
]
