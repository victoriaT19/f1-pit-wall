"""Evaluate the agent on questions whose answers the code can compute from the data.

Each case has a question in Portuguese, the tool the agent should use, and a checker that
derives the expected answer from data/races.csv (not hard-coded), so the evaluation stays
valid as the data grows. A case passes when the answer contains the expected value
(driver code or surname, or the number). It also records whether the expected tool was
called, the number of LLM calls, tokens and seconds.

Usage:
  python -m f1agent.agent_eval --out results/agent_eval
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

from . import tools as T

NAMES = {
    "ALB": "Albon", "ALO": "Alonso", "ANT": "Antonelli", "BEA": "Bearman", "BOR": "Bortoleto",
    "BOT": "Bottas", "CAM": "Camara", "COL": "Colapinto", "DEV": "de Vries", "DOO": "Doohan",
    "GAS": "Gasly", "HAD": "Hadjar", "HAM": "Hamilton", "HIR": "Hirakawa", "HUL": "Hulkenberg",
    "LAW": "Lawson", "LEC": "Leclerc", "LIN": "Lindblad", "MAG": "Magnussen", "NOR": "Norris",
    "OCO": "Ocon", "PER": "Perez", "PIA": "Piastri", "RIC": "Ricciardo", "RUS": "Russell",
    "SAI": "Sainz", "SAR": "Sargeant", "STR": "Stroll", "TSU": "Tsunoda", "VER": "Verstappen",
    "ZHO": "Zhou",
}


def _plain(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def mentions_driver(answer: str, code: str) -> bool:
    a = _plain(answer)
    return code.lower() in re.findall(r"[a-z]+", a) or _plain(NAMES.get(code, code)) in a


# "um/uma" are left out: they are also the article ("um pódio").
NUMBER_WORDS = {"dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6,
                "sete": 7, "oito": 8, "nove": 9, "dez": 10}


def mentions_number(answer: str, value: float, tol: float = 0.051) -> bool:
    """True if the answer states `value`, in digits ("4", "4,25") or, for small whole
    numbers, in words ("quatro")."""
    nums = [float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", answer)]
    nums += [float(v) for w, v in NUMBER_WORDS.items() if re.search(r"\b" + w + r"\b", _plain(answer))]
    return any(abs(n - value) < tol for n in nums)


def winner(year: int, event: str) -> str:
    r = T.race_result(year, event)["result"]
    return r[0]["driver"]


def cases() -> list[dict]:
    """Questions with expected answers computed from the data."""
    st26 = T.standings(2026)["standings"]
    nor25 = T.driver_season("NOR", 2025)
    ver24 = T.driver_season("VER", 2024)
    pia25, nor25p = T.driver_season("PIA", 2025)["podiums"], nor25["podiums"]
    lec_ad24 = next(x for x in T.race_result(2024, "Abu Dhabi")["result"] if x["driver"] == "LEC")
    monaco = T.track_info("Monaco")
    bt = T.backtest_summary()["overall"]
    future = T.predict_race(2026, "Las Vegas")["prediction"][0]["driver"]
    ham25 = T.driver_season("HAM", 2025)["team"]
    ver26 = T.driver_season("VER", 2026)["mean_finish"]

    return [
        {"id": "winner_sp26", "q": "Quem venceu o GP de São Paulo de 2026?", "tool": "race_result",
         "check": lambda a: mentions_driver(a, winner(2026, "São Paulo"))},
        {"id": "winner_monaco24", "q": "Quem ganhou em Mônaco em 2024?", "tool": "race_result",
         "check": lambda a: mentions_driver(a, winner(2024, "Monaco"))},
        {"id": "winner_ad25", "q": "Quem venceu o GP de Abu Dhabi de 2025?", "tool": "race_result",
         "check": lambda a: mentions_driver(a, winner(2025, "Abu Dhabi"))},
        {"id": "points_nor25", "q": "Quantos pontos o Lando Norris fez em GPs em 2025?", "tool": "driver_season",
         "check": lambda a: mentions_number(a, nor25["points"])},
        {"id": "wins_ver24", "q": "Quantas corridas o Verstappen venceu em 2024?", "tool": "driver_season",
         "check": lambda a: mentions_number(a, ver24["wins"])},
        {"id": "leader26", "q": "Quem lidera o campeonato de 2026 até agora?", "tool": "standings",
         "check": lambda a: mentions_driver(a, st26[0]["driver"])},
        {"id": "second26", "q": "Quem está em segundo no campeonato de 2026?", "tool": "standings",
         "check": lambda a: mentions_driver(a, st26[1]["driver"])},
        {"id": "podiums_cmp25", "q": "Quem fez mais pódios em 2025, Norris ou Piastri?", "tool": "driver_season",
         "check": lambda a: mentions_driver(a, "NOR" if nor25p >= pia25 else "PIA")
         and mentions_number(a, max(nor25p, pia25))},
        {"id": "grid_lec_ad24", "q": "De que posição o Leclerc largou no GP de Abu Dhabi de 2024?", "tool": "race_result",
         "check": lambda a: mentions_number(a, lec_ad24["grid"])},
        {"id": "team_ham25", "q": "Por qual equipe o Hamilton correu em 2025?", "tool": "driver_season",
         "check": lambda a: _plain(ham25.split()[0]) in _plain(a)},
        {"id": "mean_ver26", "q": "Qual a posição média de chegada do Verstappen em 2026?", "tool": "driver_season",
         "check": lambda a: mentions_number(a, ver26)},
        {"id": "monaco_overtaking", "q": "Mônaco tem mais ou menos ultrapassagens que a média dos circuitos? Mostre o índice.",
         "tool": "track_info",
         "check": lambda a: mentions_number(a, monaco["overtaking_index"]) and "menos" in _plain(a)},
        {"id": "monaco_pole", "q": "Em Mônaco, quem larga na pole costuma vencer? Quantas vezes desde 2023?", "tool": "track_info",
         "check": lambda a: mentions_number(a, sum(w["grid"] == 1 for w in monaco["winners"]))},
        {"id": "model_accuracy", "q": "Com que frequência o modelo acerta o vencedor, comparado com simplesmente apostar no pole?",
         "tool": "backtest_summary",
         # Percentages: "44%" and "44,12%" are both right, so allow half a point.
         "check": lambda a: any(mentions_number(a, bt[m]["winner"] * 100, tol=0.6)
                                for m in ("ensemble", "model") if m in bt)
         and mentions_number(a, bt["grid"]["winner"] * 100, tol=0.6)},
        {"id": "predict_vegas", "q": "Quem o modelo prevê como vencedor do GP de Las Vegas de 2026?", "tool": "predict_race",
         "check": lambda a: mentions_driver(a, future) and "provis" in _plain(a)},
        {"id": "out_of_data", "q": "Quem vai ser campeão de 2027?", "tool": None,
         "check": lambda a: any(w in _plain(a) for w in ("nao ha dados", "nao tenho dados", "nao e possivel",
                                                          "nao consigo", "sem dados", "nao posso", "nao ha como"))},
    ]


def rescore(out: Path) -> dict:
    """Re-check saved answers with the current checkers (no API calls)."""
    by_id = {c["id"]: c for c in cases()}
    rows = [json.loads(l) for l in (out / "answers.jsonl").read_text(encoding="utf-8").splitlines() if l]
    for r in rows:
        r["correct"] = bool(r["answer"]) and bool(by_id[r["id"]]["check"](r["answer"]))
    return _write(out, rows, json.loads((out / "summary.json").read_text(encoding="utf-8"))["model"])


def _write(out: Path, rows: list[dict], model: str) -> dict:
    n = len(rows)
    summary = {
        "model": model, "cases": n,
        "correct": sum(r["correct"] for r in rows),
        "tool_ok": sum(r["tool_ok"] for r in rows),
        "errors": sum(bool(r["error"]) for r in rows),
        "mean_llm_calls": round(sum(r["llm_calls"] for r in rows) / n, 2),
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / n),
        "mean_output_tokens": round(sum(r["output_tokens"] for r in rows) / n),
        "mean_thought_tokens": round(sum(r.get("thought_tokens", 0) for r in rows) / n),
        "api_retries_total": sum(len(r.get("api_retries", [])) for r in rows),
        "questions_with_retries": sum(bool(r.get("api_retries")) for r in rows),
        "median_seconds": sorted(r["seconds"] for r in rows)[n // 2],
    }
    (out / "answers.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                       encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def evaluate(out: Path, model: str | None, only: list[str] | None = None, thinking: str | None = None,
             provider: str = "gemini") -> dict:
    from .agent import ask

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if only and (out / "answers.jsonl").exists():  # re-run some cases, keep the others
        rows = [json.loads(l) for l in (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()
                if l and json.loads(l)["id"] not in only]
    for c in cases():
        if only and c["id"] not in only:
            continue
        t = ask(c["q"], model=model, thinking=thinking, provider=provider, run_id=f"eval-{c['id']}")
        called = [tc["name"] for tc in t.tool_calls]
        ok = bool(t.answer) and bool(c["check"](t.answer))
        tool_ok = (not called) if c["tool"] is None else (c["tool"] in called)
        rows.append({"id": c["id"], "question": c["q"], "answer": t.answer, "error": t.error,
                     "correct": ok, "expected_tool": c["tool"], "tools_called": called,
                     "tool_ok": tool_ok, "llm_calls": t.llm_calls, "prompt_tokens": t.prompt_tokens,
                     "output_tokens": t.output_tokens, "thought_tokens": t.thought_tokens,
                     "api_retries": t.api_retries, "seconds": t.seconds})
        print(f"[{len(rows)}/{len(cases())}] {c['id']}: correct={ok} tool_ok={tool_ok} "
              f"calls={called} {t.seconds}s", flush=True)
        time.sleep(1)
    order = [c["id"] for c in cases()]
    rows.sort(key=lambda r: order.index(r["id"]))
    label = f"{provider}:{model or 'default'}"
    return _write(out, rows, label + (f" (thinking={thinking})" if thinking else " (default thinking)"))


def main(argv: list[str] | None = None) -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/agent_eval")
    p.add_argument("--model", default=None, help="default: the provider's Gemma 4 31B")
    p.add_argument("--provider", default="gemini", choices=["gemini", "openrouter", "auto"])
    p.add_argument("--only", nargs="*", help="re-run only these case ids, keeping the others")
    p.add_argument("--thinking", default=None, help="minimal or high; default: the model's own")
    p.add_argument("--rescore", action="store_true", help="re-check saved answers, no API calls")
    args = p.parse_args(argv)
    if args.rescore:
        rescore(Path(args.out))
    else:
        evaluate(Path(args.out), args.model, args.only, args.thinking, args.provider)


if __name__ == "__main__":
    main()
