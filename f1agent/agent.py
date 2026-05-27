"""The F1 agent: an LLM that answers in Portuguese by calling the tools in tools.py.

Runs Gemma 4 31B through the Gemini API's native function calling (its OpenAI-compatible
endpoint ignored tools for Gemma in our tests). The model decides which tools to call;
this loop executes them and sends the results back until the model writes an answer.
Every run returns a trace: tool calls, arguments, latency and tokens.

Usage:
  python -m f1agent.agent "Quem venceu o GP de São Paulo de 2026?"
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .tools import TOOL_SCHEMAS, TOOLS, latest_race

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemma-4-31b-it"

SYSTEM = """Você é um analista de Fórmula 1. Responda em português, de forma direta.

Regras:
- Todo número, resultado ou previsão deve vir das ferramentas. Nunca responda de memória:
  seus dados de treino não cobrem as temporadas recentes, e as ferramentas sim.
- Os dados vão de 2023 até {latest}. Para algo fora disso, diga que não há dados.
- Pilotos aparecem pelo código de 3 letras (VER, NOR, LEC...). Pode citar o nome completo.
- Ao dar uma previsão, diga quão confiável o modelo é (use backtest_summary): ele empata
  com a ordem de largada no geral e acerta o vencedor menos vezes que o grid.
- Se a previsão usar grid provisório, avise.
- Os pontos vêm só de GPs; corridas sprint não estão nos dados."""


def _key() -> str:
    env = Path(__file__).resolve().parents[1] / ".env"
    if "GEMINI_API_KEY" not in os.environ and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                os.environ["GEMINI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (environment or .env)")
    return key


def to_gemini_schema(schema: dict) -> dict:
    """OpenAI-style JSON schema -> Gemini's OpenAPI subset (upper-case types)."""
    out = {}
    for k, v in schema.items():
        if k == "type":
            out[k] = v.upper()
        elif k == "properties":
            out[k] = {name: to_gemini_schema(s) for name, s in v.items()}
        elif k == "items":
            out[k] = to_gemini_schema(v)
        elif k in ("description", "required", "enum"):
            out[k] = v
    return out


def gemini_tools() -> list[dict]:
    decls = []
    for t in TOOL_SCHEMAS:
        f = t["function"]
        d = {"name": f["name"], "description": f["description"]}
        if f["parameters"].get("properties"):
            d["parameters"] = to_gemini_schema(f["parameters"])
        decls.append(d)
    return [{"functionDeclarations": decls}]


@dataclass
class Trace:
    question: str
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    api_retries: list = field(default_factory=list)  # HTTP codes/timeouts that were retried
    seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _post(model: str, body: dict, retries: int = 6, failures: list | None = None) -> dict:
    """POST with retries on transient errors; each retried failure is appended to `failures`."""
    req_body = json.dumps(body).encode()
    delay = 3.0
    for attempt in range(retries):
        req = urllib.request.Request(API.format(model=model), data=req_body, headers={
            "Content-Type": "application/json", "x-goog-api-key": _key()})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # 429 (rate limit), 500 and 503 are frequent on the free tier and transient.
            if failures is not None:
                failures.append(e.code)
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise RuntimeError(f"Gemini API {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError):
            if failures is not None:
                failures.append("timeout")
            if attempt == retries - 1:
                raise
        time.sleep(delay)
        delay = min(delay * 2, 40)
    raise RuntimeError("unreachable")


def json_safe(x):
    """NaN and numpy types are not valid JSON; the API rejects the whole request."""
    import math

    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if hasattr(x, "item"):  # numpy scalar
        x = x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def run_tool(name: str, args: dict) -> dict:
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"ferramenta desconhecida: {name}"}
    try:
        return fn(**args)
    except TypeError as exc:
        return {"error": f"argumentos inválidos para {name}: {exc}"}
    except Exception as exc:  # a tool failure is reported to the model, not raised
        return {"error": f"{type(exc).__name__}: {exc}"}


def ask(question: str, history: list[dict] | None = None, model: str = DEFAULT_MODEL,
        max_steps: int = 6, thinking: str | None = None) -> Trace:
    """Answer one question. `history` is a list of Gemini `contents` from earlier turns."""
    trace = Trace(question=question)
    t0 = time.time()
    contents = list(history or []) + [{"role": "user", "parts": [{"text": question}]}]
    base = {
        "systemInstruction": {"parts": [{"text": SYSTEM.format(latest=latest_race())}]},
        "tools": gemini_tools(),
        "generationConfig": {"temperature": 0.2},
    }
    # By default the model thinks as it normally does; "minimal" or "high" override it.
    if thinking:
        base["generationConfig"]["thinkingConfig"] = {"thinkingLevel": thinking}
    try:
        for _ in range(max_steps):
            data = _post(model, {**base, "contents": list(contents)}, failures=trace.api_retries)
            trace.llm_calls += 1
            usage = data.get("usageMetadata", {})
            trace.prompt_tokens += usage.get("promptTokenCount", 0)
            trace.output_tokens += usage.get("candidatesTokenCount", 0)
            trace.thought_tokens += usage.get("thoughtsTokenCount", 0)
            content = data["candidates"][0].get("content", {"role": "model", "parts": []})
            parts = content.get("parts", [])
            # Keep the model turn exactly as returned (thought signatures included).
            contents.append({"role": "model", "parts": parts})
            calls = [p["functionCall"] for p in parts if "functionCall" in p]
            if not calls:
                trace.answer = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
                break
            responses = []
            for call in calls:
                args = call.get("args", {}) or {}
                result = json_safe(run_tool(call["name"], args))
                trace.tool_calls.append({"name": call["name"], "args": args,
                                         "error": result.get("error") if isinstance(result, dict) else None})
                resp = {"name": call["name"], "response": {"result": result}}
                if "id" in call:
                    resp["id"] = call["id"]
                responses.append({"functionResponse": resp})
            contents.append({"role": "user", "parts": responses})
        else:
            trace.error = f"no answer after {max_steps} steps"
    except Exception as exc:
        trace.error = f"{type(exc).__name__}: {exc}"
    trace.seconds = round(time.time() - t0, 2)
    trace.history = contents  # type: ignore[attr-defined]
    return trace


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    q = " ".join(sys.argv[1:]) or "Quem venceu o GP de São Paulo de 2026?"
    t = ask(q)
    print(t.answer or t.error)
    print("\n--- trace:", json.dumps({k: v for k, v in t.to_dict().items() if k not in ("answer", "history")},
                                    ensure_ascii=False))


if __name__ == "__main__":
    main()
