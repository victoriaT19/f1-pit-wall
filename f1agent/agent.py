"""The F1 agent: an LLM that answers by calling the tools in tools.py and strategy_tools.py.

Two providers, free tiers only:
  * OpenRouter, OpenAI-compatible tool calling (only ':free' model ids are accepted);
  * the Gemini API with native function calling (its OpenAI-compatible endpoint ignored
    tools for Gemma in our tests).
Free tiers are often rate-limited or overloaded, so provider="auto" walks a chain of free
routes (CHAIN: Gemma 4 31B and 26B and Nemotron on OpenRouter, then Gemma 4 31B on Gemini)
and moves on as soon as one fails; a 429 is not retried. An optional time budget caps the
whole answer, so a busy free API ends in a clear error instead of minutes of retries. The model decides which tools to call; this loop runs them and sends the results back until the model writes an answer. Every run returns a
trace: provider, tool calls, arguments, latency and tokens.

OPENROUTER_BASE_URL can point the OpenAI-compatible calls at a logging proxy such as
tokenprof (`tokenprof proxy --upstream https://openrouter.ai/api/v1`).

Usage:
  python -m f1agent.agent "Qual a melhor estratégia para Las Vegas?"
"""

from __future__ import annotations

import contextvars
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import strategy_tools
from .tools import TOOL_SCHEMAS as RESULT_SCHEMAS
from .tools import TOOLS as RESULT_TOOLS
from .tools import latest_race

# Results and finishing-order tools (tools.py) plus strategy and simulation (strategy_tools.py).
TOOLS = {**RESULT_TOOLS, **strategy_tools.TOOLS}
TOOL_SCHEMAS = RESULT_SCHEMAS + strategy_tools.TOOL_SCHEMAS

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemma-4-31b-it"                     # Gemini API name
OPENROUTER_MODEL = "google/gemma-4-31b-it:free"     # same model on OpenRouter, free tier
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# Free routes tried in order by provider="auto" (all checked to call tools correctly).
CHAIN = [
    ("openrouter", OPENROUTER_MODEL),
    ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    ("openrouter", "nvidia/nemotron-3.5-super-vl:free"),
    ("gemini", DEFAULT_MODEL),  # large daily quota, but slow (~25 s per step) when overloaded
]

SYSTEM = """Você é o engenheiro de estratégia de um time de Fórmula 1 no pit wall. Responda no idioma
da pergunta (português por padrão), de forma direta, como num rádio de equipe: primeiro a
resposta, depois o porquê em uma ou duas frases com os números.

Ferramentas de estratégia (as mesmas contas do dashboard):
- upcoming_races: próximas corridas, compostos, grid real ou estimado, sessões já feitas.
- strategy_plan: melhor estratégia de pit stop e cenários "e se" (compostos, temperatura,
  perda no pit, chance de safety car).
- simulate_race: a corrida inteira simulada para comparar duas estratégias de um piloto.
- prediction_log: previsões registradas antes das corridas.
Para perguntas de estratégia, rode as ferramentas em vez de opinar. Estratégias se escrevem
como 'M20 → H30' (composto + voltas do stint) e as voltas somam o total da corrida; use
strategy_plan para saber o número de voltas antes de montar uma estratégia.

Regras:
- Todo número, resultado ou previsão deve vir das ferramentas. Nunca responda de memória:
  seus dados de treino não cobrem as temporadas recentes, e as ferramentas sim.
- Os dados vão de 2023 até {latest}. Para algo fora disso, diga que não há dados.
- Pilotos aparecem pelo código de 3 letras (VER, NOR, LEC...). Pode citar o nome completo.
- Ao dar uma previsão, diga quão confiável o modelo é (use backtest_summary): ele empata
  com a ordem de largada no geral e acerta o vencedor menos vezes que o grid.
- Se a previsão usar grid provisório, avise.
- Os pontos vêm só de GPs; corridas sprint não estão nos dados.
- Em estratégia, diga a incerteza (faixa 10–90% ou chance de A à frente de B) e lembre que
  compostos provisórios ou grid estimado podem mudar a resposta. Safety car muda tudo:
  se perguntarem, compare cenários com strategy_plan ou simulate_race."""


def _env(name: str) -> str | None:
    """Environment variable, else the project's .env (never logged)."""
    env = Path(__file__).resolve().parents[1] / ".env"
    if name not in os.environ and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith(name + "="):
                os.environ[name] = line.split("=", 1)[1].strip().strip('"')
    return os.environ.get(name) or None


def _key() -> str:
    key = _env("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (environment or .env)")
    return key


def available_providers() -> list[str]:
    return [p for p, k in (("gemini", "GEMINI_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")) if _env(k)]


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
    provider: str = ""
    model: str = ""
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


# Absolute time after which no new request or retry starts (None: no limit). A context
# variable, so concurrent chats in the app each keep their own budget.
_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("deadline", default=None)


class BudgetExceeded(RuntimeError):
    pass


# Optional callback(kind, data) for progress: ("route", {...}) when a model is tried and
# ("tool", {...}) before each tool runs. The app uses it to show the steps live.
_on_event: contextvars.ContextVar = contextvars.ContextVar("on_event", default=None)


def _emit(kind: str, data: dict) -> None:
    cb = _on_event.get()
    if cb is not None:
        try:
            cb(kind, data)
        except Exception:  # a display problem must never break the answer
            pass


def _remaining() -> float:
    d = _deadline.get()
    return float("inf") if d is None else d - time.time()


def _wait_or_give_up(delay: float, what: str, failures: list | None) -> None:
    """Sleep before a retry, unless the time budget would run out first."""
    if delay >= _remaining():
        raise BudgetExceeded(f"{what} still failing ({failures or []}); time budget used up")
    time.sleep(delay)


def _timeout(cap: float) -> float:
    left = _remaining()
    if left <= 0:
        raise BudgetExceeded("time budget used up")
    return max(5.0, min(cap, left))


def _post(model: str, body: dict, retries: int = 6, failures: list | None = None) -> dict:
    """POST with retries on transient errors; each retried failure is appended to `failures`."""
    req_body = json.dumps(body).encode()
    delay = 3.0
    for attempt in range(retries):
        req = urllib.request.Request(API.format(model=model), data=req_body, headers={
            "Content-Type": "application/json", "x-goog-api-key": _key()})
        try:
            with urllib.request.urlopen(req, timeout=_timeout(60)) as r:
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
        _wait_or_give_up(delay, "Gemini API", failures)
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


def _gemini_answer(question: str, history: list[dict], trace: Trace, model: str, max_steps: int,
                   thinking: str | None) -> None:
    contents = [{"role": "user" if h["role"] == "user" else "model", "parts": [{"text": h["text"]}]}
                for h in history] + [{"role": "user", "parts": [{"text": question}]}]
    base = {
        "systemInstruction": {"parts": [{"text": SYSTEM.format(latest=latest_race())}]},
        "tools": gemini_tools(),
        "generationConfig": {"temperature": 0.2},
    }
    # By default the model thinks as it normally does; "minimal" or "high" override it.
    if thinking:
        base["generationConfig"]["thinkingConfig"] = {"thinkingLevel": thinking}
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
            return
        responses = []
        for call in calls:
            args = call.get("args", {}) or {}
            _emit("tool", {"name": call["name"], "args": args})
            result = json_safe(run_tool(call["name"], args))
            trace.tool_calls.append({"name": call["name"], "args": args,
                                     "error": result.get("error") if isinstance(result, dict) else None})
            resp = {"name": call["name"], "response": {"result": result}}
            if "id" in call:
                resp["id"] = call["id"]
            responses.append({"functionResponse": resp})
        contents.append({"role": "user", "parts": responses})
    raise RuntimeError(f"no answer after {max_steps} steps")


def _openrouter_post(body: dict, failures: list, headers: dict, retries: int = 4) -> dict:
    """OpenAI-compatible POST with short retries. A 429 is not retried: on the free tier it
    means the quota (ours or the shared upstream one) is used up, and the next route is tried."""
    url = (_env("OPENROUTER_BASE_URL") or OPENROUTER_URL).rstrip("/") + "/chat/completions"
    req_body = json.dumps(body).encode()
    delay = 3.0
    for attempt in range(retries):
        req = urllib.request.Request(url, data=req_body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {_env('OPENROUTER_API_KEY')}",
            "X-Title": "Pit wall F1 strategy", **headers})
        try:
            with urllib.request.urlopen(req, timeout=_timeout(90)) as r:
                data = json.load(r)
            if "error" in data:  # OpenRouter can answer 200 with an upstream error inside
                raise RuntimeError(f"OpenRouter: {str(data['error'])[:300]}")
            return data
        except urllib.error.HTTPError as e:
            failures.append(e.code)
            if e.code not in (500, 502, 503, 504) or attempt == retries - 1:
                raise RuntimeError(f"OpenRouter {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError):
            failures.append("timeout")
            if attempt == retries - 1:
                raise
        _wait_or_give_up(delay, "OpenRouter", failures)
        delay = min(delay * 2, 30)
    raise RuntimeError("unreachable")


def _openrouter_answer(question: str, history: list[dict], trace: Trace, model: str, max_steps: int,
                       run_id: str | None) -> None:
    # Only free models: a paid model id is refused here, before any request is made.
    if not model.endswith(":free"):
        raise ValueError(f"refusing non-free OpenRouter model {model!r}: only ':free' models are allowed")
    messages = [{"role": "system", "content": SYSTEM.format(latest=latest_race())}]
    messages += [{"role": h["role"], "content": h["text"]} for h in history]
    messages.append({"role": "user", "content": question})
    for step in range(max_steps):
        headers = {"X-Tokenprof-Run": run_id, "X-Tokenprof-Step": f"step-{step}"} if run_id else {}
        data = _openrouter_post({"model": model, "messages": messages, "tools": TOOL_SCHEMAS,
                                 "temperature": 0.2}, trace.api_retries, headers)
        trace.llm_calls += 1
        usage = data.get("usage") or {}
        trace.prompt_tokens += usage.get("prompt_tokens", 0)
        trace.output_tokens += usage.get("completion_tokens", 0)
        trace.thought_tokens += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         **({"tool_calls": calls} if calls else {})})
        if not calls:
            trace.answer = (msg.get("content") or "").strip()
            if not trace.answer:
                raise RuntimeError("empty answer")
            return
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            _emit("tool", {"name": name, "args": args})
            result = json_safe(run_tool(name, args))
            trace.tool_calls.append({"name": name, "args": args,
                                     "error": result.get("error") if isinstance(result, dict) else None})
            messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                             "content": json.dumps(result, ensure_ascii=False)})
    raise RuntimeError(f"no answer after {max_steps} steps")


def routes(provider: str = "auto", model: str | None = None) -> list[tuple[str, str]]:
    """(provider, model) pairs to try. An explicit provider or model gives a single route
    (evals stay reproducible); "auto" gives the free chain, limited to the keys that are set."""
    if provider == "auto" and model is None:
        have = set(available_providers())
        return [r for r in CHAIN if r[0] in have]
    if provider == "auto":
        provider = "openrouter" if "/" in model else "gemini"
        if provider not in available_providers():
            return []
    return [(provider, model or (OPENROUTER_MODEL if provider == "openrouter" else DEFAULT_MODEL))]


def ask(question: str, history: list[dict] | None = None, model: str | None = None,
        max_steps: int = 6, thinking: str | None = None, provider: str = "auto",
        run_id: str | None = None, time_budget: float | None = None, on_event=None) -> Trace:
    """Answer one question.

    `history` holds earlier turns as plain text, [{"role": "user"|"assistant", "text": ...}];
    tool results are not carried over (they are recomputed when needed), which keeps prompts
    short and lets a conversation switch provider. provider: "auto", "openrouter" or "gemini".
    time_budget (seconds) caps the whole answer: each route but the last gets at most half of
    what is left, the last one all of it. on_event(kind, data) reports progress.
    """
    trace = Trace(question=question)
    t0 = time.time()
    history = [h for h in (history or []) if h.get("text")]
    order = routes(provider, model)
    errors = []
    token = _on_event.set(on_event)
    for i, (prov, mod) in enumerate(order):
        trace.provider, trace.model = prov, mod
        trace.answer, trace.tool_calls = "", []
        if time_budget is not None:
            left = time_budget - (time.time() - t0)
            if left <= 1:
                break
            share = left * 0.5 if i < len(order) - 1 else left
            _deadline.set(time.time() + share)
        _emit("route", {"provider": prov, "model": mod, "attempt": i + 1})
        try:
            if prov == "openrouter":
                _openrouter_answer(question, history, trace, mod, max_steps, run_id)
            else:
                _gemini_answer(question, history, trace, mod, max_steps, thinking)
            trace.error = None
            break
        except Exception as exc:
            errors.append(f"{mod}: {type(exc).__name__}: {str(exc)[:160]}")
            trace.error = "; ".join(errors)
        finally:
            _deadline.set(None)
    _on_event.reset(token)
    if not order:
        trace.error = "no provider: set OPENROUTER_API_KEY or GEMINI_API_KEY"
    trace.seconds = round(time.time() - t0, 2)
    trace.history = history + ([{"role": "user", "text": question}, {"role": "assistant", "text": trace.answer}]
                               if trace.answer else [])  # type: ignore[attr-defined]
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
