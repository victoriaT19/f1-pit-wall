"""Tools the agent can call. Plain functions over the race table and the model.

Every number the agent reports should come from one of these functions; the system
prompt tells it never to answer from memory. Each tool returns a JSON-serialisable dict.
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import predict, to_order
from .features import add_race_pace, build_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "races.csv"
LAPS = ROOT / "data" / "laps_clean.csv.gz"
BACKTEST = ROOT / "results" / "backtest"


# ── Loading ─────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def races() -> pd.DataFrame:
    return pd.read_csv(DATA)


@lru_cache(maxsize=1)
def laps() -> pd.DataFrame:
    return pd.read_csv(LAPS)


@lru_cache(maxsize=1)
def features() -> pd.DataFrame:
    return add_race_pace(build_features(races()), laps())


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    for w in ("grand prix", "grande premio", "gp", "do ", "de ", "da "):
        s = s.replace(w, " ")
    return " ".join(s.split())


ALIASES = {
    "brasil": "sao paulo", "brazil": "sao paulo", "interlagos": "sao paulo",
    "eua": "united states", "estados unidos": "united states", "austin": "united states",
    "mexico": "mexico city", "abu dhabi": "abu dhabi", "yas marina": "abu dhabi",
    "inglaterra": "british", "gra bretanha": "british", "silverstone": "british",
    "italia": "italian", "monza": "italian", "espanha": "spanish", "japao": "japanese",
    "australia": "australian", "holanda": "dutch", "belgica": "belgian", "hungria": "hungarian",
    "austria": "austrian", "canada": "canadian", "china": "chinese", "arabia saudita": "saudi arabian",
    "imola": "emilia romagna", "baku": "azerbaijan", "monaco": "monaco", "catar": "qatar",
}


def find_event(event: str, year: int | None = None) -> str | None:
    """Match a user's GP name ("Brasil", "Interlagos", "Abu Dhabi") to an event name."""
    q = _norm(event)
    q = ALIASES.get(q, q)
    names = races()["event"] if year is None else races().loc[races()["year"] == year, "event"]
    for name in dict.fromkeys(names):
        if q and (q in _norm(name) or _norm(name) in q):
            return name
    return None


def _driver(code: str) -> str:
    return str(code).strip().upper()[:3]


# ── Tools ───────────────────────────────────────────────────────────────────


def race_result(year: int, event: str) -> dict:
    """Finishing order of a past Grand Prix."""
    name = find_event(event, year)
    if not name:
        return {"error": f"GP '{event}' de {year} não encontrado nos dados (até {latest_race()})."}
    r = races()[(races()["year"] == year) & (races()["event"] == name)].sort_values("position")
    return {
        "year": year, "event": name, "date": r["date"].iloc[0],
        "result": [{"position": int(x.position), "driver": x.driver, "team": x.team,
                    "grid": int(x.grid), "status": x.status, "points": float(x.points)}
                   for x in r.itertuples()],
    }


def driver_season(driver: str, year: int) -> dict:
    """A driver's results and totals in one season."""
    d = _driver(driver)
    r = races()[(races()["year"] == year) & (races()["driver"] == d)].sort_values("round")
    if r.empty:
        return {"error": f"Piloto '{d}' não tem corridas em {year} nos dados."}
    return {
        "driver": d, "year": year, "team": r["team"].iloc[-1], "races": int(len(r)),
        "points": float(r["points"].sum()), "wins": int((r["position"] == 1).sum()),
        "podiums": int((r["position"] <= 3).sum()),
        "mean_finish": round(float(r["position"].mean()), 2),
        "mean_grid": round(float(r["grid"].mean()), 2),
        "results": [{"round": int(x.round), "event": x.event, "grid": int(x.grid),
                     "position": int(x.position)} for x in r.itertuples()],
    }


def standings(year: int, after_round: int | None = None) -> dict:
    """Drivers' championship (points from Grands Prix only; sprints are not in the data)."""
    r = races()[races()["year"] == year]
    if after_round is not None:
        r = r[r["round"] <= after_round]
    if r.empty:
        return {"error": f"Sem corridas de {year} nos dados."}
    pts = r.groupby("driver")["points"].sum().sort_values(ascending=False)
    return {"year": year, "after_round": int(r["round"].max()),
            "note": "Somente pontos de GPs; corridas sprint não estão nos dados.",
            "standings": [{"position": i + 1, "driver": d, "points": float(p)}
                          for i, (d, p) in enumerate(pts.items())]}


def track_info(event: str) -> dict:
    """History of a circuit: winners, pole-to-win rate, overtaking index."""
    name = find_event(event)
    if not name:
        return {"error": f"Circuito '{event}' não encontrado."}
    r = races()[races()["event"] == name]
    winners = r[r["position"] == 1].sort_values("year")
    return {
        "event": name, "editions": sorted(int(y) for y in r["year"].unique()),
        "winners": [{"year": int(x.year), "driver": x.driver, "team": x.team, "grid": int(x.grid)}
                    for x in winners.itertuples()],
        "pole_won_share": round(float((winners["grid"] == 1).mean()), 2),
        "overtaking_index": round(float((r["position"] - r["grid"]).abs().mean()), 2),
        "overtaking_index_all_tracks": round(float((races()["position"] - races()["grid"]).abs().mean()), 2),
    }


def predict_race(year: int, event: str, grid: dict | list | None = None) -> dict:
    """Model prediction for a race, trained only on races before it.

    For a past race the real grid is used and the actual result is returned next to the
    prediction. For a future race a grid must be given ({"VER": 1, ...}); without one the
    tool uses the current championship order as a provisional grid and says so.
    """
    feats = features()
    name = find_event(event, year)
    if name:  # past race: same procedure as the backtest
        race = feats[(feats["year"] == year) & (feats["event"] == name)]
        rnd = int(race["round"].iloc[0])
        train = feats[(feats["year"] < year) | ((feats["year"] == year) & (feats["round"] < rnd))]
        race = race.set_index("driver")
        note = "Corrida já disputada: previsão feita só com corridas anteriores (como no backtest)."
    else:
        race, train, note = _future_race(year, event, grid)
        if isinstance(race, dict):
            return race
    out = predict(train, race)
    pred = to_order(out["scores"]["ensemble"])
    probs = {o: pd.Series(v, index=race.index) for o, v in out["probs"]["ensemble"].items()}
    rows = []
    for drv in pred.sort_values().index:
        row = {"predicted": int(pred[drv]), "driver": drv, "grid": int(race.loc[drv, "grid"]),
               "p_win": round(float(probs["win"][drv]), 3), "p_podium": round(float(probs["podium"][drv]), 3),
               "p_top10": round(float(probs["top10"][drv]), 3)}
        if "position" in race and not pd.isna(race.loc[drv, "position"]):
            row["actual"] = int(race.loc[drv, "position"])
        rows.append(row)
    acc = backtest_summary()
    return {"year": year, "event": name or event, "note": note,
            "model": "ensemble: gradient boosting + Plackett-Luce ranking (probabilities: Plackett-Luce "
                     "blended with how often each grid slot won)",
            "model_accuracy": {k: acc.get("overall", {}).get(k) for k in ("ensemble", "grid")},
            "prediction": rows}


def _future_race(year: int, event: str, grid: dict | None):
    df = races()
    last = df[df["year"] == year]
    if last.empty:
        return {"error": f"Sem dados de {year}."}, None, None
    rnd = int(last["round"].max()) + 1
    drivers = last[last["round"] == last["round"].max()][["driver", "team"]]
    provisional = grid is None
    if provisional:
        st = standings(year)["standings"]
        order = [s["driver"] for s in st if s["driver"] in set(drivers["driver"])]
        grid = {d: i + 1 for i, d in enumerate(order)}
    if isinstance(grid, list):  # [{"driver": "VER", "position": 1}, ...] from the agent
        grid = {g["driver"]: g["position"] for g in grid}
    grid = {_driver(k): float(v) for k, v in grid.items()}
    fake = drivers.assign(year=year, round=rnd, event=event, date=_upcoming_date(event), position=np.nan,
                          status="", points=0.0, laps=np.nan, quali_time_s=np.nan)
    fake["grid"] = fake["driver"].map(grid).fillna(len(fake))
    feats = add_race_pace(build_features(pd.concat([df, fake], ignore_index=True)), laps())
    race = feats[(feats["year"] == year) & (feats["round"] == rnd)].set_index("driver")
    train = feats[~((feats["year"] == year) & (feats["round"] == rnd))]
    train = train[train["position"].notna()]
    note = ("Corrida futura. Grid provisório = ordem do campeonato (o grid real só existe após a "
            "classificação); a previsão vai mudar com o grid real." if provisional
            else "Corrida futura, com o grid informado.")
    return race, train, note


def _upcoming_date(event: str) -> str:
    """Date of an upcoming race from data/upcoming.csv (today if it is not listed)."""
    path = ROOT / "data" / "upcoming.csv"
    if path.exists():
        up = pd.read_csv(path)
        q = ALIASES.get(_norm(event), _norm(event))
        for _, r in up.iterrows():
            if q and q in _norm(r["meeting_name"]):
                return str(r["date"])[:10]
    return pd.Timestamp.today().strftime("%Y-%m-%d")


@lru_cache(maxsize=1)
def backtest_summary() -> dict:
    """How accurate the model has been, against simple baselines (from the saved backtest)."""
    path = BACKTEST / "summary.json"
    if not path.exists():
        return {"error": "Backtest ainda não foi rodado."}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_race() -> str:
    r = races().sort_values(["year", "round"]).iloc[-1]
    return f"{r['event']} {r['year']} ({r['date']})"


TOOLS = {
    "race_result": race_result,
    "driver_season": driver_season,
    "standings": standings,
    "track_info": track_info,
    "predict_race": predict_race,
    "backtest_summary": lambda: backtest_summary(),
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "race_result", "description": "Ordem de chegada de um GP já disputado.",
        "parameters": {"type": "object", "properties": {
            "year": {"type": "integer"}, "event": {"type": "string", "description": "Nome do GP, ex.: 'São Paulo', 'Monaco'"}},
            "required": ["year", "event"]}}},
    {"type": "function", "function": {
        "name": "driver_season", "description": "Resultados e totais de um piloto numa temporada.",
        "parameters": {"type": "object", "properties": {
            "driver": {"type": "string", "description": "Código de 3 letras, ex.: VER, NOR, LEC"},
            "year": {"type": "integer"}}, "required": ["driver", "year"]}}},
    {"type": "function", "function": {
        "name": "standings", "description": "Classificação do campeonato de pilotos (só pontos de GPs).",
        "parameters": {"type": "object", "properties": {
            "year": {"type": "integer"}, "after_round": {"type": "integer"}}, "required": ["year"]}}},
    {"type": "function", "function": {
        "name": "track_info", "description": "Histórico de um circuito: vencedores, quanto a pole vence, índice de ultrapassagem.",
        "parameters": {"type": "object", "properties": {"event": {"type": "string"}}, "required": ["event"]}}},
    {"type": "function", "function": {
        "name": "predict_race",
        "description": "Previsão do modelo para um GP. Para corrida futura, aceita o grid; sem grid usa a ordem do campeonato como grid provisório.",
        "parameters": {"type": "object", "properties": {
            "year": {"type": "integer"}, "event": {"type": "string"},
            "grid": {"type": "array", "description": "Opcional: grid de largada, um item por piloto",
                     "items": {"type": "object", "properties": {
                         "driver": {"type": "string"}, "position": {"type": "integer"}},
                         "required": ["driver", "position"]}}},
            "required": ["year", "event"]}}},
    {"type": "function", "function": {
        "name": "backtest_summary", "description": "Precisão histórica do modelo comparada com o grid, o campeonato e a forma recente.",
        "parameters": {"type": "object", "properties": {}}}},
]
