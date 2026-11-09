"""Pit wall: an F1 race-strategy dashboard (Gradio).

Tabs: strategy plan for a race (single car, Monte Carlo), whole-field race simulation
with an animated on-track replay, the scoreboard of every backtest, and the methodology.

Run locally:  python app.py   (then open http://127.0.0.1:7860)
"""

from __future__ import annotations

import html
import io
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from f1agent import planner as P

# Free Gradio Spaces run on ZeroGPU, which refuses to start an app with no @spaces.GPU
# function. Everything here runs on CPU; this placeholder only satisfies that check.
try:
    import spaces

    @spaces.GPU(duration=10)
    def _zerogpu_placeholder() -> None:
        return None
except ImportError:  # running locally
    pass

ROOT = Path(__file__).resolve().parent
COMPOUND_COLOUR = {"SOFT": "#E8002D", "MEDIUM": "#FFC906", "HARD": "#E6E8EC",
                   "INTERMEDIATE": "#43B02A", "WET": "#0067AD"}
A_COLOUR, B_COLOUR = "#3D8BFF", "#FF8A1F"
INK, INK_STRONG = "#9AA3B2", "#D6DBE4"
GRID = "rgba(154,163,178,0.14)"
SURFACE = "#121722"
C_CHOICES = [f"C{i}" for i in range(1, 7)]
SPEED_SCALE = [[0.0, "#2F5597"], [0.5, "#5B9BF0"], [1.0, "#EAF2FF"]]  # one hue: brighter = faster


def _layout(fig: go.Figure, title: str, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0.01, y=0.97, font=dict(size=14, color=INK_STRONG)), height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK, size=12, family="Titillium Web, system-ui, sans-serif"),
                      margin=dict(l=12, r=12, t=52, b=12), hoverlabel=dict(font_size=12, bgcolor="#05070b"),
                      legend=dict(orientation="h", y=1.0, x=1, xanchor="right", yanchor="bottom",
                                  font=dict(color=INK)), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def cards(items: list[tuple[str, str, str]]) -> str:
    """KPI cards: (label, value, detail)."""
    cells = "".join(f'<div class="kpi"><div class="kpi-label">{html.escape(a)}</div>'
                    f'<div class="kpi-value">{b}</div><div class="kpi-sub">{c}</div></div>' for a, b, c in items)
    return f'<div class="kpis">{cells}</div>'


def _stints_html(desc: str) -> str:
    """Strategy text as coloured tyre chips: M8 → S42."""
    out = []
    for comp, n in P.parse_strategy(desc):
        out.append(f'<span class="tyre" style="--c:{COMPOUND_COLOUR[comp]}">{comp[0]}</span>{n}')
    return '<span class="arrow">→</span>'.join(out)


# ── Strategy tab ────────────────────────────────────────────────────────────


def race_defaults(key: str):
    row, n_laps, info = P.race_row(key)
    inp, extra, _ = P.plan(key, n_sims=50)
    return (row["hard"], row["medium"], row["soft"], float(extra["track_temp"]), round(inp.pit_loss, 1),
            round(inp.sc_lap_prob, 3))


def run_plan(key, hard, medium, soft, temp, pit, sc):
    comps = tuple(float(c.lstrip("C")) for c in (hard, medium, soft))
    inp, info, res = P.plan(key, compounds=comps, track_temp=float(temp), pit_loss=float(pit), sc_lap_prob=float(sc))
    alt = _alternative(res)
    options = list(dict.fromkeys(res["strategy"].head(25).tolist() + [alt]))
    return (plan_cards(info, inp, res, (hard, medium, soft)), map_fig(key), ranking_fig(res),
            timeline_fig(key, res, inp.laps), degradation_fig(inp), notes_md(key, info, res),
            gr.update(choices=options, value=res.iloc[0]["strategy"]), gr.update(choices=options, value=alt))


def _alternative(res: pd.DataFrame) -> str:
    other = res[res["stops"] != res.iloc[0]["stops"]]
    return (other.iloc[0] if len(other) else res.iloc[1])["strategy"]


def plan_cards(info, inp, res, comps) -> str:
    best = res.iloc[0]
    alt = res[res["stops"] != best["stops"]].iloc[0]
    near = res[(res["stops"] == best["stops"]) & (res["median_gap_s"] <= best["median_gap_s"] + 2.0)]
    first = sorted({p[0] for p in near["pit_laps"]})
    stops = int(best["stops"])
    provisional = info["compounds_status"].startswith("provisional")
    return cards([
        ("Recommended", _stints_html(best["strategy"]),
         f"{stops} stop{'s' if stops > 1 else ''} · pit lap {', '.join(map(str, best['pit_laps']))}"),
        ("First-stop window", f"L{first[0]}–L{first[-1]}", f"{len(near)} plans within 2 s of the best"),
        (f"Best {int(alt['stops'])}-stop", _stints_html(alt["strategy"]),
         f"{alt['median_gap_s'] - best['median_gap_s']:+.1f} s in the median race"),
        ("Tyres", f'<span class="tyre" style="--c:{COMPOUND_COLOUR["HARD"]}">H</span>{comps[0]} '
                  f'<span class="tyre" style="--c:{COMPOUND_COLOUR["MEDIUM"]}">M</span>{comps[1]} '
                  f'<span class="tyre" style="--c:{COMPOUND_COLOUR["SOFT"]}">S</span>{comps[2]}',
         "⚠ provisional (last year's)" if provisional else "official Pirelli nomination"),
        ("Race", f"{inp.laps} laps", f"pit loss {inp.pit_loss:.1f} s · SC {inp.sc_lap_prob * 100:.1f}%/lap"),
    ])


def notes_md(key, info, res) -> str:
    stints = None if info["upcoming"] else P.real_stints(int(key.split(":")[1]))
    wet = stints is not None and stints["compound"].isin(["INTERMEDIATE", "WET"]).any()
    near = res[res["median_gap_s"] <= res.iloc[0]["median_gap_s"] + 2.0]
    lines = ["#### Reading the plan"]
    if len(near) >= 10:
        lines.append(f"- **{len(near)} strategies are within 2 s of the best**: with this little wear, timing "
                     "barely matters. React to safety cars instead.")
    lines += [
        "- Degradation comes from a model fitted on earlier races (practice long runs made it worse in the backtest).",
        "- Backtest on 2026: the plan's first stop is off by a median of 8 laps, like the historical baseline; "
        "a quarter of real first stops come from safety cars. Use it to compare options, not as a forecast.",
    ]
    if info["compounds_status"].startswith("provisional"):
        lines.append("- Compounds are **last year's nomination** until Pirelli announces this year's: edit them on the left.")
    if wet:
        lines.append("- 🌧️ This race was wet: the dry-strategy plan does not apply to the real race.")
    return "\n".join(lines)


def ranking_fig(res: pd.DataFrame, top: int = 10) -> go.Figure:
    r = res.head(top).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=r["median_gap_s"], y=r["strategy"], mode="markers",
        marker=dict(size=11, color=A_COLOUR, line=dict(width=2, color=SURFACE)),
        error_x=dict(type="data", symmetric=False, array=r["p90_gap_s"] - r["median_gap_s"],
                     arrayminus=r["median_gap_s"] - r["p10_gap_s"], color="rgba(61,139,255,0.55)", thickness=3,
                     width=0),
        customdata=np.c_[r["p10_gap_s"], r["p90_gap_s"], r["p_fastest"] * 100],
        hovertemplate="<b>%{y}</b><br>median %{x:.1f} s behind the best<br>"
                      "10–90%%: %{customdata[0]:.1f} – %{customdata[1]:.1f} s<br>fastest in %{customdata[2]:.0f}%% "
                      "of races<extra></extra>"))
    fig = _layout(fig, "Time lost vs the best plan in each simulated race", 430)
    fig.update_xaxes(title="seconds (dot: median · bar: 10–90% of races)", rangemode="tozero")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=INK_STRONG))
    return fig


def timeline_fig(key: str, res: pd.DataFrame, laps: int) -> go.Figure:
    rows = [(f"Plan {i + 1}", P.parse_strategy(s)) for i, s in enumerate(res["strategy"].head(5))]
    real = None if key.startswith("next") else P.real_stints(int(key.split(":")[1]))
    fig = go.Figure()
    seen = set()

    def bar(label, comp, start, length):
        fig.add_trace(go.Bar(
            y=[label], x=[length], base=[start - 1], orientation="h", name=comp.title(),
            marker=dict(color=COMPOUND_COLOUR.get(comp, INK), line=dict(color=SURFACE, width=2),
                        cornerradius=4),
            legendgroup=comp, showlegend=comp not in seen, width=0.58,
            hovertemplate=f"<b>{label}</b><br>{comp.title()}: laps {start}–{start + length - 1}<extra></extra>"))
        seen.add(comp)

    labels = []
    for label, strat in rows:
        lap = 1
        for comp, n in strat:
            bar(label, comp, lap, n)
            lap += n
        labels.append(label)
    if real is not None and len(real):
        for drv, g in real.groupby("driver", sort=False):
            lab = f"P{int(g['rank'].iloc[0]) + 1} {drv}"
            for _, s in g.iterrows():
                bar(lab, s["compound"], int(s["start"]), int(s["end"] - s["start"] + 1))
            labels.append(lab)
    title = "Stints: top 5 plans" + (" · what the top 10 really did" if real is not None else "")
    fig = _layout(fig, title, 110 + 25 * len(labels), barmode="overlay", bargap=0.35)
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1], gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(color=INK_STRONG))
    fig.update_xaxes(title="lap", range=[0, laps])
    if real is not None and len(real):  # divider between plans and real stints
        fig.add_hline(y=len(labels) - len(rows) - 0.5, line=dict(color=GRID, width=1, dash="dot"))
    return fig


def degradation_fig(inp) -> go.Figure:
    d = P.degradation_curves(inp)
    fig = go.Figure()
    for c in ("SOFT", "MEDIUM", "HARD"):
        g = d[d["compound"] == c]
        col = COMPOUND_COLOUR[c]
        fig.add_trace(go.Scatter(x=np.r_[g["age"], g["age"][::-1]], y=np.r_[g["high"], g["low"][::-1]],
                                 fill="toself", fillcolor=_alpha(col, 0.06), line=dict(width=0),
                                 hoverinfo="skip", showlegend=False, legendgroup=c))
        fig.add_trace(go.Scatter(x=g["age"], y=g["delta"], mode="lines", name=c.title(), legendgroup=c,
                                 line=dict(color=col, width=2.5),
                                 hovertemplate=f"{c.title()} · age %{{x}} laps: %{{y:+.2f}} s/lap<extra></extra>"))
        last = g.iloc[-1]
        fig.add_annotation(x=last["age"], y=last["delta"], text=c.title(), showarrow=False, xanchor="left",
                           xshift=6, font=dict(color=INK_STRONG, size=11))
    fig = _layout(fig, "Lap time vs a new hard tyre, by tyre age", 360, hovermode="x unified")
    fig.update_xaxes(title="tyre age (laps)", range=[0, 39])
    fig.update_yaxes(title="s per lap (lower = faster) · band ±1 sd")
    return fig


def _alpha(hex_colour: str, a: float) -> str:
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def map_fig(key: str):
    tr = P.smooth_track(key)
    if tr is None:
        return None
    x, y, sp = np.array(tr["x"]), np.array(tr["y"]), np.array(tr["speed"])
    xc, yc = np.r_[x, x[0]], np.r_[y, y[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xc, y=yc, mode="lines", line=dict(color="#05070b", width=17), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xc, y=yc, mode="lines", line=dict(color="#2a303c", width=12), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=6, color=sp, colorscale=SPEED_SCALE, cmin=float(sp.min()), cmax=float(sp.max()),
                    colorbar=dict(title=dict(text="km/h", side="top", font=dict(color=INK)), thickness=8, len=0.5,
                                  outlinewidth=0, tickfont=dict(color=INK), x=1.0, y=0.5)),
        customdata=np.array(tr["f"]) * tr["lap_time"],
        hovertemplate="%{marker.color:.0f} km/h · %{customdata:.1f} s into the lap<extra></extra>"))
    # start/finish and sector marks
    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers", hoverinfo="skip",
                             marker=dict(symbol="square", size=11, color="#ffffff", line=dict(color="#05070b", width=2))))
    for name, f in (("S2", tr["sectors"][0]), ("S3", tr["sectors"][1])):
        if f is None:
            continue
        i = int(np.searchsorted(tr["f"], f))
        i = min(i, len(x) - 1)
        fig.add_annotation(x=x[i], y=y[i], text=name, showarrow=True, arrowhead=0, arrowcolor=INK, ax=18, ay=-18,
                           font=dict(color=INK, size=11))
    fig.add_annotation(x=x[0], y=y[0], text="Start", showarrow=True, arrowhead=0, arrowcolor=INK, ax=-22, ay=22,
                       font=dict(color=INK_STRONG, size=11))
    circuit = P.race_row(key)[0]["circuit"]
    fig = _layout(fig, f"{circuit} · speed on a fast lap ({tr['driver']}, {tr['lap_time']:.1f} s)", 430,
                  showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig


# ── Race simulation tab ─────────────────────────────────────────────────────


def field_drivers(key: str):
    drivers, _, real = P.field(key)
    return gr.update(choices=drivers, value=drivers[min(2, len(drivers) - 1)]), gr.update(
        choices=["Simulated race (A)", "Real race"] if key.startswith("past") else ["Simulated race (A)"],
        value="Simulated race (A)")


def run_scenario(key, driver, strat_a, strat_b, hard, medium, soft, temp, pit, sc):
    comps = tuple(float(c.lstrip("C")) for c in (hard, medium, soft))
    out = P.scenario(key, driver, strat_a, strat_b, n_sims=300, compounds=comps, track_temp=float(temp),
                     pit_loss=float(pit), sc_lap_prob=float(sc))
    a, b = out["a"]["positions"], out["b"]["positions"]
    kpis = cards([
        ("Grid", f"P{out['grid_pos']}", "real grid" if out["real_grid"] else "estimated from recent race pace"),
        ("Expected finish · A", f"P{a.mean():.1f}", _stints_html(strat_a)),
        ("Expected finish · B", f"P{b.mean():.1f}", _stints_html(strat_b)),
        ("A vs B, same conditions", f"{np.mean(a < b) * 100:.0f}% ahead",
         f"level {np.mean(a == b) * 100:.0f}% · behind {np.mean(a > b) * 100:.0f}%"),
    ])
    sim_trace = pd.DataFrame(out["a"]["trace"], index=np.arange(1, out["laps"] + 1), columns=out["drivers"])
    state = {"key": key, "driver": driver, "trace": sim_trace.to_json(orient="split")}
    replay, gap = show_source("Simulated race (A)", state)
    return kpis, positions_fig(a, b, len(out["drivers"])), replay, gap, state, gr.update(value="Simulated race (A)")


def show_source(source, state):
    if not state:
        return "", None
    key, driver = state["key"], state["driver"]
    if source == "Real race" and key.startswith("past"):
        trace = P.real_trace(int(key.split(":")[1]))
        title, gap_title = "Real race", "Real race: gap to the leader"
    else:
        trace = pd.read_json(io.StringIO(state["trace"]), orient="split")
        title, gap_title = f"One simulated race · {driver} on strategy A", f"One simulated race: gap to the leader"
    data = P.replay_data(key, trace, driver, title)
    return (json.dumps(data, separators=(",", ":")) if data else ""), trace_fig(trace, driver, gap_title)


def positions_fig(a: np.ndarray, b: np.ndarray, n: int) -> go.Figure:
    pos = np.arange(1, n + 1)
    fig = go.Figure()
    for arr, name, col in ((a, "Strategy A", A_COLOUR), (b, "Strategy B", B_COLOUR)):
        share = np.array([(arr == p).mean() * 100 for p in pos])
        fig.add_trace(go.Bar(x=pos, y=share, name=name, marker=dict(color=col, line=dict(width=2, color=SURFACE),
                                                                     cornerradius=4),
                             hovertemplate=f"{name}: P%{{x}} in %{{y:.0f}}%% of races<extra></extra>"))
    fig = _layout(fig, "Finishing position across 300 simulated races", 360, barmode="group", bargap=0.25)
    fig.update_xaxes(title="position", dtick=1, gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title="% of races")
    return fig


def trace_fig(trace: pd.DataFrame, focus: str, title: str) -> go.Figure:
    colours = P.store().colours
    fig = go.Figure()
    for d in trace.columns:
        if d == focus:
            continue
        fig.add_trace(go.Scatter(x=trace.index, y=trace[d], mode="lines", name=d, showlegend=False,
                                 line=dict(color="rgba(154,163,178,0.28)", width=1),
                                 hovertemplate=f"{d} · lap %{{x}}: +%{{y:.1f}} s<extra></extra>"))
    if focus in trace.columns:
        col = colours.get(focus, A_COLOUR)
        fig.add_trace(go.Scatter(x=trace.index, y=trace[focus], mode="lines", name=focus,
                                 line=dict(color=col, width=3),
                                 hovertemplate=f"<b>{focus}</b> · lap %{{x}}: +%{{y:.1f}} s<extra></extra>"))
        last = trace[focus].dropna()
        if len(last):
            fig.add_annotation(x=last.index[-1], y=last.iloc[-1], text=focus, showarrow=False, xanchor="left",
                               xshift=6, font=dict(color=INK_STRONG, size=12))
    cap = np.nanpercentile(trace.values, 90) * 1.15 if np.isfinite(trace.values).any() else 60
    fig = _layout(fig, title, 360, showlegend=False)
    fig.update_yaxes(title="seconds behind the leader", autorange=False, range=[cap, -2])
    fig.update_xaxes(title="lap")
    return fig


# ── Scoreboard ──────────────────────────────────────────────────────────────


def scoreboard():
    bt = pd.read_csv(ROOT / "results/backtest/per_race.csv")
    b26 = bt[bt["year"] == 2026].pivot_table(index=["round", "event"], columns="method", values="spearman").reset_index()
    cols = [c for c in ("model", "grid", "championship", "form") if c in b26.columns]
    b26 = b26[["round", "event"] + cols].round(3)
    means = b26[cols].mean()
    rs = json.loads((ROOT / "results/racesim/summary.json").read_text(encoding="utf-8"))
    order_cards = cards([(f"Spearman · {c}", f"{means[c]:.3f}", "mean over 2026 races") for c in cols])
    stop_cards = cards([
        ("Whole-field simulation", f"{rs['multi']['first_stop_error_median']:.1f} laps",
         f"first-stop error · stops right {rs['multi']['stops_match'] * 100:.0f}%"),
        ("Single-car simulation", f"{rs['single']['first_stop_error_median']:.1f} laps",
         f"first-stop error · stops right {rs['single']['stops_match'] * 100:.0f}%"),
        ("Historical baseline", f"{rs['baseline']['first_stop_error_median']:.1f} laps",
         f"first-stop error · stops right {rs['baseline']['stops_match'] * 100:.0f}%"),
    ])
    return order_cards, b26, stop_cards


def prediction_log() -> pd.DataFrame:
    """Forecasts saved before each race by `python -m f1agent.update`, latest first."""
    path = ROOT / "results/live/predictions.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["logged (UTC)", "race", "sessions seen", "recommended", "first stop", "grid"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame([{
        "logged (UTC)": r["logged_at"][:16].replace("T", " "), "race": r["race"],
        "sessions seen": ", ".join(r["sessions_seen"]) or "none", "recommended": r["recommended"],
        "first stop": f"L{r['first_stop_window'][0]}–L{r['first_stop_window'][1]}",
        "grid": r["grid"] + (f" ({' / '.join(r['front_row'])})" if r["grid"] == "real" else ""),
        "tyres": f"{r['compounds']['hard']}/{r['compounds']['medium']}/{r['compounds']['soft']} "
                 f"{'' if r['compounds']['status'] == 'official' else '(provisional)'}".strip(),
    } for r in rows])
    return df.iloc[::-1].reset_index(drop=True)


# ── App ─────────────────────────────────────────────────────────────────────

CSS = """
.gradio-container {width: 100% !important; max-width: 1680px !important; margin: 0 auto !important; padding: 12px 16px !important}
.block:focus-within, .form:focus-within {z-index: 1000 !important; position: relative}
#hero h1 {font-size: 30px; margin: 0; letter-spacing: .5px}
#hero h1 span {color: #e10600}
#hero p {margin: 2px 0 0; color: #9aa3b2}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px}
.kpi {background: #121722; border: 1px solid #232a38; border-left: 3px solid #e10600; border-radius: 10px;
      padding: 10px 14px}
.kpi-label {font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: #9aa3b2}
.kpi-value {font-size: 20px; font-weight: 700; color: #ffffff; margin: 2px 0; font-variant-numeric: tabular-nums;
            display: flex; align-items: center; gap: 4px; flex-wrap: wrap}
.kpi-sub {font-size: 12px; color: #9aa3b2; display: flex; align-items: center; gap: 4px; flex-wrap: wrap}
.tyre {display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px;
       border-radius: 50%; border: 3px solid var(--c); color: #fff; font-size: 10px; font-weight: 700;
       background: #05070b; margin-right: 2px}
.kpi-sub .tyre {width: 16px; height: 16px; border-width: 2px; font-size: 8px}
.arrow {color: #5d6677; margin: 0 4px}
#sidebar {background: #0f131c; border: 1px solid #1d2330; border-radius: 12px; padding: 6px}
footer {opacity: .5}
/* Gradio resets border widths inside HTML blocks on some deployments: restate them */
.kpis .kpi {border-style: solid !important; border-width: 1px 1px 1px 3px !important}
.kpis .tyre {border-style: solid !important; border-width: 3px !important}
.kpis .kpi-sub .tyre {border-width: 2px !important}
"""


def theme() -> gr.themes.Base:
    t = gr.themes.Base(primary_hue=gr.themes.colors.red, neutral_hue=gr.themes.colors.slate,
                       font=[gr.themes.GoogleFont("Titillium Web"), "system-ui", "sans-serif"],
                       radius_size=gr.themes.sizes.radius_md)
    return t.set(
        body_background_fill="#0b0e14", body_background_fill_dark="#0b0e14",
        body_text_color="#d6dbe4", body_text_color_dark="#d6dbe4",
        body_text_color_subdued="#9aa3b2", body_text_color_subdued_dark="#9aa3b2",
        background_fill_primary="#121722", background_fill_primary_dark="#121722",
        background_fill_secondary="#0f131c", background_fill_secondary_dark="#0f131c",
        block_background_fill="#121722", block_background_fill_dark="#121722",
        block_border_color="#232a38", block_border_color_dark="#232a38",
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        block_label_text_color="#9aa3b2", block_label_text_color_dark="#9aa3b2",
        block_title_text_color="#9aa3b2", block_title_text_color_dark="#9aa3b2",
        border_color_primary="#232a38", border_color_primary_dark="#232a38",
        input_background_fill="#0b0e14", input_background_fill_dark="#0b0e14",
        input_border_color="#2a3242", input_border_color_dark="#2a3242",
        button_primary_background_fill="#e10600", button_primary_background_fill_dark="#e10600",
        button_primary_background_fill_hover="#ff2a1f", button_primary_background_fill_hover_dark="#ff2a1f",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#1b2130", button_secondary_background_fill_dark="#1b2130",
        button_secondary_text_color="#d6dbe4", button_secondary_text_color_dark="#d6dbe4",
        slider_color="#e10600", slider_color_dark="#e10600",
        color_accent_soft="#2a1416", color_accent_soft_dark="#2a1416",
        table_even_background_fill="#121722", table_even_background_fill_dark="#121722",
        table_odd_background_fill="#0f131c", table_odd_background_fill_dark="#0f131c",
    )


REPLAY_TEMPLATE = """
<div class="rp">
  <div class="rp-head"><div class="rp-title"></div>
    <div class="rp-controls"><button class="rp-play">▶ Play</button><button class="rp-speed">1×</button>
      <input class="rp-scrub" type="range" min="0" max="1" step="0.1" value="0"><span class="rp-lap"></span></div></div>
  <div class="rp-empty">Run a simulation to watch the race on the track.</div>
  <div class="rp-body"><div class="rp-canvas"><canvas></canvas><div class="rp-tip"></div></div><div class="rp-board"></div></div>
</div>
"""


def build() -> gr.Blocks:
    choices = P.race_choices()
    first = choices[0][1]
    h, m, s, t, pl, sc = race_defaults(first)
    replay_js = (ROOT / "assets/replay.js").read_text(encoding="utf-8")
    replay_css = (ROOT / "assets/replay.css").read_text(encoding="utf-8")
    with gr.Blocks(title="Pit wall — F1 strategy", fill_width=True) as demo:
        gr.HTML('<div id="hero"><h1>PIT <span>WALL</span></h1><p>Race strategy for the rest of the 2026 season '
                "from public timing data (OpenF1) and Pirelli's compound nominations.</p></div>")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=230, elem_id="sidebar"):
                race = gr.Dropdown(choices, value=first, label="Race")
                with gr.Row():
                    hard = gr.Dropdown(C_CHOICES, value=h, label="Hard", min_width=70)
                    medium = gr.Dropdown(C_CHOICES, value=m, label="Medium", min_width=70)
                    soft = gr.Dropdown(C_CHOICES, value=s, label="Soft", min_width=70)
                with gr.Accordion("Conditions", open=True):
                    temp = gr.Slider(10, 60, value=t, step=0.5, label="Track temperature (°C)")
                    pit = gr.Slider(15, 35, value=pl, step=0.5, label="Pit-stop time loss (s)")
                    scp = gr.Slider(0, 0.08, value=sc, step=0.005, label="Safety-car start chance per lap")
                go_btn = gr.Button("Run strategy", variant="primary")
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("Strategy"):
                        kpi_plan = gr.HTML()
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=5, min_width=320):
                                tmap_plot = gr.Plot(show_label=False)
                            with gr.Column(scale=6, min_width=320):
                                rank = gr.Plot(show_label=False)
                        timeline = gr.Plot(show_label=False)
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=6, min_width=320):
                                deg = gr.Plot(show_label=False)
                            with gr.Column(scale=4, min_width=260):
                                notes = gr.Markdown()
                    with gr.Tab("Race simulation"):
                        with gr.Row(equal_height=True):
                            driver = gr.Dropdown(label="Driver", scale=1, min_width=120)
                            strat_a = gr.Dropdown(label="Strategy A", scale=2, min_width=160)
                            strat_b = gr.Dropdown(label="Strategy B", scale=2, min_width=160)
                            sim_btn = gr.Button("Simulate 300 races", variant="primary", scale=1, min_width=150)
                        kpi_sim = gr.HTML()
                        source = gr.Radio(["Simulated race (A)"], value="Simulated race (A)", label="Replay",
                                          container=False)
                        replay = gr.HTML(value="", html_template=REPLAY_TEMPLATE, css_template=replay_css,
                                         js_on_load=replay_js, padding=True)
                        with gr.Row(equal_height=True):
                            with gr.Column(min_width=320):
                                pos_plot = gr.Plot(show_label=False)
                            with gr.Column(min_width=320):
                                gap_plot = gr.Plot(show_label=False)
                        sim_state = gr.State(None)
                    with gr.Tab("Scoreboard"):
                        order_cards, b26, stop_cards = scoreboard()
                        gr.Markdown("### Finishing order, 2026 races\nSpearman correlation between predicted "
                                    "and real finishing order (1 = perfect).")
                        gr.HTML(order_cards)
                        gr.Dataframe(b26, interactive=False)
                        gr.Markdown("### Pit strategy vs what drivers really did\n2026 races, top 5 on the grid. "
                                    "A quarter of real first stops were made under a safety car, which no pre-race "
                                    "plan can foresee.")
                        gr.HTML(stop_cards)
                        gr.Markdown("### Prediction log\nThe plan for each upcoming race as saved after every "
                                    "session (practice, qualifying), before the race, so it can be checked against "
                                    "the real race afterwards.")
                        gr.Dataframe(prediction_log(), interactive=False, wrap=True)
                    with gr.Tab("Method"):
                        gr.Markdown((ROOT / "docs/methodology.md").read_text(encoding="utf-8"),
                                    latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                                      {"left": "$", "right": "$", "display": False}])

        controls = [race, hard, medium, soft, temp, pit, scp]
        plan_out = [kpi_plan, tmap_plot, rank, timeline, deg, notes, strat_a, strat_b]
        race.change(race_defaults, race, [hard, medium, soft, temp, pit, scp]).then(
            run_plan, controls, plan_out).then(field_drivers, race, [driver, source])
        go_btn.click(run_plan, controls, plan_out)
        sim_btn.click(run_scenario, [race, driver, strat_a, strat_b, hard, medium, soft, temp, pit, scp],
                      [kpi_sim, pos_plot, replay, gap_plot, sim_state, source])
        source.input(show_source, [source, sim_state], [replay, gap_plot])
        demo.load(run_plan, controls, plan_out).then(field_drivers, race, [driver, source])
    return demo


FORCE_DARK = "() => { document.body.classList.add('dark'); document.documentElement.classList.add('dark'); }"

if __name__ == "__main__":
    build().launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), css=CSS, theme=theme(),
                   js=FORCE_DARK)
