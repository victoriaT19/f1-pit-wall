"""Pit wall: an F1 race-strategy dashboard (Gradio).

Tabs: strategy plan for a race (single car, Monte Carlo), whole-field race simulation
with a lap-by-lap replay, the scoreboard of every backtest, and the methodology.

Run locally:  python app.py   (then open http://127.0.0.1:7860)
"""

from __future__ import annotations

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
COMPOUND_COLOUR = {"SOFT": "#E8002D", "MEDIUM": "#FFC906", "HARD": "#A8ADB4",
                   "INTERMEDIATE": "#43B02A", "WET": "#0067AD"}
A_COLOUR, B_COLOUR = "#3671C6", "#FF8000"
INK = "#8A8F98"
GRID = "rgba(138,143,152,0.18)"
C_CHOICES = [f"C{i}" for i in range(1, 7)]


def _layout(fig: go.Figure, title: str, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)), height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK, size=12), margin=dict(l=10, r=10, t=48, b=10),
                      hoverlabel=dict(font_size=12), legend=dict(orientation="h", y=1.02, x=1, xanchor="right",
                                                                 yanchor="bottom"), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


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
    return (summary_md(key, info, inp, res, (hard, medium, soft)), map_fig(key), ranking_fig(res), timeline_fig(key, res, inp.laps),
            degradation_fig(inp), gr.update(choices=options, value=res.iloc[0]["strategy"]),
            gr.update(choices=options, value=alt))


def _alternative(res: pd.DataFrame) -> str:
    other = res[res["stops"] != res.iloc[0]["stops"]]
    return (other.iloc[0] if len(other) else res.iloc[1])["strategy"]


def summary_md(key, info, inp, res, comps) -> str:
    best = res.iloc[0]
    alt = res[res["stops"] != best["stops"]].iloc[0]
    src = {c: ("practice + model" if s.startswith("blend") else "model fitted on earlier races")
           for c, s in info["rate_source"].items()}
    status = ("⚠️ Compounds **provisional** (last year's nomination) until Pirelli announces them — edit them on the left."
              if info["compounds_status"].startswith("provisional") else "Compounds: official Pirelli nomination.")
    stints = None if info["upcoming"] else P.real_stints(int(key.split(":")[1]))
    wet = stints is not None and stints["compound"].isin(["INTERMEDIATE", "WET"]).any()
    lines = [
        f"### Recommended: **{best['strategy']}**",
        f"{best['stops']} stop(s), pit on lap {', '.join(map(str, best['pit_laps']))}. "
        f"Best {alt['stops']}-stop option (**{alt['strategy']}**) is {alt['median_gap_s'] - best['median_gap_s']:+.1f} s "
        f"in the median race.",
        "",
        _flatness(res),
        "",
        f"Hard {comps[0]} · Medium {comps[1]} · Soft {comps[2]} · {inp.laps} laps · "
        f"pit loss {inp.pit_loss:.1f} s · safety-car start chance {inp.sc_lap_prob * 100:.1f}% per lap",
        "",
        f"Degradation source — soft: {src['SOFT']}, medium: {src['MEDIUM']}, hard: {src['HARD']}.",
        "",
        status,
    ]
    if wet:
        lines += ["", "🌧️ This race was wet: the dry-strategy plan does not apply to the real race."]
    lines += ["", "<small>Backtest on 2026: the plan's first stop is off by a median of 8 laps, like the "
                  "historical baseline; a quarter of real stops come from safety cars. Use it to compare "
                  "options, not as a forecast. See the Scoreboard tab.</small>"]
    return "\n".join(lines)


def _flatness(res: pd.DataFrame, tol: float = 2.0) -> str:
    near = res[res["median_gap_s"] <= res.iloc[0]["median_gap_s"] + tol]
    first = sorted({p[0] for p in near["pit_laps"]})
    if len(near) >= 10:
        return (f"{len(near)} strategies are within {tol:.0f} s of the best (first stop anywhere from lap "
                f"{first[0]} to {first[-1]}): with this little wear, timing barely matters — react to safety cars.")
    return f"{len(near)} strategies within {tol:.0f} s of the best; first stop between laps {first[0]} and {first[-1]}."


def ranking_fig(res: pd.DataFrame, top: int = 12) -> go.Figure:
    r = res.head(top).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=r["median_gap_s"], y=r["strategy"], mode="markers",
        marker=dict(size=10, color=A_COLOUR, line=dict(width=2, color="white")),
        error_x=dict(type="data", symmetric=False, array=r["p90_gap_s"] - r["median_gap_s"],
                     arrayminus=r["median_gap_s"] - r["p10_gap_s"], color=A_COLOUR, thickness=2, width=0),
        customdata=np.c_[r["p10_gap_s"], r["p90_gap_s"], r["p_fastest"] * 100],
        hovertemplate="<b>%{y}</b><br>median %{x:.1f} s behind the best<br>"
                      "10–90%%: %{customdata[0]:.1f} – %{customdata[1]:.1f} s<br>fastest in %{customdata[2]:.0f}%% "
                      "of races<extra></extra>"))
    fig = _layout(fig, "Time lost vs the best strategy in each simulated race (median, 10–90% range)", 420)
    fig.update_xaxes(title="seconds", rangemode="tozero")
    return fig


def timeline_fig(key: str, res: pd.DataFrame, laps: int) -> go.Figure:
    rows = [(f"Plan #{i + 1}", P.parse_strategy(s)) for i, s in enumerate(res["strategy"].head(5))]
    real = None if key.startswith("next") else P.real_stints(int(key.split(":")[1]))
    fig = go.Figure()
    seen = set()

    def bar(label, comp, start, length, note):
        fig.add_trace(go.Bar(
            y=[label], x=[length], base=[start - 1], orientation="h", name=comp.title(),
            marker=dict(color=COMPOUND_COLOUR.get(comp, INK), line=dict(color="rgba(0,0,0,0)", width=0)),
            legendgroup=comp, showlegend=comp not in seen, width=0.62,
            hovertemplate=f"<b>{label}</b><br>{comp.title()}: laps {start}–{start + length - 1}{note}<extra></extra>"))
        seen.add(comp)

    labels = []
    for label, strat in rows:
        lap = 1
        for comp, n in strat:
            bar(label, comp, lap, n, "")
            lap += n
        labels.append(label)
    if real is not None and len(real):
        for drv, g in real.groupby("driver", sort=False):
            lab = f"P{int(g['rank'].iloc[0]) + 1} {drv} (real)"
            for _, s in g.iterrows():
                bar(lab, s["compound"], int(s["start"]), int(s["end"] - s["start"] + 1), "")
            labels.append(lab)
    fig = _layout(fig, "Stints: top plans" + (" vs what the top 10 really did" if real is not None else ""),
                  120 + 26 * len(labels), barmode="overlay", bargap=0.35)
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1], gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="lap", range=[0, laps])
    # 2px surface gap between consecutive stints
    fig.update_traces(marker_line_width=2, marker_line_color="rgba(255,255,255,0.9)")
    return fig


def degradation_fig(inp) -> go.Figure:
    d = P.degradation_curves(inp)
    fig = go.Figure()
    for c in ("SOFT", "MEDIUM", "HARD"):
        g = d[d["compound"] == c]
        col = COMPOUND_COLOUR[c]
        for edge in ("high", "low"):  # uncertainty as thin dashed edges: three filled bands would hide the lines
            fig.add_trace(go.Scatter(x=g["age"], y=g[edge], mode="lines", line=dict(color=_alpha(col, 0.45), width=1,
                                     dash="dot"), hoverinfo="skip", showlegend=False, legendgroup=c))
        fig.add_trace(go.Scatter(x=g["age"], y=g["delta"], mode="lines", name=c.title(), legendgroup=c,
                                 line=dict(color=col, width=2),
                                 hovertemplate=f"{c.title()} · age %{{x}} laps: %{{y:+.2f}} s/lap<extra></extra>"))
        last = g.iloc[-1]
        fig.add_annotation(x=last["age"], y=last["delta"], text=c.title(), showarrow=False, xanchor="left",
                           xshift=6, font=dict(color=INK, size=11))
    fig = _layout(fig, "Lap time vs a new hard tyre, by tyre age (dotted: ±1 sd of the wear rate)", 360,
                  hovermode="x unified")
    fig.update_xaxes(title="tyre age (laps)")
    fig.update_yaxes(title="s per lap (lower = faster)")
    return fig


def _alpha(hex_colour: str, a: float) -> str:
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# ── Race simulation tab ─────────────────────────────────────────────────────


def field_drivers(key: str):
    drivers, _, real = P.field(key)
    note = "Real starting grid." if real else "Grid **estimated** from recent race pace (qualifying not run yet)."
    return gr.update(choices=drivers, value=drivers[min(2, len(drivers) - 1)]), note


def run_scenario(key, driver, strat_a, strat_b, hard, medium, soft, temp, pit, sc):
    comps = tuple(float(c.lstrip("C")) for c in (hard, medium, soft))
    out = P.scenario(key, driver, strat_a, strat_b, n_sims=300, compounds=comps, track_temp=float(temp),
                     pit_loss=float(pit), sc_lap_prob=float(sc))
    a, b = out["a"]["positions"], out["b"]["positions"]
    txt = (f"**{driver}** starts P{out['grid_pos']}. Expected finish: **A {a.mean():.1f}** vs **B {b.mean():.1f}**. "
           f"In the same simulated conditions, A finishes ahead of where B would in {np.mean(a < b) * 100:.0f}% "
           f"of races, level in {np.mean(a == b) * 100:.0f}%, behind in {np.mean(a > b) * 100:.0f}%. "
           f"Rivals run the top plan with stops spread ±3 laps.")
    real_fig = None
    if key.startswith("past"):
        real_fig = trace_fig(P.real_trace(int(key.split(":")[1])), driver, "Real race: gap to the leader")
    sim_trace = pd.DataFrame(out["a"]["trace"], index=np.arange(1, out["laps"] + 1), columns=out["drivers"])
    replay = replay_fig(key, sim_trace, driver, f"On-track replay, one simulated race ({driver} on A): drag the lap slider")
    return (txt, positions_fig(a, b, len(out["drivers"])), replay,
            trace_fig(sim_trace, driver, f"One simulated race, {driver} on A: gap to the leader"), real_fig)


def real_replay(key, driver):
    if not key.startswith("past"):
        return None
    return replay_fig(key, P.real_trace(int(key.split(":")[1])), driver or "", "On-track replay of the real race: drag the lap slider")


# ── Track map ───────────────────────────────────────────────────────────────

SPEED_SCALE = [[0.0, "#1F3A73"], [0.5, "#4F86D9"], [1.0, "#CFE2FF"]]  # one hue: dark = slow, light = fast


def _track_base(fig: go.Figure, tmap: pd.DataFrame, colour_speed: bool) -> None:
    x, y = np.r_[tmap["x"], tmap["x"].iloc[0]], np.r_[tmap["y"], tmap["y"].iloc[0]]
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="rgba(138,143,152,0.45)", width=10),
                             hoverinfo="skip", showlegend=False))
    if colour_speed:
        fig.add_trace(go.Scatter(
            x=tmap["x"], y=tmap["y"], mode="markers", showlegend=False,
            marker=dict(size=6, color=tmap["speed"], colorscale=SPEED_SCALE,
                        colorbar=dict(title=dict(text="km/h", side="right"), thickness=10, len=0.6, outlinewidth=0)),
            customdata=tmap["t"],
            hovertemplate="%{marker.color:.0f} km/h · %{customdata:.1f} s into the lap<extra></extra>"))
    # start/finish line and sector boundaries, from the lap's sector times
    marks = [("Start / finish", 0.0)]
    for name, t in (("S2", tmap["s1"].iloc[0]), ("S3", tmap["s1"].iloc[0] + tmap["s2"].iloc[0])):
        if pd.notna(t):
            marks.append((name, float(t)))
    for name, t in marks:
        i = int(np.clip(np.searchsorted(tmap["t"].to_numpy(), t), 0, len(tmap) - 1))
        fig.add_annotation(x=tmap["x"].iloc[i], y=tmap["y"].iloc[i], text=name, showarrow=True, arrowhead=0,
                           arrowcolor=INK, ax=24, ay=-24, font=dict(color=INK, size=11))


def _map_layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig = _layout(fig, title, height, showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig


def map_fig(key: str):
    tmap = P.track_map(key)
    if tmap is None:
        return None
    fig = go.Figure()
    _track_base(fig, tmap, colour_speed=True)
    title = (f"{P.race_row(key)[0]['circuit']} · speed on a fast lap ({tmap['driver'].iloc[0]}, "
             f"{tmap['lap_duration'].iloc[0]:.1f} s)")
    return _map_layout(fig, title, 420)


def replay_fig(key: str, trace: pd.DataFrame, focus: str, title: str):
    """Cars on the map lap by lap, placed from their gap to the leader; drag the slider."""
    tmap = P.track_map(key)
    if tmap is None:
        return None
    colours = P.store().colours
    fig = go.Figure()
    _track_base(fig, tmap, colour_speed=False)

    def cars(lap):
        pos = P.on_track(trace.loc[lap], tmap)
        order = pos.sort_values("gap")
        rank = {d: i + 1 for i, d in enumerate(order["driver"])}
        show = pos["driver"].isin([focus, order["driver"].iloc[0]])
        return go.Scatter(
            x=pos["x"], y=pos["y"], mode="markers+text",
            text=np.where(show, pos["driver"], ""), textposition="top center", textfont=dict(color=INK, size=12),
            marker=dict(size=np.where(pos["driver"] == focus, 16, 11),
                        color=[colours.get(d, INK) for d in pos["driver"]], line=dict(width=2, color="white")),
            customdata=np.c_[pos["driver"], [rank[d] for d in pos["driver"]], pos["gap"].round(1)],
            hovertemplate="<b>%{customdata[0]}</b> P%{customdata[1]} · +%{customdata[2]} s<extra></extra>")

    # Gradio drops Plotly animation frames, so every lap is its own trace and the slider
    # switches which one is visible (restyle works without frames).
    laps = list(trace.index)
    n_base = len(fig.data)
    for i, l in enumerate(laps):
        tr = cars(l)
        tr.visible = i == 0
        fig.add_trace(tr)
    steps = []
    for i, l in enumerate(laps):
        vis = [True] * n_base + [j == i for j in range(len(laps))]
        steps.append(dict(label=str(l), method="restyle", args=[{"visible": vis}]))
    fig = _map_layout(fig, title, 480)
    fig.update_layout(
        sliders=[dict(active=0, x=0, len=1, y=0, yanchor="top", pad=dict(t=10),
                      currentvalue=dict(prefix="Lap ", font=dict(color=INK)), steps=steps)],
        margin=dict(l=10, r=10, t=48, b=70))
    return fig


def positions_fig(a: np.ndarray, b: np.ndarray, n: int) -> go.Figure:
    pos = np.arange(1, n + 1)
    fig = go.Figure()
    for arr, name, col in ((a, "Strategy A", A_COLOUR), (b, "Strategy B", B_COLOUR)):
        share = np.array([(arr == p).mean() * 100 for p in pos])
        fig.add_trace(go.Bar(x=pos, y=share, name=name, marker=dict(color=col, line=dict(width=2, color="white")),
                             hovertemplate=f"{name}: P%{{x}} in %{{y:.0f}}%% of races<extra></extra>"))
    fig = _layout(fig, "Finishing position across simulated races", 340, barmode="group", bargap=0.25)
    fig.update_xaxes(title="position", dtick=1)
    fig.update_yaxes(title="% of races")
    return fig


def trace_fig(trace: pd.DataFrame, focus: str, title: str) -> go.Figure:
    colours = P.store().colours
    fig = go.Figure()
    for d in trace.columns:
        if d == focus:
            continue
        fig.add_trace(go.Scatter(x=trace.index, y=trace[d], mode="lines", name=d, showlegend=False,
                                 line=dict(color="rgba(138,143,152,0.35)", width=1),
                                 hovertemplate=f"{d} · lap %{{x}}: +%{{y:.1f}} s<extra></extra>"))
    if focus in trace.columns:
        col = colours.get(focus, A_COLOUR)
        fig.add_trace(go.Scatter(x=trace.index, y=trace[focus], mode="lines", name=focus,
                                 line=dict(color=col, width=3),
                                 hovertemplate=f"<b>{focus}</b> · lap %{{x}}: +%{{y:.1f}} s<extra></extra>"))
        last = trace[focus].dropna()
        if len(last):
            fig.add_annotation(x=last.index[-1], y=last.iloc[-1], text=focus, showarrow=False, xanchor="left",
                               xshift=6, font=dict(color=INK, size=12))
    cap = np.nanpercentile(trace.values, 90) * 1.15 if np.isfinite(trace.values).any() else 60
    fig = _layout(fig, title, 380, showlegend=False)
    fig.update_yaxes(title="seconds behind the leader", autorange=False, range=[cap, -2])
    fig.update_xaxes(title="lap")
    return fig


# ── Scoreboard ──────────────────────────────────────────────────────────────


def scoreboard():
    bt = pd.read_csv(ROOT / "results/backtest/per_race.csv")
    b26 = bt[bt["year"] == 2026].pivot_table(index=["round", "event"], columns="method", values="spearman").reset_index()
    cols = [c for c in ("model", "grid", "championship", "form") if c in b26.columns]
    b26 = b26[["round", "event"] + cols].round(3)
    means = b26[cols].mean().round(3)
    order_md = ("Spearman correlation between predicted and real finishing order (1 = perfect). Mean over 2026: "
                + ", ".join(f"**{c} {means[c]:.3f}**" for c in cols) + ".")
    rs = json.loads((ROOT / "results/racesim/summary.json").read_text(encoding="utf-8"))
    strat = pd.DataFrame([
        {"method": "Whole-field simulation", "first-stop error (median laps)": rs["multi"]["first_stop_error_median"],
         "stops match": rs["multi"]["stops_match"]},
        {"method": "Single-car simulation", "first-stop error (median laps)": rs["single"]["first_stop_error_median"],
         "stops match": rs["single"]["stops_match"]},
        {"method": "Historical baseline", "first-stop error (median laps)": rs["baseline"]["first_stop_error_median"],
         "stops match": rs["baseline"]["stops_match"]},
    ])
    strat["stops match"] = (strat["stops match"] * 100).round(0).astype(int).astype(str) + "%"
    return order_md, b26, strat


# ── App ─────────────────────────────────────────────────────────────────────

CSS = """
.gradio-container {max-width: 1280px !important}
#title h1 {margin-bottom: 0}
#title p {margin-top: 4px; opacity: .75}
.block:focus-within, .form:focus-within {z-index: 1000 !important; position: relative}
"""


def build() -> gr.Blocks:
    choices = P.race_choices()
    first = choices[0][1]
    h, m, s, t, pl, sc = race_defaults(first)
    with gr.Blocks(title="Pit wall — F1 strategy") as demo:
        gr.Markdown("# 🏁 Pit wall\nRace strategy for the rest of the 2026 season, from public timing data "
                    "(OpenF1) and Pirelli's compound nominations. Every number is measured from earlier races.",
                    elem_id="title")
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                race = gr.Dropdown(choices, value=first, label="Race")
                with gr.Row():
                    hard = gr.Dropdown(C_CHOICES, value=h, label="Hard", min_width=70)
                    medium = gr.Dropdown(C_CHOICES, value=m, label="Medium", min_width=70)
                    soft = gr.Dropdown(C_CHOICES, value=s, label="Soft", min_width=70)
                temp = gr.Slider(10, 60, value=t, step=0.5, label="Track temperature (°C)")
                pit = gr.Slider(15, 35, value=pl, step=0.5, label="Pit-stop time loss (s)")
                scp = gr.Slider(0, 0.08, value=sc, step=0.005, label="Safety-car start chance per lap")
                go_btn = gr.Button("Run strategy", variant="primary")
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Strategy"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                summary = gr.Markdown()
                            with gr.Column(scale=1):
                                tmap_plot = gr.Plot(show_label=False)
                        rank = gr.Plot(show_label=False)
                        timeline = gr.Plot(show_label=False)
                        deg = gr.Plot(show_label=False)
                    with gr.Tab("Race simulation"):
                        grid_note = gr.Markdown()
                        with gr.Row():
                            driver = gr.Dropdown(label="Driver")
                            strat_a = gr.Dropdown(label="Strategy A")
                            strat_b = gr.Dropdown(label="Strategy B")
                        sim_btn = gr.Button("Simulate the race (300 runs)", variant="primary")
                        sim_txt = gr.Markdown()
                        pos_plot = gr.Plot(show_label=False)
                        replay_plot = gr.Plot(show_label=False)
                        sim_trace = gr.Plot(show_label=False)
                        real_trace = gr.Plot(show_label=False)
                        real_btn = gr.Button("Replay the real race on the map (past races)")
                        real_replay_plot = gr.Plot(show_label=False)
                    with gr.Tab("Scoreboard"):
                        order_md, b26, strat_tab = scoreboard()
                        gr.Markdown("### Finishing order, 2026 races\n" + order_md)
                        gr.Dataframe(b26, interactive=False)
                        gr.Markdown("### Pit strategy vs what drivers really did (2026, top 5 on the grid)")
                        gr.Dataframe(strat_tab, interactive=False)
                        gr.Markdown("A quarter of real first stops were made under a safety car, which no "
                                    "pre-race plan can foresee. Details in `docs/test-log.md`.")
                    with gr.Tab("Method"):
                        gr.Markdown((ROOT / "docs/methodology.md").read_text(encoding="utf-8"),
                                    latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                                      {"left": "$", "right": "$", "display": False}])

        controls = [race, hard, medium, soft, temp, pit, scp]
        plan_out = [summary, tmap_plot, rank, timeline, deg, strat_a, strat_b]
        race.change(race_defaults, race, [hard, medium, soft, temp, pit, scp]).then(
            run_plan, controls, plan_out).then(field_drivers, race, [driver, grid_note])
        go_btn.click(run_plan, controls, plan_out)
        sim_btn.click(run_scenario, [race, driver, strat_a, strat_b, hard, medium, soft, temp, pit, scp],
                      [sim_txt, pos_plot, replay_plot, sim_trace, real_trace])
        real_btn.click(real_replay, [race, driver], real_replay_plot)
        demo.load(run_plan, controls, plan_out).then(field_drivers, race, [driver, grid_note])
    return demo


if __name__ == "__main__":
    build().launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), css=CSS,
                   theme=gr.themes.Soft(primary_hue="red"))
