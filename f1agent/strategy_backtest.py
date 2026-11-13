"""Does the recommended strategy match what the top finishers did?

For each 2026 race, the simulator gets only what was known before the start: degradation
from earlier races (prior) or blended with the weekend's practice, compound offsets, pit
loss and safety-car rate from the circuit's earlier editions (or all earlier races). It
then recommends a strategy, which is compared with the real strategies of the top 10:

  * stops: does the recommended number of stops equal the most common one in the top 10?
  * window: share of the top 10's first stops that fall inside the recommended first-stop
    window (the range of first-stop laps among near-best strategies with that many stops)
  * start compound: does the recommendation start on the compound most of the top 10 used?

Baseline: "do what was most common in all earlier races" (a fixed number of stops and the
average first-stop point as a share of race distance).

Usage:
  python -m f1agent.strategy_backtest --laps data/laps_clean.csv.gz --out results/strategy
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .practice import practice_rates
from .strategy import RaceInputs, compound_offsets, pit_loss, simulate, strategies
from .tyres import FUEL_S_PER_LAP, MIN_RATE_LAPS, fit_predict, measure, race_laps

DRY = ["SOFT", "MEDIUM", "HARD"]
RATE_SD_PRIOR = 0.033  # walk-forward MAE of the prior (docs/test-log.md)
WINDOW_TOLERANCE_S = 3.0


def actual_strategies(laps: pd.DataFrame, session_key: int, top: pd.Series) -> pd.DataFrame:
    """Per driver: stints (compound, first lap) from the race laps."""
    r = laps[(laps["session_key"] == session_key) & laps["compound"].isin(DRY + ["INTERMEDIATE", "WET"])]
    rows = []
    for drv, g in r.groupby("name_acronym"):
        g = g.sort_values("lap_number")
        st = g.groupby("stint").agg(compound=("compound", "first"), start=("lap_number", "min"),
                                    end=("lap_number", "max")).sort_values("start")
        rows.append({"driver": drv, "stops": len(st) - 1,
                     "first_stop": int(st["end"].iloc[0]) if len(st) > 1 else None,
                     "start_compound": st["compound"].iloc[0],
                     "wet": bool(st["compound"].isin(["INTERMEDIATE", "WET"]).any())})
    df = pd.DataFrame(rows)
    return df[df["driver"].isin(top)]


def inputs_for(race_row: pd.Series, laps: pd.DataFrame, rates_all: pd.DataFrame, fp: pd.DataFrame,
               use_practice: bool, fuel: float, n_laps: int | None = None,
               track_temp: float | None = None) -> tuple[RaceInputs, dict]:
    date, circuit = race_row["date"], race_row["circuit"]
    past_laps = laps[(laps["session_name"] == "Race") & (laps["session_date_start"] < date)]
    past_rates = rates_all[(rates_all["date"] < date) & (rates_all["laps"] >= MIN_RATE_LAPS)]
    # Track temperature before the start: the weekend's practice average, else earlier
    # editions at the circuit, else all earlier races (the race's own value would leak).
    fp_t = laps.loc[(laps["meeting_name"] == race_row["meeting_name"]) & (laps["year"] == race_row["year"])
                    & laps["session_name"].str.startswith("Practice"), "track_temperature"].mean()
    hist_t = past_rates.loc[past_rates["circuit"] == circuit, "track_temp"].mean()
    if track_temp is None:
        track_temp = next(v for v in (fp_t, hist_t, past_rates["track_temp"].mean()) if not pd.isna(v))
    target = pd.DataFrame([{**race_row.to_dict(), "compound": c, "track_temp": track_temp,
                            "compound_c": race_row[f"c_{c.lower()}"]} for c in DRY])
    prior = dict(zip(DRY, fit_predict(past_rates, target)))
    rate, sd, source = {}, {}, {}
    for c in DRY:
        rate[c], sd[c], source[c] = prior[c], RATE_SD_PRIOR, "prior"
        if use_practice:
            m = fp[(fp["meeting_name"] == race_row["meeting_name"]) & (fp["year"] == race_row["year"]) & (fp["compound"] == c)]
            if len(m) and m["fp_laps"].iloc[0] >= 10 and not np.isnan(m["fp_rate"].iloc[0]):
                n = m["fp_laps"].iloc[0]
                w = n / (n + 30)
                rate[c] = w * m["fp_rate"].iloc[0] + (1 - w) * prior[c]
                sd[c] = RATE_SD_PRIOR * np.sqrt(1 - w)  # practice narrows the uncertainty
                source[c] = f"blend(fp_laps={int(n)})"
    same = past_laps[past_laps["circuit_short_name"] == circuit]
    pool = same if same["session_key"].nunique() >= 1 else past_laps
    # Compound pace from all earlier races: one or two editions of a circuit gave unstable
    # values (e.g. medium 0.47 s/lap faster than hard at Yas Marina).
    offs = compound_offsets(race_laps(past_laps), rate, fuel)
    # A softer tyre is never slower when new. Race laps can say otherwise (softs run early,
    # in traffic and on full tanks), so estimates that break the order are capped: medium no
    # slower than hard, soft no slower than medium.
    med = min(offs["MEDIUM"][0], offs["HARD"][0])
    offs = {**offs, "MEDIUM": (med, offs["MEDIUM"][1]), "SOFT": (min(offs["SOFT"][0], med), offs["SOFT"][1])}
    pl = pit_loss(pool)
    if np.isnan(pl):
        pl = pit_loss(past_laps)
    neutral = pool.groupby("session_key").apply(lambda g: g.drop_duplicates("lap_number")["neutralised"].mean())
    sc_prob = float(neutral.mean()) / 4.0 if len(neutral) else 0.02
    if n_laps is None:
        n_laps = int(laps.loc[laps["session_key"] == race_row["session_key"], "lap_number"].max())
    inp = RaceInputs(laps=n_laps, rate=rate, rate_sd=sd,
                     offset={c: offs[c][0] for c in DRY}, offset_sd={c: offs[c][1] for c in DRY},
                     pit_loss=pl, sc_lap_prob=sc_prob)
    info = {"rate_source": source, "pit_loss": round(pl, 1), "sc_lap_prob": round(sc_prob, 3),
            "offsets": {c: round(offs[c][0], 3) for c in DRY}, "rates": {c: round(rate[c], 4) for c in DRY},
            "circuit_history_races": int(same["session_key"].nunique()), "track_temp": round(float(track_temp), 1)}
    return inp, info


def recommend(inp: RaceInputs, n_sims: int = 1500) -> tuple[pd.DataFrame, dict]:
    res = simulate(inp, strategies(inp.laps, max_stops=2, step=2), n_sims=n_sims)
    best = res.iloc[0]
    near = res[(res["stops"] == best["stops"]) & (res["median_gap_s"] <= best["median_gap_s"] + WINDOW_TOLERANCE_S)]
    first = [p[0] for p in near["pit_laps"]]
    rec = {"strategy": best["strategy"], "stops": int(best["stops"]), "start_compound": best["compounds"][0],
           "window": (int(min(first)), int(max(first))), "p_fastest": float(best["p_fastest"])}
    return res, rec


def run(laps: pd.DataFrame, races_csv: pd.DataFrame, fuel: float = FUEL_S_PER_LAP) -> tuple[pd.DataFrame, dict]:
    rates_all = measure(laps, fuel)
    fp = practice_rates(laps, fuel)
    sessions = (laps[laps["session_name"] == "Race"]
                .drop_duplicates("session_key")[["session_key", "year", "meeting_name", "circuit_short_name",
                                                  "session_date_start", "hard", "medium", "soft"]]
                .rename(columns={"circuit_short_name": "circuit", "session_date_start": "date"}))
    sessions = sessions[(sessions["year"] >= 2026) & sessions["soft"].notna()].sort_values("date")
    for c in DRY:
        sessions[f"c_{c.lower()}"] = sessions[c.lower()].str.lstrip("C").astype(float)
    rows = []
    for _, s in sessions.iterrows():
        res_csv = races_csv[(races_csv["year"] == s["year"]) & (races_csv["event"] == s["meeting_name"])]
        if res_csv.empty:
            continue
        top = res_csv.sort_values("position").head(10)["driver"]
        actual = actual_strategies(laps, s["session_key"], top)
        if actual.empty or actual["wet"].any():  # wet races are outside a dry-tyre model
            continue
        common_stops = Counter(actual["stops"]).most_common(1)[0][0]
        common_start = Counter(actual["start_compound"]).most_common(1)[0][0]
        firsts = actual["first_stop"].dropna()
        # baseline from earlier races: modal stops and mean first-stop share of distance
        past = rows_baseline(laps, races_csv, s["date"])
        for variant, use_fp in (("prior", False), ("prior+practice", True)):
            inp, info = inputs_for(s, laps, rates_all, fp, use_fp, fuel)
            if any(np.isnan(v) for v in list(inp.offset.values()) + [inp.pit_loss]):
                continue
            res_sim, rec = recommend(inp)
            lo, hi = rec["window"]
            rec_first = res_sim.iloc[0]["pit_laps"][0]
            rows.append({"session_key": s["session_key"], "meeting_name": s["meeting_name"], "variant": variant,
                         "recommended": rec["strategy"], "rec_stops": rec["stops"], "window": f"{lo}-{hi}",
                         "actual_common_stops": common_stops, "stops_match": rec["stops"] == common_stops,
                         "window_hit_share": float(firsts.between(lo, hi).mean()) if len(firsts) else np.nan,
                         "start_match": rec["start_compound"] == common_start[0],
                         "rec_first_stop": rec_first, "actual_first_median": float(firsts.median()) if len(firsts) else np.nan,
                         "first_stop_error": abs(rec_first - firsts.median()) if len(firsts) else np.nan,
                         "window_width": hi - lo + 1,
                         "laps": inp.laps, **{f"info_{k}": json.dumps(v) for k, v in info.items()}})
        if past:
            b_stops, b_share = past
            b_lap = round(b_share * inp.laps)
            rows.append({"session_key": s["session_key"], "meeting_name": s["meeting_name"], "variant": "baseline",
                         "recommended": f"{b_stops} stop(s), first at lap {b_lap}", "rec_stops": b_stops,
                         "window": f"{b_lap - 3}-{b_lap + 3}", "actual_common_stops": common_stops,
                         "stops_match": b_stops == common_stops,
                         "window_hit_share": float(firsts.between(b_lap - 3, b_lap + 3).mean()) if len(firsts) else np.nan,
                         "start_match": np.nan, "rec_first_stop": b_lap,
                         "actual_first_median": float(firsts.median()) if len(firsts) else np.nan,
                         "first_stop_error": abs(b_lap - firsts.median()) if len(firsts) else np.nan,
                         "window_width": 7, "laps": inp.laps})
    per = pd.DataFrame(rows)
    summary = {"races": int(per["session_key"].nunique())}
    for v, g in per.groupby("variant"):
        summary[v] = {"stops_match": round(float(g["stops_match"].mean()), 3),
                      "first_stop_error_laps_median": round(float(g["first_stop_error"].median()), 1),
                      "first_stop_error_laps_mean": round(float(g["first_stop_error"].mean()), 1),
                      "window_hit_share": round(float(g["window_hit_share"].mean()), 3),
                      "window_width_mean": round(float(g["window_width"].mean()), 1)}
        # Start compound is not reported: the simulator's race time does not depend on the
        # order of the stints (fuel affects all strategies alike), so the order it picks
        # carries no information.
    return per, summary


def rows_baseline(laps: pd.DataFrame, races_csv: pd.DataFrame, date: str):
    """Modal stops and mean first-stop share among top-10 finishers of all earlier dry races."""
    stops, shares = [], []
    for key, g in laps[(laps["session_name"] == "Race") & (laps["session_date_start"] < date)].groupby("session_key"):
        meet, year = g["meeting_name"].iloc[0], g["year"].iloc[0]
        res = races_csv[(races_csv["year"] == year) & (races_csv["event"] == meet)]
        if res.empty:
            continue
        a = actual_strategies(laps, key, res.sort_values("position").head(10)["driver"])
        if a.empty or a["wet"].any():
            continue
        n = g["lap_number"].max()
        stops += a["stops"].tolist()
        shares += (a["first_stop"].dropna() / n).tolist()
    if not stops:
        return None
    return Counter(stops).most_common(1)[0][0], float(np.mean(shares))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--laps", default="data/laps_clean.csv.gz")
    p.add_argument("--races", default="data/races.csv")
    p.add_argument("--out", default="results/strategy")
    args = p.parse_args(argv)
    per, summary = run(pd.read_csv(args.laps), pd.read_csv(args.races))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_race.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(per[["meeting_name", "variant", "recommended", "rec_first_stop", "actual_first_median",
               "actual_common_stops", "stops_match"]].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
