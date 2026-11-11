# Test log

## 2026-11-08: data and walk-forward backtest

**Environment:** Windows 11, CPU, Python 3.13, FastF1 3.9.2, scikit-learn 1.9.1.

### Unit tests: 9 passed

`tests/test_features.py`: changing one race's result leaves its own features and every
earlier race's features unchanged, while the next race sees it (no leakage); the first
race has no history; season form resets each year; championship rank matches cumulative
points; metrics give Spearman 1/-1, winner and podium hits for perfect/reversed orders;
sign test; win/loss counting in the summary.

### Data

`python -m f1agent.data --years 2023 2024 2025 2026`: 90 Grands Prix (2023: 22, 2024: 24,
2025: 24, 2026: 20, up to São Paulo on 8 November), 1,838 driver rows. The F1 data APIs
allow about 500 calls per hour; the first run stopped at the limit and the second one
resumed from the saved table. 2025 Miami has no qualifying times in the source (all 20
missing after a reload); 25 other rows miss a qualifying time (no valid lap).

### Backtest (`python -m f1agent.backtest --from-year 2024`)

68 races (2024 to 2026 São Paulo); each predicted by a model trained only on earlier races
(from 2023). Model: HistGradientBoostingRegressor on 11 pre-race features.

| Method | Spearman | MAE (positions) | Winner | Podium | Top 10 |
|---|---|---|---|---|---|
| Model (position) | 0.691 | 3.31 | 44% | 58% | 79% |
| Model (change from grid) | 0.690 | 3.32 | 40% | 62% | 79% |
| Starting grid | 0.677 | 3.25 | 60% | 68% | 79% |
| Championship order | 0.607 | 3.85 | 34% | 58% | 75% |
| Recent form (last 5) | 0.594 | 3.95 | 22% | 50% | 73% |

| Spearman, per race | model better | baseline better | ties | sign test p |
|---|---|---|---|---|
| Model (position) vs starting grid | 36 | 32 | 0 | 0.716 |
| Model (change from grid) vs starting grid | 31 | 35 | 2 | 0.712 |
| Model (position) vs championship order | 47 | 21 | 0 | 0.002 |
| Model (position) vs recent form | 54 | 13 | 1 | < 0.001 |

Spearman by part of the season:

| | rounds 1-5: model / grid | rounds 6+: model / grid |
|---|---|---|
| 2024 | 0.686 / 0.760 | 0.724 / 0.722 |
| 2025 | 0.727 / 0.669 | 0.654 / 0.647 |
| 2026 (new regulations) | 0.608 / 0.611 | 0.712 / 0.656 |

Findings:
- The model beats the championship order and recent form, but not the starting grid
  overall (36-32 races, p = 0.72), and it picks the winner less often (44% vs 60%).
- In 2026, with new technical regulations, the model only matched the grid in the first
  five races and pulled ahead afterwards, once the season had its own results to learn from.
- Predicting the change from the grid instead of the position did not help. Two circuit
  features (overtaking index, driver's history at the circuit) moved the overall Spearman
  from 0.704 to 0.707 on the first 37 races.

## 2026-11-08: agent

### Choosing the model (Gemini API, free tier)

One question ("Quem venceu o GP de São Paulo de 2026?") with one tool, through the
OpenAI-compatible endpoint: `gemini-flash-latest` and `gemini-3.8-flash` only returned
timeouts and HTTP 503 in 4 tries each; `gemini-2.5-flash` returned 404; `gemma-4-26b-a4b-it`
and `gemma-4-31b-it` answered in text and ignored the tool ("não tenho informações...").
Through the native `generateContent` API with `functionDeclarations`, `gemma-4-31b-it`
called `race_result(year=2026, event="São Paulo")` in 4.9 s (after one HTTP 500);
`gemma-4-26b-a4b-it` still answered without the tool. The agent uses gemma-4-31b-it on the
native API, with retries on 429/500/503.

### Unit tests: 19 passed

`tests/test_agent.py` (no API): GP name aliases; tools agree with the table (season points
= sum of race points = standings entry); a past-race prediction is a permutation and
carries the actual result; a future race uses a provisional grid unless one is given;
tool errors are returned, not raised; NaN becomes null; tool schemas convert to Gemini
types; the loop with a fake model executes one tool call and sends the result back with
its id; the loop stops after `max_steps`; answer checkers.

### Evaluation: 16 questions with answers computed from the data

`python -m f1agent.agent_eval` (results in `results/agent_eval/`):

| | first run | after fixing the checker | after re-running the API failure |
|---|---|---|---|
| Correct answers | 14/16 | 15/16 | **16/16** |
| Expected tool called | 16/16 | 16/16 | 16/16 |
| Runs ended by an API error | 1 | 1 | 0 |

The two first-run failures were not agent errors:
- `model_accuracy`: the answer ("44,12% ... 60,29%") was right; the checker looked for
  44 and 60 with a 0.05 tolerance. Percentages now accept half a point. Raw first run kept
  in `answers_first_run.jsonl`.
- `leader26`: HTTP 500 on all 6 tries; re-run alone with `--only leader26`, correct.

Per question: 1.9 LLM calls, about 1,900 prompt tokens and 54 output tokens on average;
median 65 s, almost all of it waiting on retries of the free tier's HTTP 500/503.

### Default thinking vs minimal thinking

Same 16 questions, back to back, gemma-4-31b-it on the free tier. The agent now uses the
model's own (default) thinking; `--thinking minimal` is optional.

| | default thinking | thinking = minimal |
|---|---|---|
| Correct answers | 15/16 | 16/16 |
| Expected tool called | 15/16 | 16/16 |
| Runs ended by an API error | 1 (HTTP 503 on all 6 tries) | 0 |
| Retried API errors (HTTP 500/503) | 37 (503: 24, 500: 13) | 24 (503: 12, 500: 12) |
| Questions that hit at least one | 12/16 | 14/16 |
| Thinking tokens per question | 137 | 0 |
| Median seconds per question | 129 | 74 |

Every answer the API returned was correct in both modes; the only miss was a request that
failed six times. Thinking does not change correctness on these questions. The
HTTP 500/503 errors come from the free tier and hit most questions in both modes; they,
not the model, are what make an answer take one or two minutes.

## 2026-11-08: tyre model

### Data

OpenF1 (free historical API), race sessions 2024-2026 up to São Paulo: 68 races, 76,474
laps; 61,553 clean race laps (not lap 1, not pit in/out, not under SC/VSC/red flag, no
rain, within 107% of the driver's median). Pirelli nominations (C1-C6 per race) for all 68
races from each race's Wikipedia "Tyre choices" text, in `data/pirelli_compounds.csv`
with the source of every row. The 4,212 race laps without a C-number are all on
intermediate or wet tyres (plus 4 laps labelled UNKNOWN), which the dry model excludes.
2026 detail: the April Bahrain and Saudi Arabian rounds were cancelled; the "Bahrain Grand
Prix" of 2026 ran at Sepang, matched by name and year.

A first download with practice sessions was too slow (whole tables rewritten after every
session) and an interrupted run left a truncated gzip file; table writes are now atomic
(temporary file, then rename) and practice is downloaded only when needed.

### Unit tests: 26 passed (7 new)

`tests/test_tyres.py`: the within-stint method recovers the exact degradation rates of a
synthetic race (0.12 and 0.05 s/lap) and is within 0.02 s/lap with 0.3 s of lap noise; a
fuel assumption 0.02 s/lap too low lowers every rate by exactly 0.02; stint error is zero
with the true rates; stint offsets do not affect the slope; safety-car and VSC lap ranges;
Pirelli join across name variants ("Barcelona-Catalunya" vs "Barcelona").

### First measurements (2024, first four races)

Bahrain 0.09-0.12 s/lap, Jeddah about 0 (track evolution offsets wear), Suzuka C3 0.16 vs
0.08 for the harder compounds: abrasive tracks wear more, and within a race the softer
compound wears faster.

### Predicting a race's degradation from earlier races (walk-forward)

Rate per race and compound (s/lap), only race-compounds with at least 40 clean laps;
56 races, 135 race-compounds.

| Predictor | MAE, all | MAE, 2026 |
|---|---|---|
| Ridge: real compound + track temperature + 2026 flag | **0.0333** | **0.0315** |
| Average of the same label (SOFT/MEDIUM/HARD) | 0.0337 | 0.0376 |
| Average of the same real compound (C1-C6) | 0.0349 | 0.0353 |
| Same circuit's earlier rate | 0.0413 | 0.0349 |

Row by row, the ridge model beats the label average 64 times and loses 71. The first
version, a gradient-boosted model with the circuit's history, was worse than the label
average (0.073 vs 0.059 on all rows, including thinly measured ones); it fitted noise on
about 160 examples and was replaced.

Across races, the real compound alone barely predicts the rate (correlation 0.04): Pirelli
picks softer compounds for tracks that wear tyres less, so compound and track are
confounded. Before a weekend, about 0.03 s/lap is the error to expect; measuring the
weekend's own practice long runs is the next step to test.

The comparison between predictors does not depend on the fuel assumption (0.035, 0.055
and 0.075 s/lap give the same ranking); only the absolute rates shift.

## 2026-11-08: practice long runs and the strategy simulator

### Practice data

OpenF1 practice sessions (FP1-FP3) for the 2026 races: 48 sessions. The first download
attempt rewrote every combined table after each session and slowed to about one session
per minute; sessions now go to per-session files merged at the end (about 4.5 per minute).

### Do practice long runs predict race degradation? (2026, 19 races, 43 race-compounds)

First attempt, with the race's clean-lap filter (within 107% of the driver's median):
practice rates were off by 0.42 s/lap. In practice the median includes cool-down laps, so
laps of 80-85 s passed at a track where race pace was 71 s (seen in Verstappen's Austria
FP2). Long runs are now laps within 3.5% of their own stint's fastest lap, in stints of at
least 5 such laps.

| Rate predictor (s/lap, MAE vs the race) | value |
|---|---|
| Prior (ridge from earlier races) | **0.0296** |
| Practice long runs only | 0.0881 |
| Blend, practice weight n/(n + 10) | 0.0762 |
| Blend, n/(n + 30) | 0.0630 |
| Blend, n/(n + 100) | 0.0447 |

The blend with n0 = 30 beats the prior in 10 of 43 race-compounds. Practice rates correlate
0.26 with race rates (prior: 0.40), are 0.048 s/lap higher on average and twice as spread
(sd 0.103 vs 0.046). Ordering compounds within a weekend: practice 20/32 pairs right, prior
21/32. With public lap times, practice long runs did not improve the forecast in 2026
(different fuel loads, engine modes and track evolution in practice are not observable in
this data).

### Simulator unit tests: 8 passed (34 in total)

`tests/test_strategy.py`: every generated strategy covers the race distance, uses two dry
compounds and respects the minimum stint; simulated gaps equal the closed-form arithmetic
when there is no uncertainty; high degradation and cheap pit stops lead to more stops;
safety cars make stops cheaper; uncertainty widens the 10-90% gap range.

### Strategy backtest (2026 dry races, 17 races)

Each race simulated with only pre-race information (rates from earlier races, optionally
blended with practice; offsets, pit loss and safety-car rate from earlier editions; track
temperature from the weekend's practice or earlier editions, never the race's own).
Compared with the top 10's real strategies. Baseline: the most common number of stops in
all earlier dry races and their mean first-stop point.

| | stops match | first-stop error, median (mean) | first-stop window width |
|---|---|---|---|
| Baseline | 53% | **6.5 laps** (8.1) | 7 laps |
| Simulator, prior | **65%** | 7.0 laps (10.6) | 23.9 laps |
| Simulator, prior + practice | 47% | 8.0 laps (14.4) | 29.2 laps |

- The window "hit rate" (share of the top 10's first stops inside the recommended window)
  was 69% for the simulator against 18% for the baseline, but the simulator's windows are
  more than three times wider, so that comparison is not fair and is not used.
- The start compound is not evaluated: the simulator's race time does not depend on the
  order of stints (fuel is the same for every strategy and drops out), so the order it
  picks carries no information.
- The simulator gets the number of stops right more often (65% vs 53%, 17 races; not
  significant), but places the first stop no better than the simple baseline. The
  near-best strategies span about 24 laps: with rates known to about 0.03 s/lap, the
  optimum is flat. What the simulator leaves out, in order of likely impact: track
  position and the undercut (other cars), tyre wear growing with fuel load, a non-linear
  drop-off late in a stint, and when safety cars actually happen.

## 2026-11-08: multi-car race simulation

### Unit tests: 6 passed (40 in total)

`tests/test_racesim.py`: identical cars keep the grid order where passing is hard; a faster
car starting last moves forward, more where passing is easy; **the undercut emerges without
being coded**: two equal cars nose to tail with no on-track passing, the chaser stays
second with the same strategy (mean position > 1.9) and finishes ahead most of the time if
it stops 6 laps earlier (mean < 1.5); an extra stop costs positions when tyres do not wear;
a safety car shrinks the gap to the leader; calibration picks a lower overtaking threshold
when history shows more position changes.

### Backtest (2026 dry races, 17 races, top 5 on the grid: 78 driver cases)

Field from the real grid, base pace from the qualifying gap to pole, rivals on the
single-car best strategy with stops moved by up to 3 laps, overtaking threshold calibrated
to the circuit's earlier position changes. For each top-5 driver, the 40 best single-car
strategies are simulated against the rivals and the best expected finishing position wins.

| | first-stop error, median (mean) | stops match |
|---|---|---|
| Multi-car simulator | 10.0 laps (11.0) | 50% |
| Single-car simulator | 8.0 laps (11.1) | 53% |
| Historical baseline | 8.0 laps (8.9) | 42% |

Multi-car closer to the real first stop in 29 cases, single-car in 37 (12 ties).

| Finishing order, Spearman vs the real result | mean | races better |
|---|---|---|
| Multi-car simulation (expected positions) | **0.710** | 11 |
| Starting grid | 0.672 | 6 |

The calibration did not behave: the overtaking threshold landed on the edges of its range
(0 or 3) in every race. Diagnosis on Austria 2026: simulated position changes stayed near
0.4 per car for any threshold, against 2.8 in the circuit's history (finishers only). The
simulation is too deterministic: the grid comes from qualifying and race pace is taken from
the same qualifying gap, so cars start already in pace order. Missing sources of position
change: race pace differing from one-lap pace, the start and first lap, and race-day
variation. The finishing-order result (11 of 17 races better than the grid) is
encouraging but not significant (sign test p ≈ 0.33) and rests on this mis-calibration.

## 2026-11-08: multi-car simulation, refined

### Unit tests: 8 passed (42 in total)

Added: race-day form and start spread raise position changes; pre-race pace uses only
earlier races of the same season (blend 0.7 race history + 0.3 qualifying gap).

### Changes

- Base pace from **race pace** (`racepace.py`): median gap to the field on the same lap,
  last 3 races of the season, blended with the qualifying gap.
- Race-day form (`form_sd`) and a start spread (`start_sd`) as sources of position change.
- Parameters calibrated once on 2025 (21 races, grid of 27 combinations) against finishers'
  mean |finish − grid|, then frozen for 2026.

Calibration: form_sd 0.1, start_sd 1.5, pass_threshold 3.0 (rmse 1.17 positions,
Spearman on 2025 0.644). **All three on the edges of the grid.**

### Backtest (2026, 17 races, 78 top-5 cases)

| | first-stop error, median (mean) | stops match |
|---|---|---|
| Multi-car simulator | 9.5 laps (10.9) | 50% |
| Single-car simulator | 8.0 laps (11.1) | 53% |
| Historical baseline | 8.0 laps (8.9) | 42% |

Multi-car closer in 36 cases, single-car in 40 (2 ties).

Split by whether the real first stop came under a safety car / VSC (19 of 78):

| median (mean) error | multi | single | baseline |
|---|---|---|---|
| green-flag stops (59) | 7.0 (7.8) | 6.0 (7.1) | 6.0 (6.8) |
| SC/VSC stops (19) | 19.0 (20.5) | 21.0 (23.5) | 14.0 (15.4) |

On green-flag stops the baseline beats the multi-car simulator 36 to 14.

| Finishing order, Spearman | mean |
|---|---|
| Multi-car simulation | 0.672 (better than grid in 7 races, worse in 9) |
| Starting grid | 0.672 |
| Finishing-order model (section 2) | 0.693 |

Simulated position changes 2.38 per car vs 2.70 real.

**Conclusion.** The refinement fixed the calibration problem (position changes now close
to reality) but the earlier finishing-order gain disappeared: it came from the
mis-calibration. The multi-car simulator does not pick better pit timing than the
single-car one or the historical baseline. A quarter of real first stops are triggered by
safety cars, which no pre-race plan can predict; on green-flag stops all methods err by
6–7 laps. The multi-car simulator is kept for what it does model (traffic, undercut,
what-if scenarios in the dashboard), not as a better pre-race predictor.

## 2026-11-08: dashboard (Gradio)

### Unit tests: 7 passed (49 in total)

`tests/test_planner.py` (on the real data): upcoming races listed first; overrides of pit
loss and safety-car rate are applied; degradation rates ≥ 0 and compound offsets within
[-1.2, 0] s/lap; changing the Pirelli compounds changes the inputs; strategy text round
trip; the grid of an upcoming race is estimated and ordered by recent race pace; the real
race trace has a leader (gap 0) on every lap; the same strategy for A and B gives
identical simulated races (common random numbers).

### Manual check in the browser (local server)

- Strategy tab, Las Vegas (upcoming, compounds provisional): plan in ~7 s; ranking with
  10–90% range, stint timeline, degradation curves.
- Mexico City 2026: plan vs the real stints of the top 10; race simulation (300 runs) for
  LEC with finishing-position histogram, one simulated race trace and the real race trace.
- Problems found and fixed: strategy B empty when the alternative was outside the top 25;
  race dropdown drawn under the labels; filled uncertainty bands hid the degradation
  curves; Mexico 2026 offsets of -2.4 s/lap (medium vs hard) → dashboard guards
  (methodology section 5).

Known limits: the simulated field spreads more than real races (150 s vs ~30 s to the
last car at Mexico 2026, a race with safety cars); with hard-tyre degradation floored at 0,
some plans favour very long hard stints (Mexico: H62 → S9).

## 2026-11-08: track map and on-track replay

`python -m f1agent.trackmap`: fastest clean lap of the latest race at each circuit, x/y
from OpenF1 `location` and speed from `car_data` (2 calls per circuit). 25 of 26 circuits
(Monte Carlo: no telemetry returned for the chosen lap); 262–431 points per lap.

### Unit tests: 2 passed (51 in total)

Track maps exist for the three upcoming circuits with increasing time and plausible speeds;
`on_track` puts the leader at the line, drops retired cars and places a car one lap down
at the same spot as one with the same gap modulo a lap.

### Manual check in the browser

Strategy tab shows the circuit coloured by speed with start/finish and sector marks.
Race simulation: on-track replay of the simulated race (and of the real one, for past
races), lap chosen with a slider. **Problem found:** Gradio's Plot component drops Plotly
animation frames, so the first version (Play button + frames) did not move; replaced by
one trace per lap with the slider toggling visibility (checked: lap 40 shows the field
spread around Las Vegas). No Play button.

Placement is approximate: a gap of g seconds puts a car where the reference fast lap was
g seconds before the line, so cars on slower laps (fuel, tyres, safety car) are drawn
slightly off their true spot.

## 2026-11-08: deploy on Hugging Face Spaces

`scripts/deploy_space.py --repo vtaua/f1-pit-wall` (Space created on the website: the API
answers 402, and free Gradio Spaces run on ZeroGPU, which needs one `@spaces.GPU` function
to start; app.py has an empty placeholder, everything runs on CPU). Build ~1 min, then
RUNNING. Checked in the browser: Las Vegas plan loads on start; race simulation, on-track
replay and gap chart render.

## 2026-11-08: is the simulated field too spread out?

Hypothesis (from a replay where a car ended 56 s behind at P8): the simulated field spreads
too much. Measured before changing anything: gap of the classified P5 and P10 to the
winner, each on the car's last lap against the winner on the same lap (lapped cars count),
vs the median over 100 simulated races with the calibrated parameters.

| median gap to the winner | real P5 | sim P5 | real P10 | sim P10 |
|---|---|---|---|---|
| 2025 (21 races) | 29.6 s | 30.9 s | 66.9 s | 57.7 s |
| 2026 (17 races) | 26.3 s | 27.5 s | 90.9 s | 75.8 s |

Race by race the simulator has no skill at telling which races spread out (Spearman of
P5 gaps 0.11 in 2025, 0.02 in 2026; mean absolute error 16–19 s at P5, 38–44 s at P10).

**The hypothesis is wrong**: on average the spread is about right at P5 and too small
at P10. The earlier "150 s vs ~30 s at Mexico 2026" compared the simulated last car with
real gaps that left out lapped and retired cars. Making dirty air cheaper or the start
calmer would shrink the spread further, so the change was not made. How spread out a
given race ends depends mostly on when safety cars come, which a pre-race simulation
cannot know.

## 2026-11-08: dashboard redesign and animated replay

- Track outline: periodic smoothing spline through the telemetry samples (time as the
  parameter), rotated so the long side is horizontal; drawn as a ribbon coloured by speed.
- Replay: own canvas component (`assets/replay.js` in a `gr.HTML` with `js_on_load`), since
  Gradio drops Plotly animation frames. Cars move continuously between lap times, with
  play/pause, 0.5–8× speed, a scrubber, live order with gaps, hover details; simulated or
  real race.
- Layout: dark theme, KPI cards, map next to the strategy ranking, stints full width,
  degradation next to the notes.

### Unit tests: 2 passed (53 in total)

Smoothed track is ordered by lap fraction, closed, rotated and scaled to [-1, 1]; replay
data has every car's race time per lap and stops retired cars.

### Manual check in the browser

Las Vegas: cards, map and ranking render; simulation runs; Play moves the field (lap 7 at
4× after 3 s, live order VER, LEC, HAD, HAM, PIA); LEC drops to P18 after his lap-8 stop.

Fix (same day): at the end of the replay every car showed +0.0, because all finished cars
sat at the line with equal progress. Order is now by who crossed the line first and gaps
are timing-screen gaps (time between the leader and the car at the last line both
crossed); at the finish, cars a lap or more behind show "+n laps". Checked: final board
Winner VER, LEC +26.0, HAD +35.1, ...

Replay end (same day): the animation now stops at the chequered flag (the winner's last
lap) and every car is frozen where it was at that instant: a car g seconds behind sits g
seconds of the lap before the line, so the spacing on track shows the finishing gaps
(lapped cars wrap around). Checked: LEC +26.0 s sits about 28% of a lap before the line.

Layout width (same day): Gradio capped the page at ~1100 px with side margins, so on a
1440 px screen the charts had 758 px. Now full width (up to 1680 px), slimmer sidebar:
1083 px for the charts at 1440 px; the replay canvas is 879×562 and as tall as the live
order. In narrow areas (< 720 px) the live order moves under the track in columns.

## 2026-11-09: session update command

`python -m f1agent.update`: new OpenF1 sessions, new results, qualifying starting grid
for upcoming races (OpenF1 `starting_grid`; checked on São Paulo 2026: 22 cars, NOR on
pole), laps rebuilt (rewritten only if the content changed, fixed gzip timestamp), and the
current plan of every upcoming race appended to `results/live/predictions.jsonl` (skipped
when identical to the previous one). `--compounds RACE=H,M,S` records Pirelli's
nomination; `--publish` tests, commits, pushes and redeploys.

First run (no new sessions since São Paulo): 35 s; Las Vegas S42 → M8, Qatar S42 → M15,
Abu Dhabi S12 → M46, all with estimated grids; second run logged nothing new.

### Unit tests: 3 passed (56 in total)

Compound override marks the nomination official and rejects unknown races; sessions seen
are listed with qualifying; a saved qualifying grid replaces the estimated one.

## 2026-11-10: race engineer (agent in the dashboard)

New agent tools (`f1agent/strategy_tools.py`), the same calculations as the dashboard:
`upcoming_races`, `strategy_plan` (with what-if compounds, track temperature, pit loss,
safety-car chance), `simulate_race` (driver on strategy A vs B, whole field),
`prediction_log`. System prompt rewritten as a strategy engineer; new "Race engineer" chat
tab (Gemma 4 31B via the Gemini API, default thinking).

Live check (free Gemini tier), each answer right and grounded in one tool call:
- "Qual a melhor estratégia para Las Vegas?" → strategy_plan → S42 → M8 (63 s, 1 retry).
- "E se a perda no pit for de 30 s e sem safety car?" → strategy_plan(pit_loss=30,
  safety_car_chance=0) → M10 → S40 (58 s).
- "HAM: M25 → H25 é melhor que o recomendado?" → simulate_race → no: expected P4.5 vs P3.4,
  podium 47% vs 67% (140 s, 1 retry).
- In the dashboard: "Compostos do Catar e o grid já é o real?" → upcoming_races → C1/C2/C3,
  provisional, grid estimated (64 s).
First answer misread the 10–90% loss range as traffic; the tool now explains that it comes
mostly from the random safety car. Latency (1–2 min) is the free API's.

### Unit tests: 8 passed (65 in total)

Race names resolve ("vegas", "Catar", "Brasil 2026"), strategies parse in several
spellings and must add up to the race distance, what-if inputs reach the plan, simulation
errors come back as messages the model can act on; the agent exposes all tools.

## 2026-11-10: OpenRouter provider with Gemini fallback

`agent.ask(provider="auto")`: OpenRouter (`google/gemma-4-31b-it:free`, OpenAI-compatible
tool calling) when `OPENROUTER_API_KEY` is set, falling back to the Gemini API. Chat history
is now plain text turns (tool results are not carried over), so a conversation can change
provider. `OPENROUTER_BASE_URL` can route the calls through the tokenprof proxy;
`agent_eval --provider` picks the provider.

Live check: every call to the free Gemma 4 models on OpenRouter (31B and 26B A4B) answered
429 "temporarily rate-limited upstream" over several minutes; the upstream provider is
Google AI Studio's shared free pool. With `auto`, OpenRouter failed in ~4 s and Gemini
answered correctly ("Las Vegas?" → S42 → M8, uncertainty attributed to the safety car).
The 16-question eval on OpenRouter and the tokenprof profile wait for the free tier to
answer.

### Unit tests: 2 passed (67 in total)

Fallback to Gemini with text history kept; a missing provider is reported.

## 2026-11-11: agent eval with the current agent (Gemini)

Same 16 questions as before, current agent (race-engineer prompt, 10 tools), Gemma 4 31B
on the Gemini API with its default thinking (`results/agent_eval_gemini`).

| | first agent (6 tools) | current agent (10 tools) |
|---|---|---|
| correct | 15/16 | **16/16** |
| right tool called | 15/16 | 16/16 |
| errors | 1 | 0 |
| mean LLM calls | 1.81 | 1.94 |
| mean prompt tokens | 2,046 | 4,024 |
| mean output / thought tokens | 50 / 137 | 58 / 157 |
| API retries (questions affected) | 37 (12) | 21 (11) |
| median seconds | – | 66 |

Prompt tokens doubled: the four new tool schemas and the longer system prompt are sent
with every call. That is the overhead tokenprof should quantify once OpenRouter answers.
