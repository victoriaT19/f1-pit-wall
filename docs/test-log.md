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

## 2026-11-11: agent eval on OpenRouter (free Gemma 4 31B), not usable

The free `google/gemma-4-31b-it:free` answered one probe after Google's daily quota reset,
so the 16-question eval started through the tokenprof proxy. 15 of 16 questions then got
429 "temporarily rate-limited upstream" after two retries each (31 retries in total); only
one question went through, and it was correct. The upstream pool is shared by every free
user, so this measures availability, not the agent: no comparison with the Gemini run is
possible, and the result files are not committed. The profile covers only 4 calls (7,213
prompt tokens), too few to quantify the tool-schema overhead.

In the app the agent tries OpenRouter first and falls back to Gemini within ~4 s, so these
429s do not reach the user.

## 2026-11-11: Gemini becomes the first provider

After the OpenRouter attempts, `available_providers()` now returns Gemini first and OpenRouter
as the fallback, so chat answers no longer wait ~4 s for a 429 before reaching Gemini.

### Unit tests: 69 passed

New `test_gemini_is_tried_first`. `test_agent_stops_after_max_steps` is now pinned to Gemini:
with the new order its failure fell through to a real OpenRouter request.

## 2026-11-11: time budget for chat answers

On the Space a strategy question sat on "Box, box…" for minutes. Locally the same question
showed why: the Gemini API answered 500/503 and timed out, one request took 169 s with three
retries, and each tool step could retry for minutes. The simulation tool itself takes ~5 s.

`ask(time_budget=...)` now caps the whole answer (the first provider gets 60% of it, the
fallback the rest; no new request or retry starts after the deadline). The app uses 150 s,
shows the elapsed time every 3 s and, when both free APIs fail, says so in plain words.

### Manual run: both free APIs busy, answer in 97 s

Gemini: `[500, 'timeout']`, stopped at its 90 s share; OpenRouter: 429 upstream. The chat
showed the timer, then the "APIs are busy, try again in a few minutes" message.

### Unit tests: 71 passed

New: a Gemini that answers 503 forever ends within the budget and the fallback still runs;
a budget error is reported in the trace.

## 2026-11-11: chain of free models and live steps in the chat

The two-provider fallback was not enough: at the same moment the free Gemma 4 31B was
rate-limited upstream on OpenRouter and the Gemini API answered 500/503 at ~25 s per step.
A probe of the free OpenRouter models with tool support found four that call tools correctly
(Gemma 4 31B, Gemma 4 26B, Nemotron 3.5 Super, ERNIE 5.5 Flash); Nemotron 3 Ultra and
Lightning answered without calling the tool.

`provider="auto"` now walks `CHAIN`: Gemma 4 31B, Gemma 4 26B and Nemotron 3.5 Super on
OpenRouter, then Gemma 4 31B on the Gemini API (large daily quota, slow when overloaded).
A 429 is no longer retried, so a busy route costs ~1 s. An explicit provider or model still
gives a single route, so evals stay reproducible. The chat shows the model being tried and
each tool call live, and ends with the model that answered.

### Manual runs: "Para o Hamilton em Las Vegas, M25 → H25 é melhor que o plano recomendado?"

| chain order | result |
|---|---|
| Gemini before Nemotron | correct, 97 s: Gemini ran 3 tools at ~25 s each, ran out of its share, Nemotron redid it |
| Nemotron before Gemini (kept) | correct, 27 s: both Gemma routes busy (~1 s), Nemotron ran `strategy_plan` and `simulate_race` |

Both answers: S42 → M8 is better (expected P3.4 vs P4.5, podium 66.7% vs 47.3%), with the
provisional compounds and estimated grid flagged.

### Unit tests: 74 passed

New: the auto chain is free-only and follows the keys that are set; busy routes are skipped
and progress events reported; an OpenRouter 429 is not retried. Two older tests called
`ask()` without a provider and, with OpenRouter first, reached the real API: pinned to Gemini.

## 2026-11-12: finishing order, Plackett-Luce ranking model and ensemble

Literature checked: van Kesteren & Bergkamp (2023), a Bayesian rank-ordered logit
(Plackett-Luce) with driver, driver-season, constructor and constructor-season effects;
Heilmeier et al. (2020), Monte Carlo race simulation with probabilistic starts, lap times
and pit stops; Pitwall (Santillana, 2026, arXiv:2607.06495), which scores race-outcome
probabilities with the Brier score; LambdaRank (LightGBM) as used by student F1 projects.

Tried, walk-forward, hyperparameters chosen on 2024 only (2025-2026 held out):

| 2024 validation | Spearman | Brier win | Brier podium | Brier top 10 |
|---|---|---|---|---|
| starting grid | 0.730 | 0.0371 | 0.0726 | 0.1397 |
| gradient boosting (current) | 0.716 | – | – | – |
| gradient boosting + race pace | 0.707 | – | – | – |
| LambdaRank (LightGBM) + race pace | 0.738 | – | – | – |
| Plackett-Luce, linear grid | 0.748 | 0.0384 | 0.0832 | 0.1204 |
| Plackett-Luce, + log grid (kept) | 0.750 | 0.0361 | 0.0748 | 0.1211 |

Plackett-Luce was stable across half-lives (120-480 days) and priors (Spearman 0.743-0.753).
Sharper probabilities (multiplying the strengths by 1.3-2.5) helped the top 10 and hurt
win and podium, so the fitted scale is kept. Adding log(grid) fixed the win probabilities:
pole is worth more than a linear grid term allows.

### A leak found on the way

Rows of each race were in finishing order (the feature table is sorted that way) and
`to_order` broke ties by row order. Summed ranks tie often, so the first ensemble number
(0.719) used the result. Rows are now put in grid order before predicting. The
championship-order baseline (drivers on 0 points tie) and recent form were affected the same
way; continuous scores (gradient boosting, Plackett-Luce) were not.

### Backtest, 68 races (2024-2026), after the fix

| | Spearman | Spearman 2025-26 (held out) | winner | podium | top 10 |
|---|---|---|---|---|---|
| ensemble (gradient boosting + Plackett-Luce ranks) | **0.705** | **0.684** | 46% | 63% | 81% |
| Plackett-Luce | 0.701 | 0.673 | 49% | 68% | 79% |
| gradient boosting | 0.691 | 0.677 | 44% | 58% | 79% |
| starting grid | 0.677 | 0.648 | 60% | 68% | 79% |

Ensemble vs gradient boosting: better in 47 races, worse in 21 (sign test p = 0.002).
Ensemble vs grid: 41 vs 27 (p = 0.11), so better on average but not significant. The grid
still picks the winner more often. A three-way ensemble with LambdaRank (0.711 before the
fix) and one with the grid as a third member did not beat the two-model ensemble on the
held-out years, so neither is used and LightGBM is not a dependency.

Probabilities (Brier, 2025-26 held out): win 0.0267 (grid 0.0250), podium 0.0626 (grid
0.0623), top 10 0.1567 (grid 0.1648). The ensemble's probabilities are the mean of
Plackett-Luce and grid-slot frequencies: as good as the grid for the podium, better for the
top 10, slightly worse for the win.

### Unit tests: 5 new

Plackett-Luce recovers the order of known strengths; the analytic gradient matches finite
differences; probabilities add up (win 1, podium 3) and a certain DNF finishes last; DNF
rates are shrunk towards the field; a retired car drops to the back in the simulator, except
in the replayed race.

## 2026-11-12: whole-field simulation, retirements and pooled race pace

Two changes, tested separately and together, calibrated on 2025 and frozen for the 17 dry
races of 2026 (`racesim_backtest.py --pace season|decayed [--no-dnf]`):

* retirements: each car retires with its driver's recent DNF rate (shrunk to the field's),
  on a uniform lap, and drops to the back (`ranking.dnf_rates`);
* race pace pooled over earlier seasons with a 90-day half-life, in % of the lap, instead of
  the last three races of the same season.

| variant | calibrated (form, start, pass) | Spearman | sim vs grid | first stop, median error | position changes sim / real | Brier win | podium | top 10 |
|---|---|---|---|---|---|---|---|---|
| first version | 0.1, 1.5, 3.0 | 0.671 | 8-9 | 10.0 | 2.38 / 2.70 | 0.0251 | 0.0646 | 0.1634 |
| retirements | 0.1, 0.3, 3.0 | 0.678 | 9-8 | 10.0 | 3.07 / 2.70 | 0.0201 | 0.0671 | 0.1674 |
| pooled pace | 0.1, 1.5, 3.0 | 0.670 | 8-9 | 10.0 | 2.53 / 2.70 | 0.0258 | 0.0660 | 0.1639 |
| **both (kept)** | 0.1, 0.3, 1.5 | **0.698** | 9-8 | 9.0 | 3.09 / 2.70 | **0.0195** | 0.0616 | 0.1608 |
| starting grid | – | 0.672 | – | – | – | 0.0227 | 0.0616 | 0.1587 |

The first-version row differs slightly from the earlier run (median stop error 10.0 vs 9.5)
because the retirement draws change the random stream even with zero probability. With both
changes the simulated order beats the grid and so do the win probabilities; the podium ties
and the top 10 is a little worse. Position changes now overshoot (3.1 vs 2.7) where they
undershot before. 17 races is a small sample: a modest gain, not a solved problem. The
dashboard and the `simulate_race` tool use the kept version; the replay shows a race
without retirements.

## 2026-11-12: agent eval on Nemotron (the chain's fallback)

Same 16 questions, `nvidia/nemotron-3.5-super-vl:free` on OpenRouter (`results/agent_eval_nemotron`),
to check that a fallback answer is as good as a Gemma one.

| | Gemma 4 31B (Gemini API) | Nemotron 3.5 Super (OpenRouter) |
|---|---|---|
| correct | 16/16 | 14/14 answered |
| errors | 0 | 2 (OpenRouter's 50 free requests per day ran out) |
| median seconds | 66 | 4.6 |
| mean prompt / output tokens | 4,024 / 58 | 4,710 / 360 |

One Nemotron answer ("o pole venceu todas as quatro edições") was right but marked wrong:
the checker only read digits. It now also reads number words from two to ten ("um/uma" are
left out, they are also the article), and both evals were re-scored without new API calls.
The model-accuracy check accepts the ensemble's or the gradient-boosting winner rate.

## 2026-11-12: rivals' pit stops drawn in every simulated race

In the dashboard, the same race replayed with a different driver selected had four different
winners, each the driver being studied. Cause: each rival ran the best plan with its stops
moved by up to 3 laps, but that shift was drawn once per scenario and kept for all simulated
races. A driver could carry the same early or late stop through every race whenever he was a
rival; selecting him gave him the exact plan and removed that fixed handicap.

Measured as a driver's mean position when selected minus when he is a rival (top 6 of the grid,
everyone on the best plan, 200 races):

| race | before: mean (range per driver) | after: mean (range per driver) |
|---|---|---|
| a 2026 race (past:11404) | -0.38 (-1.54 to +1.05) | +0.02 (-0.14 to +0.14) |
| a 2026 race (past:11396) | -0.52 (-1.86 to +0.40) | +0.01 (-0.04 to +0.05) |
| Las Vegas (next) | -0.05 (-0.56 to +0.46) | +0.07 (-0.03 to +0.21) |
| Qatar (next) | -0.18 (-1.06 to +0.48) | +0.08 (-0.04 to +0.28) |

Fix: `Car.stop_shift`; the simulator draws every rival's stop laps again in each simulated race
(never leaving a stint under 6 laps), and the selected driver runs exactly the strategy asked
for. After the fix the replayed race has the same winner whichever driver is selected. The
backtest also now keeps the selected car's DNF chance when it tries candidate strategies (it
was dropped before).

### Backtest, calibrated on 2025, tested on 2026 (17 races)

| | Spearman | sim vs grid | first stop, median error | Brier win | podium | top 10 |
|---|---|---|---|---|---|---|
| before (shift fixed per scenario) | 0.698 | 9-8 | 9.0 | 0.0195 | 0.0616 | 0.1608 |
| after (shift drawn per race) | 0.696 | 10-7 | 9.0 | 0.0227 | 0.0635 | 0.1595 |
| starting grid | 0.672 | – | – | 0.0227 | 0.0616 | 0.1587 |

Calibration moved to pass threshold 3.0 s/lap (form 0.1, start 0.3). The simulated order still
beats the grid; the win probabilities that beat the grid before were partly the artefact and
now tie with it.

### Unit tests: 81 passed

New: rival stops move by every shift from -3 to +3 across races, exactly one stop each time,
tyre age restarts at the stop, no shift without `stop_shift`, and never a stint under 6 laps.

## 2026-11-13: tyre model ranked the compounds backwards

Every recommended plan had a long soft stint (Las Vegas S42 → M8, Qatar M10 → S47, Abu Dhabi
M14 → S44). The predicted rates were inverted in every race: Las Vegas soft 0.000, medium 0.004,
hard 0.010 s/lap per lap of age. The measured rates were not: inside the same race the soft wears
0.028 faster than the hard on average (44 races with both). The prediction model regressed the
rate on the compound number (C1-C6), and across races that number is confounded with the track
(softer nominations go to gentler tracks; mean measured rate by C is flat, C2 lowest), so it
learned "softer = less wear". The compound pace offsets were then computed with those predicted
rates applied to other races' laps, and the soft came out slower than the hard (capped to 0).

Fixes:

* rate = race level + compound step (`tyres.levels_and_steps`): the steps come from within-race
  comparisons (medium +0.010, soft +0.014 over hard on all data); the level of a new race from a
  ridge regression on track temperature and the 2026 flag. Adding circuit history to the level
  made it worse (prior weight 1: MAE 0.0394, 3: 0.0350, none: 0.0327), so it is not used.
* compound offsets: age-corrected with the wear measured in the same laps, pooled over all
  earlier races (one or two editions of a circuit gave e.g. medium 0.47 s/lap faster than hard),
  and capped to soft <= medium <= hard (race laps still put the soft slightly behind the medium).
* what-if compounds: the model predicts the race's nomination; other compounds move each rate
  by the within-race wear per C step (0.008).
* the agent's prediction for an upcoming race uses the dashboard's grid (real after qualifying,
  else estimated from race pace) instead of the championship order.

### Tyre validation, walk-forward, 56 races

| | MAE per rate | MAE 2026 | MAE of the softer-minus-harder difference | softer predicted faster |
|---|---|---|---|---|
| level + step (kept) | **0.0327** | 0.0345 | **0.0291** | **100%** |
| first version (compound number) | 0.0333 | 0.0315 | 0.0300 | 15% |
| label average | 0.0337 | 0.0376 | 0.0362 | 89% |
| compound-number average | 0.0349 | 0.0353 | 0.0315 | 51% |
| circuit history | 0.0413 | 0.0349 | 0.0479 | 60% |

The measured difference is positive in 62% of the 104 pairs (measurement noise). The first
version's mean error was fine; it was the ranking of compounds, which strategy depends on, that
was wrong. Practice vs prior on 2026 races: prior 0.033 (was 0.030), practice 0.088; practice is
still not used.

### Strategy backtest, 17 races of 2026 (top 5 of the grid)

| | first version | corrected |
|---|---|---|
| plan uses the compound set most of the top 10 used | 12% | **65%** |
| share of top-10 drivers on the plan's compound set | 19% | **52%** |
| start compound matches the most common | 12% | 24% |
| plans with a soft stint of 30+ laps | 12% | 0% |
| first stop, median / mean error (laps) | 7.0 / 10.6 | 7.0 / 10.5 |
| number of stops right | 65% | 65% |

### Whole-field simulation, calibrated on 2025, tested on 2026

| | before | after | grid / baseline |
|---|---|---|---|
| Spearman, expected order | 0.696 | **0.703** | 0.672 |
| Brier win / podium / top 10 | 0.0227 / 0.0635 / 0.1595 | 0.0219 / 0.0620 / 0.1587 | 0.0227 / 0.0616 / 0.1587 |
| first stop, median error (multi / single) | 9.0 / 8.0 | 10.0 / 10.5 | 8.0 |

Compound choice is now realistic and the finishing order slightly better; the timing of the first
stop is no better (strategy backtest) or a little worse (simulation backtest) and still does not
beat the historical baseline. The plans now often put the stop at an extreme of the allowed range
(e.g. M8 → H42): with the small medium-vs-hard pace gap that race data supports, the model sees
little reason to run the medium longer.

### Unit tests: 83 passed

New: on synthetic races where gentle tracks get softer nominations, the level + step model
recovers the steps and predicts soft > medium > hard while the first version does not; the wear
per C step is measured inside races.

## 2026-11-13: dashboard swapped the soft and hard compounds

After the tyre fix the dashboard still recommended S42 → M8 for Las Vegas, while `plan()` without
arguments gave M8 → H42. The dashboard and the agent's `strategy_plan` pass the compound
override as (hard, medium, soft); `planner._inputs` zipped it with DRY = (SOFT, MEDIUM, HARD), so
the soft was planned as a C3 and the hard as a C5. The dashboard always sends the compounds, so
it was always swapped (also before the tyre fix). The backtests do not use the override and are
unaffected. Fix: the override is read as (hard, medium, soft) (`planner.COMPOUND_ORDER`).

New test: passing the race's own nomination as (hard, medium, soft) gives the same rates as no
override, with soft > medium > hard. Unit tests: 84 passed.

## 2026-11-13: why the first-stop window is so wide

After the tyre fixes the recommended first-stop window is very wide (Las Vegas: laps 8-42, ten
plans within 2 s of the best). F1 commentary says Pirelli's current compounds last a long time;
checked against the data (dry races, top 10 of the result):

| | 2024 | 2025 | 2026 |
|---|---|---|---|
| median race wear level, hard-equivalent (s/lap per lap of age) | 0.056 | 0.042 | 0.040 |
| share of the top 10 on one stop (median per race) | 30% | 60% | 70% |
| first stops, interquartile range (% of race distance) | 7.8% | 12.9% | 7.2% |
| first stops, full range (% of race distance) | 21.9% | 32.9% | 21.6% |

The durable tyres explain why the stop lap matters little to the model: wear fell by about a
quarter since 2024 and most of the top 10 now stop once. They do not explain all of it: the
model's window (68% of the Las Vegas race) is much wider than the band in which teams really
make their first stop (about 22% in 2026). What makes teams converge is not modelled: track
position on exit, covering an undercut, waiting for a safety car, and non-linear wear (graining,
the cliff at the end of a tyre's life). Documented in `docs/methodology.md` (section 3.4); no
code change.

## 2026-11-13: tyre cliff and a safety-car rule, both tested and left out

Two things the model lacks, from the analysis of the wide first-stop window.

### Tyre cliff

Within-stint regressions of the fuel-corrected lap time on tyre age (stint fixed effects):

| | linear term | curvature (per age²) |
|---|---|---|
| soft, all races | +0.0334 | +0.02e-3 (±0.11e-3) |
| soft, 2026 | -0.0118 | +0.80e-3 (±0.15e-3) |
| medium, all | +0.0551 | -0.50e-3 (±0.04e-3) |
| hard, all | +0.0606 | -0.58e-3 (±0.03e-3) |

Slope by age band in stints of 25+ laps: medium -0.011 (1-12), +0.040 (13-24), +0.054 (25-40);
hard +0.028, +0.041, +0.040. The early laps are flatter (the tyre coming in), then wear is close
to linear; only the 2026 soft shows a clear cliff. Long stints are also those of drivers whose
tyres held, which biases against seeing one. No cliff term was added.

### Safety-car rule

`strategy.react_stops`: a strategy becomes "stop at the planned lap, or earlier under a safety car
that comes at most R laps before it" (never leaving a stint under 8 laps), valued in every
simulated race. In the backtest the same rule is executed with the real neutralised laps (laps
where most drivers were flagged), and the historical baseline gets the same rule. R calibrated on
the 21 dry races of 2025 (prior only):

| R | first stop, median / mean error: plan + rule | baseline + rule |
|---|---|---|
| **0 (kept)** | **5.5 / 8.6** | 6.0 / 6.9 |
| 5 | 6.0 / 9.3 | 6.5 / 6.9 |
| 10 | 10.0 / 11.9 | 6.5 / 6.9 |
| 15 | 12.0 / 14.5 | 6.0 / 6.8 |
| 20 | 11.5 / 12.2 | 6.5 / 7.1 |

Every R > 0 made it worse: valuing the chance of a cheap stop, the planner moved the planned stop
later, and in the real races the rule rarely moved it back (teams pit under a safety car earlier
in the window than a plan that waits for one). `SC_REACT_LAPS` stays 0, so the 2026 results do
not change; the 2026 strategy backtest was re-run to add the realised-stop columns (same numbers).

### Unit tests

New: stops move only to a safety car inside the reaction range and never leave a stint under 8
laps, nothing moves with R = 0, and the realised stop uses the real neutralised laps.

## 2026-11-13: rivals covering each other's stops, tested and left out

`racesim.simulate` now keeps a tyre state per car (stint, age) instead of a fixed lap plan, so a
stop can move during the race. A rival marked `reactive` covers: it pits on the next lap when
the car just ahead or behind it, within `cover_gap` seconds at the start of the lap, pits, and
its own next stop is at most `cover_window` laps away (never leaving a stint under 6 laps). With
`cover_window = 0` the simulation is identical to before (positions and replay checked on a Las
Vegas scenario). The backtest also reports the spread of the first stops among the top 10
(interquartile range, % of race distance), simulated and real.

### Calibration year, 21 races of 2025 (base parameters frozen: form 0.1, start 0.3, pass 3.0)

| window W, gap | first stop, median / mean error (multi) | stops right | first-stop spread, simulated / real | Spearman | Brier win |
|---|---|---|---|---|---|
| **0 (off, kept)** | 9.0 / 10.6 | 59.6% | 4.8% / 12.9% | 0.669 | 0.0226 |
| 3, 1 s | 9.0 / 10.5 | 59.6% | 4.4% / 12.9% | 0.672 | 0.0223 |
| 3, 3 s | 9.0 / 10.6 | 59.6% | 3.6% / 12.9% | 0.670 | 0.0222 |
| 6, 1 s | 9.0 / 10.6 | 59.6% | 3.1% / 12.9% | 0.672 | 0.0220 |
| 6, 3 s | 9.0 / 10.7 | 59.6% | 2.5% / 12.9% | 0.673 | 0.0217 |
| 10, 1 s | 9.0 / 10.6 | 59.6% | 3.1% / 12.9% | 0.672 | 0.0220 |
| 10, 3 s | 9.0 / 10.7 | 59.6% | 2.5% / 12.9% | 0.673 | 0.0217 |

Covering did not change the recommended stops (the error is the same for every setting) and
pushed the simulated stops further from reality: they were already more bunched than the real
ones and covering bunched them more. Order and Brier moved only in the third decimal. The
selection criterion (first-stop error, then spread) keeps W = 0.

What this says about the wide recommended window: it is not missing coordination between
teams. The simulated rivals already stop closer together than real teams (all run the best plan
moved by at most 3 laps); the window is wide because one car's cost of stopping a few laps
earlier or later is nearly flat with today's durable tyres. More variety between rivals' plans
(different compounds and stop counts) is the more promising direction.

### 2026, re-run with the final code (cover off)

Same numbers as before up to rounding (the lap-time sum now runs in a different order, which
flips a few random pass decisions): Spearman 0.703 (grid 0.672), Brier win / podium / top 10
0.0218 / 0.0621 / 0.1586, first stop median error 10.0 (multi), baseline 8.0. New metric:
first-stop spread among the top 10, simulated 4.8% vs real 7.2%.

### Unit tests: 88 passed

New: a reactive rival covers a nearby stop on the next lap; no cover when switched off, when its
own stop is too far, when the neighbour is not close, or when the car is not reactive.
