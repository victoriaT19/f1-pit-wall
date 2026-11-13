# Methodology

The maths behind each part of the project, the assumptions it rests on, and where it
stops working. Code references point to the function that implements each step.

## 1. Tyre degradation

### 1.1 What is measured

For a lap $i$ we observe its time $t_i$, its lap number $n_i$, the tyre's age $a_i$ (laps
completed on that set) and the stint $s(i)$ it belongs to. The degradation rate $\beta$
of a compound is how many seconds a lap gets slower per extra lap of tyre age, other
things equal.

Only clean laps are used (`laps.build`): not the first lap, not pit in- or out-laps, not
under safety car, virtual safety car or red flag, not in rain, and not slower than 107% of
the driver's median lap in that race (traffic, mistakes). The 107% threshold borrows the
qualifying rule's number; it is a heuristic, not a fitted value.

### 1.2 Fuel

A car starts a race with roughly 100 kg of fuel and gets faster as it burns it. Inside a
stint, tyre age and lap number grow together:

$$a_i = a^{\text{start}}_{s} + (n_i - n^{\text{start}}_{s}),$$

so a regression of lap time on tyre age inside stints measures $\beta - \phi$, where
$\phi$ is the lap time gained per lap from burning fuel. Lap times alone cannot separate
the two. The project assumes a fixed fuel effect and adds it back
(`tyres.corrected`):

$$\tilde t_i = t_i + \phi\,(n_i - 1), \qquad \phi = 0.055\ \text{s/lap}.$$

$\phi = 0.055$ s/lap is a commonly quoted figure (about 0.03 s per kg times about 1.8 kg
burned per lap), not something measured here. A wrong $\phi$ shifts every measured rate
by the same amount (a unit test checks this), so it changes absolute rates but not which
compound wears faster. The validation was repeated with $\phi$ = 0.035, 0.055 and 0.075
s/lap; the ranking of predictors did not change.

### 1.3 Model per race and compound

For the clean laps of one race on one compound $c$:

$$\tilde t_i = \alpha_{s(i)} + \beta_c\, a_i + \varepsilon_i .$$

Each stint has its own intercept $\alpha_s$, which absorbs everything constant within a
stint: the driver, the car, the track's grip on that day, the compound's base pace. Only
the change within the stint identifies $\beta_c$. Removing stint means (the "within"
estimator, equivalent to OLS with one dummy per stint) gives a closed form
(`tyres.within_slope`):

$$\hat\beta_c = \frac{\sum_i \left(a_i - \bar a_{s(i)}\right)\left(\tilde t_i - \bar{\tilde t}_{s(i)}\right)}{\sum_i \left(a_i - \bar a_{s(i)}\right)^2},$$

where $\bar a_s$ and $\bar{\tilde t}_s$ are means over the stint's clean laps. Stints with
fewer than 6 clean laps are dropped, and a rate is trusted only with at least 40 clean
laps behind it.

Assumptions and limits:
- Degradation is linear in age. Real tyres often wear slowly, then fall off a "cliff";
  a linear rate averages over that.
- Track evolution (the track getting faster as rubber is laid down) is not modelled. It
  works like extra fuel burn and lowers measured rates; at Jeddah 2024 it roughly
  cancelled the wear.
- Drivers manage tyres. A measured rate is the wear of tyres as driven in that race, not
  the tyre's maximum wear.

### 1.4 Predicting a race that has not happened

For a future race the rates are predicted from earlier races. Each earlier race-compound
$k$ (race $r$, compound $c$) gives a measured rate $\rho_k$, weighted by its clean laps
$L_k$. The rate is split into the race's wear level and a compound step
(`tyres.levels_and_steps`, `tyres.fit_predict`):

$$\rho_{rc} = \ell_r + \delta_c + \varepsilon_{rc}, \qquad \delta_{\text{HARD}} = 0,$$

fitted by weighted least squares (alternating between the levels and the steps until they
converge). Because every race has its own level $\ell_r$, the steps $\delta_c$ come only
from comparisons inside the same race: how much faster a medium or a soft wears than the
hard on the same track and day. On all races: $\delta_{\text{MEDIUM}} = 0.010$ and
$\delta_{\text{SOFT}} = 0.014$ s/lap per lap of age. The level of the new race is predicted
by a ridge regression on mean track temperature and a 2026-car flag, fitted on the earlier
races' levels (weights = their clean laps, inputs standardised, $\lambda = 1$):

$$\hat\rho_{rc} = \hat\ell(\text{track temperature},\ \text{2026}) + \hat\delta_c .$$

**Why not the Pirelli compound number.** The first version regressed $\rho_k$ directly on
the compound number (C1-C6), track temperature and the 2026 flag. Across races the compound
number is confounded with the track: Pirelli brings softer compounds to tracks that are
easier on tyres, so the races' average rate is flat in C (C2 even lowest) while inside a race
the softer compound wears faster in most pairs. Fitted across races, the regression learned
"softer = less wear" and predicted the soft as the most durable tyre of every race; the
strategy tool then recommended 40-lap soft stints. The split above uses only the within-race
comparison for the compound effect. Blending in the levels of earlier editions at the same
circuit made the validation worse (circuit history was already the weakest simple predictor),
so it is not used.

**What-if compounds.** The model predicts the race's own nomination. When the dashboard or
the agent asks for other compounds, each rate moves by the within-race wear per C step
(`tyres.wear_per_c_step`, 0.008 s/lap per lap of age per step softer), estimated from
deviations around each race's mean so the track's own wear cancels out.

The real compound number comes from Pirelli's nominations for each weekend
(`data/pirelli_compounds.csv`): the "soft" at one race can be a C5 and at another a C3.

### 1.5 Validation

Walk-forward (`tyres.validate`): each race's measured rates are predicted from races
before it only (at least 10), then the error is

$$\text{MAE} = \frac{\sum_k L_k \left|\hat\rho_k - \rho_k\right|}{\sum_k L_k}.$$

Compared with the first version and three predictors that need no model: the average rate of
the same label (SOFT/MEDIUM/HARD), of the same real compound, and of the same circuit in
earlier years. Because strategy depends on the difference between compounds more than on any
single rate, the validation also scores, for every pair of compounds in the same race, the
error on the difference (softer minus harder) and whether the softer is predicted to wear
faster.

| walk-forward, 56 races | MAE per rate | MAE 2026 | MAE of the difference | softer predicted faster |
|---|---|---|---|---|
| level + step (current) | **0.0327** | 0.0345 | **0.0291** | **100%** |
| first version (compound number) | 0.0333 | 0.0315 | 0.0300 | 15% |
| average of the label | 0.0337 | 0.0376 | 0.0362 | 89% |
| average of the compound number | 0.0349 | 0.0353 | 0.0315 | 51% |
| circuit history | 0.0413 | 0.0349 | 0.0479 | 60% |

The measured softer-minus-harder difference is positive in 62% of the pairs: $\rho_k$ is itself
an estimate with its own noise, so even a perfect predictor would not reach zero error.

### 1.6 Compound pace on new tyres

The offset of a compound against the hard (s/lap on a new tyre) is the median, over drivers
who ran both in a race, of the difference between their stints' intercepts at age 0
(`strategy.compound_offsets`), pooled over all earlier races. The age correction uses the wear
measured in those same laps. (The first version used the rates predicted for the race being
planned, which were wrong, and the soft came out slower than the hard.) Race laps still put
the soft slightly behind the medium, because softs are run early, in traffic and on full
tanks, so the estimates are capped to the one thing known for sure: a softer tyre is never
slower when new (soft $\le$ medium $\le$ hard).

## 2. Race finishing order

### 2.1 Features

For each driver in race $r$, features use only races before $r$ (`features.build_features`):
grid slot, gap to pole in qualifying (%), mean finish over the last 5 races, mean finish
this season, the team's equivalents, championship position, share of the last 10 races
not finished, round number, the circuit's overtaking index
$\frac{1}{N}\sum |p_{\text{finish}} - p_{\text{grid}}|$ over earlier editions, and the
driver's mean finish at the circuit.

### 2.2 Models

**Gradient boosting** (`backtest.make_model`). A gradient-boosted regression tree ensemble
(scikit-learn's `HistGradientBoostingRegressor`) predicts each driver's finishing position
from the features; drivers are then ranked by predicted value. It handles missing values
natively (a rookie has no history), so nothing is imputed. Trees: depth 3, 300 iterations,
learning rate 0.05, at least 20 samples per leaf, L2 penalty 1.

**Plackett-Luce ranking model** (`ranking.py`). A race result is a ranking, so it is
modelled as one (rank-ordered logit, as in van Kesteren & Bergkamp, 2023, *Bayesian analysis
of Formula One race results: disentangling driver skill and constructor advantage*, Journal of
Quantitative Analysis in Sports). Each car has a strength

$$u_i = \beta^\top x_i + a_{d(i)} + b_{t(i),s},$$

with standardised pre-race covariates $x_i$ (grid slot, $\log$ grid slot, qualifying gap in %,
recent race pace), a driver effect $a_d$ and a team-season effect $b_{t,s}$. The probability of
an order $(1, \dots, m)$ of the $m$ finishers is

$$P = \prod_{k=1}^{m-1} \frac{e^{u_k}}{\sum_{j \ge k} e^{u_j}}.$$

Parameters are the maximum a posteriori estimate under Gaussian priors (ridge penalties
$\tfrac{\lambda_a}{2}\lVert a\rVert^2 + \tfrac{\lambda_b}{2}\lVert b\rVert^2 + \tfrac{\lambda_\beta}{2}\lVert\beta\rVert^2$,
$\lambda_a = 1$, $\lambda_b = 2$, $\lambda_\beta = 0.01$), fitted with L-BFGS on the analytic
gradient. Each race's log-likelihood is weighted by $0.5^{\text{age}/240\,\text{days}}$, so
recent form counts more. The paper explains past seasons; here the model forecasts the next
race, so only earlier races are used and the hyperparameters were chosen on 2024 only.

Race pace is each driver's median fuel-corrected lap relative to the field median on the
same lap, in % of the lap (so circuits of different length pool), averaged over earlier
races with weights $0.5^{\text{age}/90\,\text{days}}$. It helps the ranking model; added to
the gradient boosting it made the 2024 validation worse, so that model does not use it.

As in the paper, non-finishers are left out of the likelihood. Each driver's DNF chance is
his recent DNF share, $\hat q_d = \frac{\sum w\,\text{dnf} + k\,\bar q}{\sum w + k}$ with
$k = 10$ pseudo-races of the field rate $\bar q$. Probabilities come from sampling: adding
independent Gumbel noise to $u$ and sorting gives an exact Plackett-Luce draw; cars drawn
as DNF go to the back.

**Ensemble.** The predicted order is the mean of the two models' ranks (ties to the better
grid slot). Probabilities of a win, podium or top 10 are the mean of the Plackett-Luce
probabilities and the grid-slot frequencies (how often each grid slot won, reached the podium
or the top 10 in earlier races).

### 2.3 Metrics

For each race, with $d_j$ the difference between predicted and actual position of driver
$j$ among $n$ drivers, Spearman's rank correlation is

$$\rho_s = 1 - \frac{6\sum_j d_j^2}{n(n^2-1)},$$

1 for a perfect order and -1 for a reversed one. Also: mean absolute position error,
whether the predicted winner won, and the share of the actual podium and top 10 that the
prediction got.

Model and baseline are compared race by race with a two-sided sign test: with $w$ races
where the model is better and $l$ where the baseline is, ties dropped and $m = w + l$,

$$p = \min\left(1,\ 2\sum_{i=0}^{\min(w,l)} \binom{m}{i}\, 2^{-m}\right).$$

It asks only "who wins each race more often", so it makes no assumption about how
Spearman values are distributed.

Probabilities are scored with the Brier score, the mean over drivers of
$(\hat p_j - y_j)^2$ with $y_j = 1$ if the event happened (lower is better), as in
calibrated race-outcome engines such as Pitwall (Santillana, 2026, arXiv:2607.06495).

## 3. Race simulation

### 3.1 Single car (`strategy.py`)

For a strategy with stints $(c_k, n_k)$, race time is

$$T = \sum_k \sum_{a=1}^{n_k} \left(o_{c_k} + r_{c_k}\, a + \varepsilon\right) + (K-1)\,P,$$

with compound offset $o_c$, degradation rate $r_c$ (section 1), tyre age $a$, noise
$\varepsilon$ and pit loss $P$. Rates and offsets are drawn from their uncertainty in each
Monte Carlo run; all strategies share the same draws (common random numbers), so their
difference has much less noise than either time.

### 3.2 All cars (`racesim.py`)

Each car $j$ has base pace $b_j$. On lap $\ell$, its free-air time is

$$f_j = b_j + \eta_j + o_{c} + r_{c}\, a_j + \varepsilon_{j\ell}, \quad \eta_j \sim N(0, \sigma_{form}^2)$$

where $\eta_j$ is race-day form, drawn once per simulated race. At the start, each car gets
$|N(0, \sigma_{start}^2)|$ seconds on top of its grid slot. Cars are resolved front to
back; a car that would close within 0.4 s of the one ahead passes with probability

$$p = \frac{1}{1 + e^{-(\Delta - \theta)/0.25}},$$

with $\Delta$ its pace advantage that lap and $\theta$ the overtaking threshold; otherwise
it crosses the line 0.4 s behind. Following within 1 s costs 0.25 s/lap. A safety car
starts on any lap with the circuit's historical probability, lasts 4 laps, closes the
field to 0.6 s gaps and halves the pit loss. The undercut is not a rule: it appears when
new tyres gain more than the gap.

Retirements: car $j$ retires with probability $\hat q_j$ (its driver's DNF rate, section 2.2)
on a lap drawn uniformly, and drops behind every car still running and every car that
retired later. As in Heilmeier et al. (2020, *Application of Monte Carlo methods to consider
probabilistic effects in a race simulation for circuit motorsport*, Applied Sciences), the
point is the distribution of outcomes, not one race.

Rivals run the single-car simulator's best plan with each stop moved by up to 3 laps, drawn
again in every simulated race (so the field does not all pit on the same lap); the driver
being studied runs exactly the strategy asked for. The first version drew each rival's shift
once and kept it for all simulated races, so a driver could carry the same early or late stop
through every race whenever he was a rival: choosing him then changed his expected position by
up to about 2 places, in either direction.

### 3.3 Race pace (`racepace.py`)

For each race and driver, pace is the median over clean laps of the fuel-corrected time
minus the field median on the same lap (fuel and track evolution cancel), in % of that
median, relative to the fastest driver. Before a race, the race history is the average over
all earlier races, earlier seasons included, with weights $0.5^{\text{age}/90\,\text{days}}$,
converted to seconds with the circuit's lap time; base pace is $0.7\,\bar{p} + 0.3\, q$ with
$q$ the qualifying gap, or $q$ alone for a driver with no race laps yet. (The first version
used only the last three races of the same season, so the first races of a season had no
race history.)

### 3.4 Calibration and result

$(\sigma_{form}, \sigma_{start}, \theta)$ were chosen on the 2025 races by grid search,
minimising the error between simulated and real mean $|\text{finish} - \text{grid}|$, then
frozen for the 2026 test (17 dry races). With retirements, the pooled race pace and rivals'
stops drawn in every race, the calibration is $\sigma_{form} = 0.1$ s/lap, $\sigma_{start} = 0.3$ s
(less first-lap shuffle than before) and $\theta = 3.0$ s/lap.

| 2026 test | Spearman (expected order) | Brier win | Brier podium | Brier top 10 |
|---|---|---|---|---|
| first version | 0.671 | 0.0251 | 0.0646 | 0.1634 |
| + retirements and pooled race pace (rivals' shifts fixed) | 0.698 | 0.0195 | 0.0616 | 0.1608 |
| + rivals' stops drawn in every race | 0.696 | 0.0227 | 0.0635 | 0.1595 |
| + corrected tyre model (current) | **0.703** | **0.0218** | 0.0621 | 0.1586 |
| starting grid | 0.672 | 0.0227 | 0.0616 | 0.1587 |

Retirements and the pooled race pace only helped together. With them the simulated order
beats the grid (in 10 of the 17 races). The win probabilities looked better than the grid's
until rivals' stops were redrawn in every race, then tied with the grid-slot frequencies;
with the corrected tyre model (section 1.4) they beat them again, and podium and top-10
probabilities tie. 17 races is a small sample. Pit timing is still no better
than the historical baseline: about a quarter of real first stops are triggered by safety
cars, which a pre-race plan cannot foresee. The simulators are used for what-if comparison,
not as predictors of the stop lap.

**Why the first-stop window is so wide.** The recommended windows are wide (Las Vegas: laps 8
to 42, ten plans within 2 s of the best). Part of it is real: tyres last longer than they used
to. The median race wear level fell from 0.056 s/lap per lap of age in 2024 to 0.042 in 2025
and 0.040 in 2026, and the share of the top 10 on a single stop rose from 30% to 60% and 70%
(median per race). With little wear, stopping a few laps earlier or later costs little. But
the real top 10 still stop within a narrower band than the model's window: the first stops
span about 22% of the race in 2026 (33% in 2025; interquartile range 7-13%). Teams converge
for reasons the simulator does not model: track position on exit, covering a rival's
undercut, waiting for a safety car, and tyres that grain or fall off a cliff instead of
wearing linearly.

Two of these were tested and left out. *A tyre cliff*: within long stints the slope of the
fuel-corrected lap time is low in the first ~12 laps (the tyre coming in) and then roughly
constant (hard 0.041 then 0.040 s/lap per lap up to 40 laps; medium 0.040 then 0.054), with a
clear positive curvature only for the 2026 soft, so there is no consistent cliff to model. *A
safety-car rule* (`strategy.react_stops`): plans were valued and executed as "stop at the planned
lap, or earlier under a safety car at most R laps before it", with the real neutralised laps in
the backtest. Calibrated on 2025, every R > 0 made the first-stop error worse (R = 0: median 5.5
laps, mean 8.6; R = 10: 10.0 and 11.9): knowing it could pick up a safety car, the planner moved
the planned stop later, while real teams do not wait that long. R stays 0 (`SC_REACT_LAPS`).

A third idea was also tested and left out: *rivals covering each other's stops*
(`racesim.Track.cover_window`, `cover_gap`). A rival pits on the next lap when the car just
ahead or behind it, within a few seconds, pits and its own stop is at most W laps away. On the
2025 races it did not change the first-stop error at all (median 9.0 laps for every W from 0
to 10), and it made the simulated field's stops even more bunched: among the simulated top 10
the first stops already span less of the race than the real ones (interquartile range 4.8%
against 12.9% in 2025, 4.8% against 7.2% in 2026), because every rival runs the same plan moved
by at most 3 laps, and covering cut that to 2.5%. So the wide recommended window comes from a
single car's flat cost of stopping a little earlier or later, not from missing coordination;
what the simulated field lacks, if anything, is variety between the rivals' plans. W stays 0.

## 4. Agent

The agent (`agent.ask`) is a loop around an LLM with function calling. Each turn sends
the conversation and the tool schemas; if the model asks for tools, they run locally and
their results go back as function responses, until the model writes an answer (at most 6
turns). The model never computes numbers itself: the system prompt tells it to take every
figure from a tool. The evaluation (`agent_eval.py`) checks answers against values
computed from the same data, so it measures whether the agent picks the right tool and
reports its output faithfully, not whether the underlying models are right.

## 5. Dashboard guards

The dashboard (`app.py`, `f1agent/planner.py`) uses the same inputs as the backtests, with
two physical guards: degradation rates are floored at 0 (a negative rate, seen on cold
tracks, would mean tyres get faster with age), and compound offsets are clipped to
[-1.2, 0] s/lap (a softer tyre is faster than the hard, but not by 2 s/lap; estimates from
few stints can be that far off, e.g. Mexico 2026). The backtest results in `test-log.md`
use the raw estimates.
