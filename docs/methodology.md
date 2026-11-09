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
$k$ gives a measured rate $\rho_k$, weighted by its clean laps $L_k$. A ridge regression
(`tyres.fit_predict`) solves

$$\min_{w,\,b}\ \sum_k L_k \left(\rho_k - b - x_k^\top w\right)^2 + \lambda \lVert w\rVert^2, \qquad \lambda = 1,$$

with $x_k$ = (Pirelli compound number C1-C6, mean track temperature, 1 if 2026 car),
each standardised to mean 0 and variance 1 so the penalty treats them alike.

In practice the penalty does almost nothing: the weights $L_k$ are lap counts, hundreds
per row, so the squared-error term is thousands of times larger than $\lambda\lVert
w\rVert^2$ with $\lambda = 1$. The fit is essentially weighted least squares; adding the
standardisation left every validation number unchanged. What keeps it stable is having
only three inputs.

The real compound number comes from Pirelli's nominations for each weekend
(`data/pirelli_compounds.csv`): the "soft" at one race can be a C5 and at another a C3.

### 1.5 Validation

Walk-forward (`tyres.validate`): each race's measured rates are predicted from races
before it only (at least 10), then the error is

$$\text{MAE} = \frac{\sum_k L_k \left|\hat\rho_k - \rho_k\right|}{\sum_k L_k}.$$

Compared with three predictors that need no model: the average rate of the same label
(SOFT/MEDIUM/HARD), of the same real compound, and of the same circuit in earlier years.
Results are in `docs/test-log.md`. Note that $\rho_k$ is itself an estimate with its own
noise, so even a perfect predictor would not reach zero error.

## 2. Race finishing order

### 2.1 Features

For each driver in race $r$, features use only races before $r$ (`features.build_features`):
grid slot, gap to pole in qualifying (%), mean finish over the last 5 races, mean finish
this season, the team's equivalents, championship position, share of the last 10 races
not finished, round number, the circuit's overtaking index
$\frac{1}{N}\sum |p_{\text{finish}} - p_{\text{grid}}|$ over earlier editions, and the
driver's mean finish at the circuit.

### 2.2 Model

A gradient-boosted regression tree ensemble (scikit-learn's
`HistGradientBoostingRegressor`) predicts each driver's finishing position from the
features; drivers are then ranked by predicted value. It handles missing values natively
(a rookie has no history), so nothing is imputed. Trees: depth 3, 300 iterations, learning
rate 0.05, at least 20 samples per leaf, L2 penalty 1.

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

### 3.3 Race pace (`racepace.py`)

For each race and driver, pace is the median over clean laps of the fuel-corrected time
minus the field median on the same lap (fuel and track evolution cancel), relative to the
fastest driver. Before a race, base pace is $0.7\,\bar{p}_{last\,3} + 0.3\, q$ with $q$
the qualifying gap; with no earlier race in the season, $q$ alone.

### 3.4 Calibration and result

$(\sigma_{form}, \sigma_{start}, \theta)$ were chosen on the 2025 races by grid search,
minimising the error between simulated and real mean $|\text{finish} - \text{grid}|$, then
frozen for the 2026 test. Result (details in `test-log.md`): position changes match
(2.4 vs 2.7 per car), but neither simulator picks pit timing better than the historical
baseline, and simulated finishing order is no better than the grid. About a quarter of
real first stops are triggered by safety cars, which a pre-race plan cannot foresee. The
simulators are used for what-if comparison, not as predictors.

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
