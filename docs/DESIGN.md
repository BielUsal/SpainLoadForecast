# Day-ahead probabilistic forecasting of Spanish electricity demand

> **Internal design document.** This is the working reference for the project: the question, the data,
> the models, how they will be evaluated, the plan and the open decisions. It is written for the people
> (and assistants) building the project, not for visitors; the public-facing overview is the top-level
> [`README.md`](../README.md). Keep this file current as decisions change, and record each decision in §16
> rather than letting the document drift from the code.
>
> **Status:** design only; no code yet. ESIOS API token obtained.

---

## Table of contents

1. [Motivation and the question](#1-motivation-and-the-question)
2. [Scope and non-goals](#2-scope-and-non-goals)
3. [Data sources](#3-data-sources)
4. [Data pipeline](#4-data-pipeline)
5. [Exploratory analysis: what to look for](#5-exploratory-analysis-what-to-look-for)
6. [Feature engineering](#6-feature-engineering)
7. [Models](#7-models)
   - 7.1 [Baselines](#71-baselines)
   - 7.2 [Gradient-boosted trees](#72-gradient-boosted-trees)
   - 7.3 [Bayesian structural model (NumPyro)](#73-bayesian-structural-model-numpyro)
   - 7.4 [Sequence model (PyTorch RNN)](#74-sequence-model-pytorch-rnn)
   - 7.5 [Conditional normalizing flow over the 24-hour vector (stretch)](#75-conditional-normalizing-flow-over-the-24-hour-vector-stretch)
   - 7.6 [The professional benchmark: REE's own forecast](#76-the-professional-benchmark-rees-own-forecast)
8. [Evaluation protocol](#8-evaluation-protocol)
9. [Information leakage and other traps](#9-information-leakage-and-other-traps)
10. [Structural breaks and special periods](#10-structural-breaks-and-special-periods)
11. [Deliverables](#11-deliverables)
12. [Repository layout](#12-repository-layout)
13. [Tech stack](#13-tech-stack)
14. [Plan and milestones](#14-plan-and-milestones)
15. [Risks and mitigations](#15-risks-and-mitigations)
16. [Open decisions](#16-open-decisions)
17. [Phase 2: day-ahead price forecasting](#17-phase-2-day-ahead-price-forecasting)
18. [References and links](#18-references-and-links)

---

## 1. Motivation and the question

Spain's transmission system operator, **Red Eléctrica de España (REE)**, must balance generation and
demand in real time. Day-ahead demand forecasts drive generation scheduling, reserve sizing and
market bidding. A forecast that is off by 1 GW at the wrong hour has a real cost.

**Main question.** Given everything known at a decision time on day $d$, what is the
**probability distribution** of hourly mainland (peninsular) demand for each hour of day $d+1$?

**Questions that follow from it:**

- How close can openly available data and standard methods get to **REE's own published forecast**?
- *Where* do models fail: holidays, heat waves, cold snaps, daylight-saving transitions, structural breaks?
- Are the uncertainty bands **calibrated**? A 90% interval should contain the realised value about 90% of the time.
- Does modelling the **joint** distribution of the 24 hours (rather than 24 separate marginals)
  produce better scenarios for anyone who needs to plan a whole day?

**Why this project (portfolio angle).** It shows the full data-science loop on messy real data:
acquisition through APIs, cleaning, time-zone and calendar handling, feature engineering, classical
ML, Bayesian modelling, deep learning, rigorous probabilistic evaluation, an honest comparison against
a professional benchmark, and a deliverable a non-specialist can use (a dashboard). The domain,
energy, is a major employer of data scientists in Spain.

---

## 2. Scope and non-goals

**In scope (core):**

- Target: hourly **peninsular** electricity demand (MW), the Spanish mainland system.
- Horizon: **day-ahead**, meaning forecasts for hours 00:00–23:00 of day $d+1$ (local time), issued at a fixed time on day $d$.
- Output: a full predictive distribution per hour (samples or quantiles), not just a point forecast.
- Period: as many years as the APIs give cleanly. Aim for 2015 onwards, and at least 4 years.

**Out of scope (for now):**

- The Balearic and Canary Islands systems, which are separate grids with different dynamics. They could be an extension.
- Intraday, real-time or very-short-term (< 1 h) forecasting.
- Regional demand by autonomous community (*CCAA*).
- Price forecasting. That is Phase 2 (see §17).
- Production MLOps (scheduling, model registry, monitoring). The project should be *reproducible*, not *deployed*.

---

## 3. Data sources

> ⚠️ Indicator IDs, endpoints and access rules **must be checked** against current documentation
> before writing the pipeline. APIs change. The IDs below come from third-party documentation and
> wrappers.

### 3.1 Electricity: REE

| Source | Access | Use |
|---|---|---|
| **REData API** (`apidatos.ree.es`) | Open REST API, GET only, no token | Quick exploration; daily/monthly aggregates; generation mix |
| **ESIOS API** (`api.esios.ree.es`) | Free personal token, **requested by email from REE**; sent in the `x-api-key` header | Hourly (and 15-min) series: the main source |
| `python-esios` (PyPI) | Wrapper around the ESIOS API with caching | Convenience; or call the REST API directly with `requests` |

ESIOS indicators of interest (verify these):

| ID | Meaning | Role |
|---|---|---|
| 1293 | Real-time peninsular demand (MW) | **Target** $y_t$ |
| 544 | REE short-term demand forecast (MW) | **Benchmark** (horizon to be clarified) |
| 545 | REE daily peninsular demand forecast (MW) | **Benchmark**, day-ahead |
| 600 | Day-ahead / PVPC price | Phase 2 target |
| 10034, 10035, … | Wind / solar generation (and forecasts) | Optional covariates; Phase 2 |

**Token:** obtained. Store it in `.env` as `ESIOS_API_KEY` (git-ignored); never commit it.

### 3.2 Weather

Demand in Spain is driven mostly by temperature: heating in winter, air conditioning in summer,
which is increasingly dominant.

| Option | Notes |
|---|---|
| **Open-Meteo** historical archive API | Free, no key, hourly reanalysis-based variables at any coordinates. It also offers a *historical weather-forecast* archive, which matters for leakage (§9). Verify coverage and terms. |
| **AEMET OpenData** | Spain's official met agency; free API key. Station observations. Noisier and messier to process, but "official". |

Variables: 2 m temperature (primary), relative humidity or apparent temperature, cloud cover or
shortwave radiation (rooftop solar lowers *measured* demand around midday), and possibly wind speed.

**Spatial aggregation.** Fetch hourly series for about 6–10 large population centres, for example
Madrid, Barcelona, Valencia, Sevilla, Zaragoza, Málaga, Bilbao, Murcia, Valladolid and A Coruña.
Combine them into a **population-weighted national temperature**:

$$
\tilde T_t = \sum_{c} w_c\,T_{c,t}, \qquad w_c = \frac{P_c}{\sum_{c'} P_{c'}},
$$

with $P_c$ the population of the metropolitan area or province, taken from INE (*Instituto Nacional de
Estadística*). An alternative is to fit the weights $w_c$ as part of the model, under a simplex
constraint. That would be a nice result in itself.

### 3.3 Calendar

- The `holidays` Python package (country `ES`, with subdivisions for regional holidays).
- **National** vs **regional** holidays matter differently. A regional holiday in Madrid or Catalonia
  moves national demand much more than one in La Rioja. Encode a **population-weighted holiday
  intensity**, $h_t = \sum_r w_r \mathbb{1}[\text{holiday in } r \text{ on } t]$.
- Bridge days (*puentes*), Christmas and New Year, Holy Week (*Semana Santa*), August (the
  nationwide holiday month).
- Daylight-saving transitions, which create 23-hour and 25-hour local days.

### 3.4 Optional context

- Socio-economic indicators such as the REE large-consumer index (*IRE*), useful for long-term trend, not day-ahead.
- COVID-19 mobility or lockdown indicators, to flag 2020.

---

## 4. Data pipeline

Principles: **idempotent, cached, raw data immutable.**

```
APIs ──► data/raw/        (exact API responses, JSON/CSV, never edited)
     ──► data/interim/    (parsed, typed, UTC-indexed, one tidy table per source)
     ──► data/processed/  (merged hourly modelling table + features, Parquet)
```

Steps:

1. **Download** in monthly chunks with retries and rate limiting. Cache to `data/raw/` keyed by
   (source, indicator, period). Re-running should only fetch what is missing.
2. **Time zones.** Store *everything* in **UTC** internally. Convert to `Europe/Madrid` only for
   calendar features and for display. This sidesteps DST ambiguity: one local hour is missing in
   spring and one is repeated in autumn.
3. **Resampling.** ESIOS may return 5-, 10- or 15-minute data for some indicators. Aggregate to hourly
   **means**, since demand is a power in MW. Document the convention: is the hour labelled by its start or its end?
4. **Quality checks** (written as tests):
   - no duplicate timestamps; a complete hourly index (report gaps);
   - plausible range, roughly 15–45 GW for peninsular demand (to be confirmed in EDA);
   - flag spikes using robust z-scores on differences;
   - record every manual fix in `data/CHANGELOG.md`.
5. **Merge** electricity, weather and calendar into one hourly table.
6. **Version** the processed dataset with a hash of its contents, logged with every experiment.

---

## 5. Exploratory analysis: what to look for

A dedicated notebook, `notebooks/01_eda.ipynb`, should produce:

- The full series and its rolling annual mean: long-term trend, efficiency gains, growth of
  behind-the-meter solar, electrification.
- **Seasonal decomposition**: daily profile by weekday type × season (heatmap of hour × day-of-year).
- **Demand vs. temperature** scatter, coloured by hour. Expect a "hockey stick" or V shape with a comfort
  band of roughly 15–20 °C. Estimate where the bends are.
- How the *midday* dip evolves over the years, as a fingerprint of self-consumption solar.
- Holiday profiles vs. normal days of the same weekday.
- Autocorrelation and partial autocorrelation of demand and of residuals after seasonal removal.
- **Breaks**: 2020 lockdown, 2022 energy crisis, the April 2025 Iberian blackout (§10).
- A look at REE's own forecast errors: their distribution by hour, weekday and season. *Where does the professional struggle?*

---

## 6. Feature engineering

Information set $\mathcal{F}_{d}$: everything known at the **issue time** on day $d$ (for example
10:00 local, before the day-ahead market closes). All features must belong to $\mathcal{F}_{d}$.

| Family | Features |
|---|---|
| Calendar | hour, weekday, day-of-year (as Fourier terms), month, DST flag, holiday intensity $h_t$, bridge-day flag, days to/from holiday, special periods (Christmas, Holy Week, August) |
| Weather | $\tilde T_t$ for the *target* hours (see §9 on forecast vs. observed), lags and rolling means of $\tilde T$ (buildings have thermal inertia), heating/cooling degree hours, radiation or cloud cover |
| Autoregressive | demand at the same hour 1, 2 and 7 days earlier, i.e. $y_{t-24}$, $y_{t-48}$, $y_{t-168}$, but only lags available at issue time; daily mean and peak of the last available day |
| Trend | a slow trend term, or a rolling 365-day mean of demand |

Fourier encoding of a period $P$ with order $K$:

$$
\phi_P(t) = \big[\cos(2\pi k t/P),\ \sin(2\pi k t/P)\big]_{k=1}^{K}.
$$

Typical choices are $P=24$ ($K\approx 6$–10), $P=168$ ($K\approx 3$–5) and $P=8766$ ($K\approx 4$–6).

---

## 7. Models

There are five model families plus the external benchmark, in increasing order of complexity. **Each
one must earn its place** by improving on the previous ones on the metrics in §8, or by adding
something they cannot do (calibrated uncertainty, joint scenarios).

### 7.1 Baselines

Non-negotiable reference points. Every result is reported as a skill score against them.

- **Seasonal naive (weekly):** $\hat y_t = y_{t-168}$.
- **Seasonal naive (daily):** $\hat y_t = y_{t-24}$ when available at issue time, otherwise $y_{t-48}$.
- **Profile climatology:** mean demand for (hour, weekday type, week of year) over training years.
- Probabilistic versions: the empirical quantiles of the baseline's past residuals, by hour.

Skill score of model $m$ against baseline $b$ for a loss $L$:

$$
\mathrm{SS}_m = 1 - \frac{\overline{L}_m}{\overline{L}_b}.
$$

### 7.2 Gradient-boosted trees

The industry workhorse for tabular forecasting with strong exogenous inputs. Recruiters will expect to see it.

- LightGBM (or XGBoost) on the features in §6.
- **Direct strategy:** a single model with "target hour" as a feature, predicting $y_{t}$ for every
  $t$ in day $d+1$ from $\mathcal{F}_d$. This avoids feeding predictions back recursively.
- **Probabilistic:** quantile regression, i.e. one model per $\tau \in \{0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95\}$ trained with the pinball loss

$$
\rho_\tau(y, q) = \big(\tau - \mathbb{1}[y < q]\big)(y - q),
$$

  with a post-hoc fix for quantile crossing (sorting). Alternatively, conformal prediction on residuals of the point model.
- Feature importance and SHAP values for interpretation, so you can say *what the model learned*.

### 7.3 Bayesian structural model (NumPyro)

An interpretable, fully probabilistic model whose parameters mean something physically.

**Likelihood.**

$$
y_t = \mu_t + s^{(24)}_{\text{type}(t)}(t) + s^{(168)}(t) + s^{(8766)}(t)
      + \beta_H\, g(T_H - \tilde T^{\ast}_t) + \beta_C\, g(\tilde T^{\ast}_t - T_C)
      + \sum_{j}\gamma_{j}\, x^{\text{hol}}_{j,t} + \varepsilon_t,
$$

where:

- $\mu_t$ is a slowly varying level, for example a local linear trend or a low-order spline.
- $s^{(P)}(t) = \sum_{k=1}^{K_P} a_{P,k}\cos(2\pi k t/P) + b_{P,k}\sin(2\pi k t/P)$. The **daily profile
  is allowed to depend on day type** (working day / Saturday / Sunday-holiday) and possibly season,
  either with separate coefficient sets or with a hierarchical prior that shrinks them towards a common profile:
  $a^{(\text{type})}_{k} \sim \mathcal{N}(\bar a_k, \sigma_a^2)$.
- $\tilde T^{\ast}_t$ is a **thermally smoothed** temperature, an exponential moving average
  $\tilde T^{\ast}_t = \lambda \tilde T^{\ast}_{t-1} + (1-\lambda)\tilde T_t$ with $\lambda \in (0,1)$ inferred.
  It models building thermal inertia.
- $g(x)$ is a **smooth hinge**, $g(x) = \tfrac{1}{\kappa}\log(1+e^{\kappa x})$ (softplus), which approximates $\max(0, x)$
  but is differentiable, so NUTS works. $T_H$ and $T_C$ are the heating and cooling thresholds,
  **inferred** with priors such as $T_H \sim \mathcal{N}(15, 3^2)$ and $T_C \sim \mathcal{N}(21, 3^2)$, plus the ordering constraint $T_H < T_C$
  (parametrise $T_C = T_H + \Delta$ with $\Delta > 0$).
- $\beta_H, \beta_C > 0$ are the heating and cooling sensitivities in MW/°C. These have a direct
  interpretation and are worth reporting with credible intervals, and as they change over the years.
- $\gamma_j$ are holiday effects (national, regional-weighted, bridge, Christmas…), possibly
  hour-dependent through a small Fourier basis.

**Errors.** Hourly residuals are strongly autocorrelated:

$$
\varepsilon_t = \phi\,\varepsilon_{t-1} + \eta_t,\qquad \eta_t \sim \mathcal{N}\big(0, \sigma_{h(t)}^2\big),
$$

with $|\phi|<1$ and an **hour-dependent scale** $\sigma_{h(t)}$, since uncertainty differs between night and the evening peak.
A Student-$t$ for $\eta_t$ is a robustness option. Extensions are AR(2) or a seasonal AR term at lag 24.

For a day-ahead forecast the AR error decays across the 24-hour horizon. The posterior predictive is
obtained by simulating $\varepsilon$ forward from the last observed residual available at issue time.

**Inference.**

- Data size is about 8,760 points per year, so roughly 40–90k points in total. NUTS in NumPyro on CPU
  should be workable for a few years of data. The AR likelihood can be written in closed form as a conditional likelihood.
- Fall back to **SVI** (a low-rank multivariate normal guide) for fast iteration, and use NUTS for the final fits.
- Diagnostics: $\hat R$, effective sample size, divergences, prior and posterior predictive checks (ArviZ).
- Refit cadence for backtesting: refit monthly or quarterly, and warm-start or reuse the posterior in between,
  since NUTS on every rolling origin is too expensive.

**What this model contributes:** calibrated uncertainty, interpretable parameters (temperature
thresholds, sensitivities, holiday effects, the evolving midday-solar effect), and a principled way to
handle sparse special days through hierarchical pooling.

### 7.4 Sequence model (PyTorch RNN)

**Why include it, and the expected outcome.** RNNs (LSTM/GRU) are a natural fit for sequence
forecasting and are expected on an ML CV. On *this* kind of problem, however (strong deterministic
seasonality, a few dominant exogenous drivers, a few years of hourly data), well-tuned
gradient-boosted trees and structural models are typically **very hard to beat**, and deep models
often only match them. That is fine: an honest negative or neutral result, with an analysis of why,
is a better portfolio signal than a forced "deep learning wins". The value of the RNN here is
(a) to show the skill, (b) to provide a learned representation that the normalizing flow (§7.5) can be
conditioned on, and (c) to check whether it captures anything the others miss.

**Architecture (sequence-to-vector, direct multi-horizon):**

- **Encoder:** a GRU/LSTM over the past $L$ hours (for example $L = 168$, one week), with input at each step
  $[y_t, \tilde T_t, \text{calendar}_t]$ (demand standardised).
- **Known-future covariates** for the 24 target hours (calendar, temperature forecast), encoded by a
  small MLP or a second GRU.
- **Head:** combine the encoder state and the future-covariate encoding, and output for each target hour $j=1..24$ either
  - a **Gaussian head** $(\mu_j, \log\sigma_j)$ trained by negative log-likelihood, or
  - a **quantile head** $\{q_{j,\tau}\}$ trained with the summed pinball loss (non-parametric marginals).
- **Training:** chronological split, early stopping on a validation year, AdamW, gradient clipping,
  and seed averaging (deep models are high-variance; report the mean ± spread over seeds).
- **Alternatives** if time allows: a temporal convolutional network (TCN), or a small Transformer encoder.
  Keep the comparison fair, with the same inputs and the same backtest.

Framework: **PyTorch** (preferred) or TensorFlow/Keras. PyTorch has the better ecosystem for flows (§7.5).

### 7.5 Conditional normalizing flow over the 24-hour vector (stretch)

**Motivation.** Models §7.2–7.4 give good **marginal** distributions per hour, but the errors at
neighbouring hours are strongly correlated. If 18:00 comes in high, 19:00 almost certainly will too.
A planner who needs *scenarios for the whole day* (for example to size reserves or to plan storage)
needs the **joint** distribution

$$
p\big(\mathbf{y}_{d+1} \,\big|\, \mathbf{c}_d\big), \qquad \mathbf{y}_{d+1}\in\mathbb{R}^{24},
$$

where $\mathbf{c}_d$ is a context vector (the RNN encoder state from §7.4, or hand-built features).

**Model.** A conditional flow $f_\theta(\cdot\,;\mathbf{c})$: an invertible map
$\mathbf{y} \mapsto \mathbf{z}$ with $\mathbf{z}\sim\mathcal{N}(0, I_{24})$. Change of variables gives

$$
\log p(\mathbf{y}\mid\mathbf{c}) = \log \mathcal{N}\big(f_\theta(\mathbf{y};\mathbf{c});\,0, I\big)
  + \log\left|\det \frac{\partial f_\theta(\mathbf{y};\mathbf{c})}{\partial \mathbf{y}}\right|.
$$

- Architectures: **Masked Autoregressive Flow (MAF)**, or **neural spline flows** (rational-quadratic
  splines; more expressive per layer), with the context fed to the conditioner networks.
- Libraries: `zuko` or `nflows` (PyTorch), trained end-to-end with the encoder by maximum likelihood.
- **A residual parametrisation is easier to train:** model $\mathbf{r} = (\mathbf{y} - \hat{\mathbf{y}})/s$, where
  $\hat{\mathbf{y}}$ is a strong point forecast (for example from LightGBM) and $s$ a scale. The flow then only
  has to learn the *shape* and *dependence* of the errors, which is a much easier task on limited data.
- Data volume: about 365 training vectors per year, a few thousand in total, which is small for a flow.
  Keep it compact (few layers, weight decay, early stopping), consider adding data by sliding the
  24-hour window by a few hours, and compare against a **Gaussian copula** baseline
  (calibrated marginals + empirical rank correlation of residuals). The flow has to beat the copula to justify itself.

**Evaluation of joint forecasts** (in addition to the per-hour metrics):

- **Energy score** (the multivariate generalisation of the CRPS):
  $$
  \mathrm{ES}(F, \mathbf{y}) = \mathbb{E}_F\lVert \mathbf{X} - \mathbf{y}\rVert - \tfrac12\,\mathbb{E}_F\lVert \mathbf{X} - \mathbf{X}'\rVert .
  $$
- **Variogram score** of order $p$ (more sensitive to errors in the correlation structure):
  $$
  \mathrm{VS}_p(F,\mathbf{y}) = \sum_{i,j} w_{ij}\Big(|y_i - y_j|^p - \mathbb{E}_F|X_i - X_j|^p\Big)^2 .
  $$
- A downstream sanity check: the distribution of the **daily peak** and of **daily energy**
  $\sum_j y_j$ derived from the scenarios. Marginal-only models get these wrong.

### 7.6 The professional benchmark: REE's own forecast

REE publishes its demand forecast (indicators 544/545, to be verified). It is a **benchmark, not a
competitor to beat at all costs**:

- First establish exactly *when* each forecast is issued and for which horizon, so that the comparison is fair.
- Report all models' point errors against REE's on the same hours.
- Analyse **where** the open models lose or win (by hour, day type, temperature regime) and *why*.
- REE's forecast can also be used *as a feature*: a model that corrects REE's errors is a legitimate and
  realistic setup in industry. Keep that as a separate, clearly labelled experiment.

---

## 8. Evaluation protocol

### 8.1 Backtesting

- **Rolling-origin evaluation** (also called "time-series cross-validation"):
  for each origin day $d$ in the test period, train on data up to the issue time of $d$ and forecast all 24 hours of $d+1$.
- Splits (indicative):
  - **train:** start → end of year $Y-2$
  - **validation** (model selection, tuning, early stopping): year $Y-1$
  - **test** (touched once, at the end): year $Y$, preferably a full year so that every season is covered.
- Refit cadence: daily for cheap models (baselines, LightGBM if fast), weekly or monthly for the
  NumPyro and deep models. Document it, because it affects the comparison.
- **Never shuffle.** Never tune on the test year.

### 8.2 Point-forecast metrics

On the median (or mean) forecast:

- MAE, RMSE.
- MAPE, which is safe here since demand is far from 0 and is the metric operators commonly quote.
- Skill score against the seasonal naive and against REE.

### 8.3 Probabilistic metrics

- **CRPS** per hour, estimated from $M$ predictive samples:
  $$
  \widehat{\mathrm{CRPS}} = \frac{1}{M}\sum_{m}|x_m - y| - \frac{1}{2M^2}\sum_{m,m'}|x_m - x_{m'}| ,
  $$
  (an $O(M\log M)$ form exists after sorting the samples), or from quantiles via the average pinball loss:
  $\mathrm{CRPS} \approx \frac{2}{|\mathcal{T}|}\sum_{\tau\in\mathcal{T}}\rho_\tau(y, q_\tau)$.
- **Calibration:**
  - empirical coverage of central 50%, 80% and 90% intervals, by hour and by season;
  - **PIT histogram**: $u_t = F_t(y_t)$ should be roughly $\mathcal{U}(0,1)$;
  - reliability diagram of quantiles.
- **Sharpness:** average interval width at nominal coverage. Calibration comes first, then sharpness.
- **Joint:** energy score and variogram score (§7.5).

### 8.4 Comparing models statistically

Differences in average loss between models over one test year can be noise. Use the **Diebold–Mariano test** on the
loss differential $\delta_d = L^{(A)}_d - L^{(B)}_d$, computed per day:

$$
\mathrm{DM} = \frac{\bar\delta}{\sqrt{\widehat{\mathrm{LRV}}(\delta)/n}},
$$

with $\widehat{\mathrm{LRV}}$ a HAC (Newey–West) estimate of the long-run variance, since the daily differentials are autocorrelated.
Report it alongside a block-bootstrap confidence interval of $\bar\delta$.

### 8.5 Error analysis (the most important section of the write-up)

Break down errors by hour × day type, by temperature regime (cold / mild / hot), on holidays and bridge days, around DST
changes, and during the special periods in §10. For each failure mode, say whether it is a **data**
problem (missing information), a **model** problem (wrong structure) or **irreducible**.

---

## 9. Information leakage and other traps

1. **Observed vs. forecast weather.** At issue time on day $d$, tomorrow's temperature is **not known**,
   only *forecast*. Using observed $\tilde T$ for day $d+1$ gives an **oracle** upper bound on
   accuracy, not a realistic forecast. Plan:
   - Experiment A, "oracle": observed weather. This is an upper bound and isolates model quality.
   - Experiment B, "realistic": archived weather *forecasts* issued before the issue time (Open-Meteo's
     historical forecast archive, if its coverage is adequate), or observed weather plus synthetic noise
     matched to typical day-ahead temperature-forecast error (about 1–2 °C; estimate it if possible) as a fallback.
   - Report both. The gap between them is itself a result: *how much is weather uncertainty worth?*
2. **Demand lags.** At 10:00 on day $d$, demand for the rest of day $d$ is unknown. Only lags of
   at least (24 + hours until issue time) are available for some target hours. Build lags relative to the *issue time*, not the target time.
3. **Scalers and encoders fitted on all data.** Fit standardisation on the training window only, inside each backtest fold.
4. **Revised data.** Check whether REE revises historical demand values. If it does, the history you download today is not what was known at the time.
5. **Benchmark timing.** Make sure the REE forecast compared against was issued at a comparable time.
6. **DST.** Index in UTC (§4). Day-level vectors for §7.5 need a convention for 23- and 25-hour days: drop,
   interpolate, or pad with a mask. Document the choice.

---

## 10. Structural breaks and special periods

| Period | What happened | Handling options |
|---|---|---|
| Mar–Jun 2020 | COVID-19 lockdown; demand collapsed | Exclude from training, **or** add a regime indicator; never in the test set without discussion |
| 2022 | Energy-price crisis; demand response to high prices | Indicator, or rely on trend term; discuss |
| 28 Apr 2025 | Iberian blackout | Exclude the affected hours and days (measurement artefact, not demand) |
| Ongoing | Growth of rooftop (behind-the-meter) solar lowers measured midday demand; electrification (EVs, heat pumps) | Time-varying daily profile / trend; the Bayesian model can make the midday coefficients drift (e.g. a random walk over years) |

Detecting these in the data (for example with change-point analysis on residuals) and discussing them is
enough. They do not all need to be modelled. Being explicit about what is excluded and why is part of the deliverable.

---

## 11. Deliverables

1. **This repository:** reproducible end to end (`make data`, `make features`, `make train`, `make evaluate`).
2. **Write-up** (`reports/report.md`, possibly a blog post): 3–5 pages for a non-specialist audience,
   with a technical appendix. Structure: question → data → what drives demand → models → results vs. REE → failure analysis → what I would do next.
3. **Interactive dashboard** (Streamlit, or Plotly Dash):
   - pick any day in the test period; show realised demand, REE's forecast, and each model's median
     and 50/90% bands;
   - aggregate error panels (by hour, by month, by temperature);
   - for the flow: sample daily scenarios ("spaghetti plot");
   - deployable for free (Streamlit Community Cloud or Hugging Face Spaces), with a **live link in the README and on the CV**.
4. **Figures** good enough for a slide deck or interview: the demand–temperature hockey stick with
   posterior thresholds; the reliability diagram; the per-hour skill vs. REE.
5. A **"results at a glance" table** in the public `README.md` once results exist.

---

## 12. Repository layout

```
spain-load-forecast/
├── README.md                  ← public, non-expert overview
├── docs/
│   └── DESIGN.md              ← this document (internal reference)
├── pyproject.toml             ← dependencies, tool config (ruff, pytest)
├── Makefile                   ← make data | features | train | evaluate | app
├── .env.example               ← ESIOS_API_KEY=...   (real .env is git-ignored)
├── configs/
│   ├── data.yaml              ← indicators, cities, date ranges
│   ├── backtest.yaml          ← splits, issue time, refit cadence
│   └── models/                ← one YAML per model (hyper-parameters)
├── data/                      ← git-ignored except README/CHANGELOG
│   ├── raw/  interim/  processed/
│   └── CHANGELOG.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_results.ipynb
├── src/loadforecast/
│   ├── data/                  ← esios.py, redata.py, weather.py, calendar.py, build.py
│   ├── features/              ← fourier.py, lags.py, temperature.py
│   ├── models/                ← baselines.py, gbm.py, bayes_numpyro.py, rnn.py, flow.py, copula.py
│   ├── evaluation/            ← backtest.py, metrics.py (CRPS, pinball, ES, VS), dm_test.py
│   └── viz/
├── app/
│   └── streamlit_app.py
├── tests/                     ← data quality, metric correctness (CRPS vs. closed form for Gaussians), no-leakage tests
└── reports/
    ├── figures/
    └── report.md
```

**Leakage test idea:** a unit test that builds features for an origin $d$, then corrupts every value
with a timestamp after the issue time, and asserts the features are unchanged.

**Metric test idea:** for a Gaussian $\mathcal{N}(\mu,\sigma^2)$ the CRPS has a closed form,
$\mathrm{CRPS} = \sigma\left[z(2\Phi(z)-1) + 2\varphi(z) - \pi^{-1/2}\right]$ with $z=(y-\mu)/\sigma$.
Check that the sample-based estimator converges to it.

---

## 13. Tech stack

| Purpose | Tool |
|---|---|
| Language | Python ≥ 3.11 |
| Data | pandas or polars, pyarrow (Parquet), requests / `python-esios`, `holidays` |
| Classical ML | scikit-learn, LightGBM |
| Bayesian | **NumPyro** (JAX), ArviZ |
| Deep learning | **PyTorch** (preferred; TensorFlow/Keras acceptable), `zuko` or `nflows` for flows |
| Evaluation | own `metrics.py` (cross-checked against `properscoring` / `scoringrules`), statsmodels (HAC) |
| Visualisation | matplotlib, Plotly |
| App | Streamlit |
| Quality | ruff, pytest, pre-commit; optionally a GitHub Actions workflow running tests |
| Tracking | a simple CSV/JSON results log; MLflow only if it grows |

---

## 14. Plan and milestones

The **core** (M0–M5) is sized for about one focused week. **Extensions** (M6–M7) take another week or so.
Each milestone ends with something that runs and a commit.

| # | Milestone | Content | Done when |
|---|---|---|---|
| M0 | Setup (day 0) | ~~Request ESIOS token~~ (done); repo skeleton; environment | `make` targets exist |
| M1 | Data (days 1–2) | REData/ESIOS + weather + calendar pipeline; quality checks; EDA notebook | Clean hourly table for ≥ 4 years; EDA figures |
| M2 | Backtest harness + baselines (day 3) | Rolling-origin engine; metrics (with tests); seasonal naive, climatology; REE benchmark loaded | Baseline + REE scores on validation year |
| M3 | LightGBM (day 4) | Point + quantile models; SHAP | Beats seasonal naive clearly; calibrated quantiles |
| M4 | Bayesian model (days 5–6) | NumPyro model; SVI then NUTS; diagnostics; interpretable parameter plots | Posterior predictive scored in backtest |
| M5 | Write-up + dashboard v1 (day 7) | Results table, report draft, Streamlit app deployed | Public link works; README "results at a glance" |
| M6 | RNN (extension) | GRU/LSTM seq-to-vector with Gaussian/quantile head | Fair comparison incl. seed variance |
| M7 | Flow (extension) | Conditional spline flow on residuals; Gaussian-copula baseline; energy & variogram scores | Joint-scenario comparison + scenario plots in app |
| M8 | Phase 2 (optional) | Day-ahead price (§17) | — |

**After M5 the project can already go on the CV.** Update the dashboard and README as M6–M7 land.

---

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| ESIOS access issues (token revoked, rate limits, API changes) | Cache raw responses; REData (open) as fallback; design `data/` so the source can be swapped |
| Indicator semantics unclear (esp. REE forecast horizon) | Check documentation and inspect the data; label the comparison clearly; email REE if needed |
| No archive of historical weather forecasts | Oracle + noise-injection fallback (§9); state the limitation |
| NUTS too slow on full data | SVI for iteration; fewer Fourier terms; fit on a subsample of years; refit less often |
| Deep models don't beat LightGBM | Expected and fine; report honestly, analyse where they differ |
| Flow overfits small data | Residual parametrisation, small model, copula baseline as the bar to clear |
| Scope creep | The core (M0–M5) is the CV deliverable; everything else is optional |

---

## 16. Open decisions

- [ ] Issue time (e.g. 10:00 local on day $d$) and therefore the exact information set.
- [ ] Data period (start year; how to treat 2020).
- [ ] Weather source: Open-Meteo vs. AEMET (or both), and the city list and weights.
- [ ] PyTorch vs. TensorFlow for §7.4–7.5 (leaning PyTorch).
- [ ] Gaussian vs. quantile head for the RNN.
- [ ] Polars vs. pandas.
- [ ] Repository name and licence (MIT suggested), public from day 1?

---

## 17. Phase 2: day-ahead price forecasting

Once demand is done, the same infrastructure can forecast the **day-ahead market price** (ESIOS
indicator 600, to be verified; OMIE is the market operator). It is a much harder and more valuable
problem:

- Drivers: residual demand (demand − wind − solar), gas prices (TTF/MIBGAS) and CO₂ allowance prices, hydro reserves,
  interconnection flows, and the Iberian gas cap (*mecanismo ibérico*, 2022–2023).
- Features: renewable generation *forecasts* (ESIOS indicators), fuel prices (separate sources).
- Distinctive behaviour: heavy tails, near-zero or zero prices on sunny and windy days (solar cannibalisation), regime changes.
- Natural fit for the probabilistic machinery above: quantile models, Student-$t$ likelihoods, flows.

---

## 18. References and links

**Data and APIs**

- REData API: https://www.ree.es/en/apidatos
- ESIOS: https://www.esios.ree.es — API base `https://api.esios.ree.es`
- `python-esios`: https://github.com/datons/python-esios
- Open-Meteo: https://open-meteo.com (verify the historical forecast archive)
- AEMET OpenData: https://opendata.aemet.es
- INE (population): https://www.ine.es
- `holidays` package: https://pypi.org/project/holidays/

**Methods (background reading)**

- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.), free online. Time-series CV, baselines.
- Gneiting & Raftery (2007), "Strictly proper scoring rules, prediction, and estimation", *JASA*. CRPS, energy score.
- Scheuerer & Hamill (2015), "Variogram-based proper scoring rules for probabilistic forecasts of multivariate quantities", *MWR*.
- Diebold & Mariano (1995), "Comparing predictive accuracy", *JBES*.
- Hong & Fan (2016), "Probabilistic electric load forecasting: a tutorial review", *IJF*.
- Papamakarios et al. (2021), "Normalizing flows for probabilistic modeling and inference", *JMLR*.
- Durkan et al. (2019), "Neural spline flows", *NeurIPS*.
- Salinas et al. (2020), "DeepAR: probabilistic forecasting with autoregressive recurrent networks", *IJF*. A useful reference design for §7.4.
