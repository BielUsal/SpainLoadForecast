# Day-ahead probabilistic forecasting of Spanish electricity demand

> **Internal design document.** This is the working reference for the project: the question, the data,
> the models, how they will be evaluated, the plan and the open decisions. It is written for the people
> (and assistants) building the project, not for visitors; the public-facing overview is the top-level
> [`README.md`](../README.md). Keep this file current as decisions change, and record each decision in §16
> rather than letting the document drift from the code.
>
> **Status:** M0 to M4 done and scored on the validation year only (2026-11-13). M5 is drafted
> (2026-11-16): `reports/report.md`, the Streamlit app in `app/` and the README results table, all on
> validation numbers. Left for M5: commit and push, deploy the app, link it in the README, and score
> the test year once. Open points left by M4 are listed at the end of §7.3.
> Read §10 on the December 2025 definition change before working with the data.

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
- Period: **1 January 2021 onwards** (decided, §16). The 2020 lockdown is left out entirely, and the
  data stays in the recent regime (rooftop solar, post-COVID). That gives almost 6 years.

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

ESIOS indicators, **verified against the live API on 2026-11-12** (`scripts/spike_esios.py`). All
return geography 8741 ("Península").

| ID | ESIOS name | Resolution | Role |
|---|---|---|---|
| 1293 | Demanda real Peninsular (MW) | 10 min in Jan 2021, 5 min from mid-2022 at the latest | **Target** $y_t$, as hourly means |
| 1775 | Previsión diaria D+1 demanda | Hourly | **Benchmark.** Published once, at about **10:50 local on day $d$** for the 24 hours of $d+1$, weekends included (checked on 7 dates from 2021 to 2026). MAPE about 1.15% in the two sample weeks. |
| 544 | Demanda prevista Peninsular (MW) | Same as 1293 | Not a day-ahead forecast: it is updated intraday (MAPE 0.4–0.8% in the sample weeks). Do not use as benchmark. |
| 460 | Previsión diaria de la demanda eléctrica peninsular | Hourly in 2021, 15 min since 2022 at the latest | Also overwritten during the target day (last update about 22:30 of the day itself). Do not use as benchmark. |
| 545 | Demanda programada Peninsular (MW) | Same as 1293 | *Scheduled* demand, not a forecast. Used only to derive the self-consumption estimate (below). |
| 1941 | Demanda Programada Total Peninsular (MW) | Same as 1293; **starts 2025-12-11** | Scheduled demand plus REE's estimate of self-consumption. `1941 − 545` is the hourly self-consumption estimate. |
| 1776 | Previsión intradiaria H+3 demanda | not checked | Possible extra comparison. |
| 600 | Day-ahead / PVPC price (not verified) | | Phase 2 target |
| 10034, 10035, … | Wind / solar generation and forecasts (not verified) | | Optional covariates; Phase 2 |

> ⚠️ **Definition change on 2025-12-11.** From that local date, 1293 includes REE's estimate of
> self-consumption (mostly rooftop solar; 5.5% of demand on average, up to 5.8 GW at midday), and from
> 2025-12-12 so does the forecast 1775. Earlier history was not revised. The pipeline subtracts
> `1941 − 545` from both, so `demand_mw` and `ree_forecast_mw` keep the pre-change definition throughout;
> the published values are kept in `demand_published_mw` and `ree_forecast_published_mw`. See §10.

Notes from the spike:

- Each value carries `datetime` (local, with offset) and `datetime_utc`; timestamps label the **start**
  of the interval.
- `time_trunc=hour` on 1293 **sums** the sub-hourly values instead of averaging them. Download at
  native resolution and average in the pipeline.
- `values_updated_at` in the response gives the publication time of the latest value in the request,
  which is how the 1775 publication time was established.

**Token:** obtained. Store it in `.env` as `ESIOS_API_KEY` (git-ignored); never commit it.

### 3.2 Weather

Demand in Spain is driven mostly by temperature: heating in winter, air conditioning in summer,
which is increasingly dominant.

| Option | Notes |
|---|---|
| **Open-Meteo** historical archive API | Free, no key, hourly reanalysis-based variables at any coordinates. It also offers archives of past *forecasts*, which matter for leakage (§9). **Chosen source** (§16); coverage checked on 2026-11-12, see below. |
| **AEMET OpenData** | Spain's official met agency; free API key. Station observations. Noisier and messier to process, but "official". |

Open-Meteo coverage, checked for Madrid on one week per period (`scripts/spike_openmeteo.py`):

| Endpoint | What it holds | Coverage found |
|---|---|---|
| `archive-api` | Reanalysis (observed weather) | Complete for every week tested, 2021–2026 |
| `historical-forecast-api` | Stitched short-lead forecasts, no fixed lead time | Complete 2021–2026, but not usable as a day-ahead forecast because the lead is not controlled |
| `previous-runs-api` | Forecast as issued 1, 2, … days before the valid time (`temperature_2m_previous_day1`, `_day2`) | Empty in Jan 2021; complete in Jul 2022, Jul 2023, Nov 2024, Jul 2025, Oct 2026; **partial in Jan 2024** (60 of 168 hours). Validation and test periods look covered; the exact start date and the gaps must be mapped in M1. |

Full-period results from M1 (all 10 cities, population-weighted; `notebooks/01_eda.ipynb` §12):

- Temperature forecasts start on 2021-03-24 and have one gap, 2023-12-30 → 2024-01-20. The other
  variables (radiation, humidity, cloud cover, wind) start only on 2024-01-20.
- **The archive is not homogeneous.** National temperature MAE against reanalysis is 1.1–1.2 °C with a
  warm bias of about +1 °C until the end of 2023, and 0.5–0.65 °C with almost no bias from 2024 (the
  gap marks the change). Lead 1 and lead 2 days are almost equally accurate.

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
   **means**, since demand is a power in MW. Convention: every timestamp labels the **start** of its
   interval, as ESIOS does. (1293 moved from 10-minute to 5-minute values on 2022-05-23.)
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

Information set $\mathcal{F}_{d}$: everything known at the **issue time** on day $d$, fixed at
**10:00 Europe/Madrid** (decided, §16), before the day-ahead market closes at 12:00. The last known
demand value is the hour starting 09:00 on day $d$, so the target hours start 14 to 37 hours after the
issue time. All features must belong to $\mathcal{F}_{d}$.

| Family | Features |
|---|---|
| Calendar | hour, weekday, day-of-year (as Fourier terms), month, DST flag, holiday intensity $h_t$, bridge-day flag, days to/from holiday, special periods (Christmas, Holy Week, August) |
| Weather | $\tilde T_t$ for the *target* hours (see §9 on forecast vs. observed), lags and rolling means of $\tilde T$ (buildings have thermal inertia), heating/cooling degree hours, radiation or cloud cover |
| Autoregressive | demand at the same hour 1, 2 and 7 days earlier, i.e. $y_{t-24}$, $y_{t-48}$, $y_{t-168}$, but only lags available at issue time; daily mean and peak of the last available day |
| Trend | a slow trend term, or a rolling 365-day mean of demand |

**Pointers from the baseline error analysis** (`notebooks/02_baselines.ipynb` §6, validation year):

- **Solar radiation** deserves a place among the main features, not the optional ones. REE's
  midday error correlates 0.54 with how sunny the day is for the season (+1.2 GW on the sunniest
  tenth of days, −0.8 GW on the dullest; no effect at night), and 7 of its 10 worst days are one sunny
  spell, 25 March – 1 April 2025. Radiation forecasts exist from 2024-01-20 (§3.2).
- Lag features need the **calendar of the lagged day** next to them (holiday intensity one week
  and two days earlier): a sixth of the weekly naive's worst days are the days one week after a holiday.
- **Temperature change** relative to the lagged days, not only the level.

Fourier encoding of a period $P$ with order $K$:

$$
\phi_P(t) = \big[\cos(2\pi k t/P),\ \sin(2\pi k t/P)\big]_{k=1}^{K}.
$$

Typical choices are $P=24$ ($K\approx 6$–10), $P=168$ ($K\approx 3$–5) and $P=8766$ ($K\approx 4$–6).

**Orders from the spectrum** (`notebooks/01_eda.ipynb` §5, 2021–2024): use $K = 4$–6 for the day
(4 harmonics carry 47.0 of the daily cycle's 47.1% of variance), $K = 6$ for the week (22%), and only
$K = 2$–3 for the year (13%, almost all of it in the 6-month harmonic, 1.9 GW against 0.4 GW for the
12-month one). Higher yearly harmonics do not decay and mostly fit particular weather episodes, so
leave that variation to the temperature and holiday features. Weekly and yearly side-peaks around the
daily harmonics (3.8% of variance for the weekly ones) show that the daily shape depends on weekday
and season, which an additive Fourier model cannot represent.

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

**Implementation (M2).** `models/baselines.py` holds the point rules; `climatology_adjusted` is a
fourth variant rescaled to the level of the last 28 days. `models/quantiles.py` adds the quantiles:
empirical quantiles by hour of the forecast's own errors over the previous 365 days, using only
target days up to $D-2$ for a forecast of day $D$. The same function gives REE's forecast its bands.

**Validation-year results** (2024-11-01 → 2025-10-31; full table in `reports/results/validation.md`,
analysis in `notebooks/02_baselines.ipynb`):

| Model | MAE (MW) | MAPE (%) | CRPS (MW) | 90% coverage (%) |
|---|---:|---:|---:|---:|
| `naive_weekly` | 1,438 | 5.29 | 903 | 85.1 |
| `naive_daily` | 2,443 | 9.08 | 1,390 | 89.5 |
| `climatology` | 1,221 | 4.47 | 696 | 85.0 |
| `climatology_adjusted` | 1,234 | 4.53 | 696 | 84.2 |
| `lgbm_oracle` (§7.2, observed weather) | 523 | 1.94 | 308 | 89.1 |
| `lgbm_forecast` (§7.2, forecast weather) | 538 | 1.99 | 315 | 88.4 |
| `bayes_nuts_oracle` (§7.3, observed weather) | 580 | 2.13 | 353 | 85.4 |
| `bayes_nuts_forecast` (§7.3, forecast weather) | 586 | 2.15 | 359 | 84.7 |
| `bayes_svi_forecast` (§7.3, refitted every 28 days) | 587 | 2.15 | 358 | 85.5 |
| `ree` (benchmark) | 455 | 1.67 | 271 | 88.0 |

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

**Implementation and results (M3).** `features/build.py` builds 55 features from the engine's
(`history`, `future`, `gap`) frames; `models/gbm.py` holds the model; `configs/models/lightgbm.yaml`
the settings; `notebooks/03_lightgbm.ipynb` the analysis, including an exploration log.

- **One model, two weather inputs.** The model is trained on observed weather (§9) with squared
  error, 15 leaves, 2,000 rounds at rate 0.03, recency weights with a 2-year half-life, and is
  refitted every 28 days. `lgbm_oracle` is fed observed weather for the target day, `lgbm_forecast`
  the forecast archived two days before each hour.
- **Validation year:** `lgbm_forecast` MAPE 1.99%, MAE 538 MW, CRPS 315 MW; `lgbm_oracle` 1.94%,
  523 MW, 308 MW. That is 56% below the best baseline's MAE and 83 MW above REE's (95% interval 38
  to 130 MW, p = 0.0003). The model beats REE on 40% of days.
- **Weather uncertainty is worth only 15 MW of MAE** (oracle against forecast). It is not what
  separates the model from REE.
- **Deviation from the plan: quantiles are not from quantile regression.** One LightGBM per quantile
  covered 90.6% of its training data with the 90% band but 68.0% of the validation year (CRPS 372 MW),
  and 75–77% with early stopping and heavier regularisation. The quantiles are instead the point
  forecast plus empirical quantiles of the model's own out-of-sample errors (`models/quantiles.py`,
  the same method as the baselines): coverage 48.2 / 78.6 / 88.4% for 50 / 80 / 90%, CRPS 315 MW.
- **Where it trails REE:** night and morning hours (about 1.3 times REE's error), cold weather, and
  special periods (Christmas 3.42% against 1.85%, Holy Week 2.19% against 1.11%). **Where it leads:**
  unusually sunny days, where REE over-forecasts midday demand (late March 2025: 2.76% against 4.65%).
- **SHAP:** demand lags dominate (holiday-free 4-week reference, weekly lag), then calendar, weather
  and the recent level. Temperature has the expected U shape, about +800 MW below 8 °C and +1,000 MW
  above 28 °C.
- **Caveat:** features, settings and the quantile method were chosen on the validation year, so its
  scores are somewhat optimistic.

**What M4 should take from this.** The remaining gap is in sparse special days and in the level at
night, which is where a structural model with pooled holiday effects could add something that more
tree tuning did not (trend features, error correction, linear leaves, city-level temperatures and
more frequent refits all gave no gain).

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

**Guidance from the EDA** (`notebooks/01_eda.ipynb`, rough least-squares fits on daily means): thresholds
near $T_H \approx 16$ °C and $T_C \approx 20$ °C; sensitivities about 600–650 MW/°C (heating) and
670–710 MW/°C (cooling) on the daily mean; daily smoothing $\lambda \approx 0.6$; residual partial
autocorrelation dominated by lags 1 and 2 (0.97, −0.28), so start from **AR(2)** errors; and the
change over the years is mostly in the *shape* of the day (−2.7 GW at midday, +0.8 GW at 20:00–21:00
between 2021 and 2025), so the drift belongs in the daily-profile coefficients.

**What this model contributes:** calibrated uncertainty, interpretable parameters (temperature
thresholds, sensitivities, holiday effects, the evolving midday-solar effect), and a principled way to
handle sparse special days through hierarchical pooling.

**Implementation and results (M4).** `models/bayes_numpyro.py` holds the model, `models/bayes.py`
the fitting and the backtest wrapper, `configs/models/bayes.yaml` the settings, and
`notebooks/04_bayes.ipynb` the analysis. The model as built differs from the specification above in
three ways:

- **A direct day-ahead error model, not an AR on hourly errors.** The 24 hours of a day are
  multivariate Student-t around the structural mean plus a correction built from the residuals
  known at 10:00 (this morning's mean, the last full day's mean, the same hour 2, 7, 14 and 21 days
  back). It is fitted on the 14-to-38-hour-ahead distribution the forecast needs.
- **Two stages, not one joint fit.** Structure first, then the error model on its residuals.
  Fitted jointly, the correction absorbed the level of demand and left the structural mean biased
  by 2.3 GW. The cost: stage 1 treats days as independent, so its credible intervals are too narrow.
- **The drift is linear, not a random walk:** a piecewise-linear level, a linear drift of the daily
  shape, and a radiation effect that grows linearly in time.

Two posteriors are backtested: one NUTS fit on the training period, kept fixed (`bayes_nuts_*`;
4 chains × 500 draws, 65 minutes), and a posterior mode plus low-rank Gaussian approximation
refitted every 28 days (`bayes_svi_*`; 95 seconds per fit).

- **Validation year:** `bayes_nuts_forecast` MAPE 2.15%, MAE 586 MW, CRPS 359 MW; with observed
  weather 2.13%, 580 MW, 353 MW. That is 131 MW above REE's daily MAE (95% interval 76 to 184) and
  48 MW above LightGBM's (4 to 92, p = 0.03). It beats REE on 36% of days and LightGBM on 46%.
- **It does not earn its place on accuracy.** It is behind LightGBM in every block of the day and
  in special periods (Christmas 4.92% against 3.42%, Holy Week 3.04% against 2.19%), so pooled
  holiday effects did not close the gap §7.2 pointed to. It is ahead only on Sundays (1.82% against
  2.04%). Its case is interpretability.
- **Parameters** (posterior mean, 90% interval): heating below 16.1 °C (15.3–17.0), cooling above
  20.3 °C (20.0–20.7); 448 MW/°C heating and 846 MW/°C cooling on average over the day, peaking at
  640 MW/°C (20:00) and 1,230 MW/°C (16:00); temperature half-life 19 hours. A national holiday on
  a Wednesday removes 4.6 GW on average. The radiation effect grows by 1.0 GW per kW/m² per year
  (about −2.9 GW on a clear midday at the start of 2025).
- **The day-ahead correction** takes the structural mean from 2.72% to 2.15% MAPE.
- **Bands are too narrow:** 47.9 / 75.4 / 84.7% coverage for 50 / 80 / 90%, against LightGBM's
  88.4% at the same width.
- **SVI forecasts as well as NUTS** (difference in daily MAE +1 MW, interval −2 to +5), but its
  posterior standard deviations are a median of 0.34 times the NUTS ones. Use NUTS for parameter
  intervals.
- **Sampler:** no divergences; the error stage mixed well. The structure stage hit the tree-depth
  limit (8) on every draw; the raw hierarchical profile terms have $\hat R$ up to 1.11, while the
  weekday profiles and temperature parameters have $\hat R \le 1.03$ and ESS of 150 or more.

**Open points left by M4:**

- The forecast-weather variant has a bias of −118 MW against −16 MW with observed weather, although
  the weather forecasts are almost unbiased over the year (+0.17 °C, +1 W/m²). Not explained.
- A cleaner NUTS fit (a centred parametrisation of the weekday profiles, or a higher tree depth)
  before any parameter interval goes into the report. It costs at least another hour of sampling.
- Recalibrating the bands from the model's own past errors, as for LightGBM, would probably fix
  the coverage; not tried.

**Future idea: sequential updating (logged 2026-11-13, not planned).** Proposed on epistemic
grounds, not for accuracy: the posterior should absorb each new hour or day as an update, with
yesterday's posterior as today's prior, in place of a periodic refit on all the data.

- *Today:* the NUTS fit is frozen at 2024-10-30; the SVI variant is refitted from scratch on all
  data since 2021 every 28 days. New data reaches the daily forecast only through the residuals of
  the last 30 days, not through the parameters.
- *What stands in the way:* NUTS returns samples, which cannot be reused as a prior without an
  approximation (a Gaussian fitted to the posterior, or reweighting the samples day by day).
- *What makes it feasible:* about 200 of the 216 structural parameters are linear coefficients.
  With the few non-linear ones held fixed (temperature thresholds, half-life, hinge sharpness,
  noise parameters; re-estimated by NUTS now and then), the rest can be updated exactly by a Kalman
  filter, per day or per hour.
- *What it would change:* as a filter the coefficients can drift, so the level becomes a latent
  state (replacing the two-stage fit and the hand-built correction) and the solar and daily-shape
  drift are tracked, not extrapolated linearly. This is the random-walk drift first planned above.
- *Expectation:* with fixed parameters it gives the same posterior as a full refit, and a year of
  extra data did not change the scores (SVI against frozen NUTS: 1 MW), so little accuracy gain
  is expected from updating alone. Any gain would come from the drifting states.
- *First test, if taken up:* a Kalman filter on the linear coefficients with the temperature
  parameters fixed at their NUTS means, backtested on the validation year. About half a day.

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

REE publishes its day-ahead demand forecast as ESIOS indicator **1775** (verified, §3.1). It is a
**benchmark, not a competitor to beat at all costs**:

- Timing: 1775 is published at about 10:50 local on day $d$ for all of $d+1$. Our issue time is 10:00,
  so our models see 50 minutes *less* information than REE. The comparison is fair and slightly
  conservative for us. Indicators 544 and 460 are revised intraday and must not be used here.
- Report all models' point errors against REE's on the same hours.
- Analyse **where** the open models lose or win (by hour, day type, temperature regime) and *why*.
- REE's forecast can also be used *as a feature*: a model that corrects REE's errors is a legitimate and
  realistic setup in industry. Keep that as a separate, clearly labelled experiment.

---

## 8. Evaluation protocol

### 8.1 Backtesting

- **Rolling-origin evaluation** (also called "time-series cross-validation"):
  for each origin day $d$ in the test period, train on data up to the issue time of $d$ and forecast all 24 hours of $d+1$.
- Splits, by target day (decided, §16; exact bounds to be confirmed once the data is downloaded):
  - **train:** 2021-01-01 → 2024-10-31
  - **validation** (model selection, tuning, early stopping): 2024-11-01 → 2025-10-31. It contains the
    28 April 2025 blackout; the affected days are masked from training and scoring (§10).
  - **test** (touched once, at the end): 2025-11-01 → 2026-10-31, the latest full 12 months, so every
    season is covered and the archive of weather forecasts is at its best (§9).
- Refit cadence: daily for cheap models (baselines, LightGBM if fast), weekly or monthly for the
  NumPyro and deep models. Document it, because it affects the comparison.
- **Never shuffle.** Never tune on the test year.

**Implementation (M2).** `evaluation/backtest.py` builds, for every target day, the pair
(`history`, `future`): `history` is every row whose hour started before the issue time, and `future`
is the target day restricted to columns known in advance (calendar by default; weather columns are
added per experiment). Models only ever receive these two frames, so reading tomorrow's demand is
impossible by construction, and `tests/test_backtest.py` checks it by corrupting everything after the
issue time. Demand on masked days is hidden from models as well as from scoring.
`python -m loadforecast.evaluation.run --split validation` (`make evaluate`) runs all models, saves
predictions under `data/predictions/` and scores under `reports/results/`. The test split needs
`--final`.

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

**Common output format (decided in M2).** Every model delivers a point forecast and the seven
quantiles 5, 10, 25, 50, 75, 90, 95% per target hour. The headline CRPS is the quantile
approximation over those seven levels, identical for all models, so it is comparable across them but
coarser than a sample-based CRPS (it ignores the tails beyond 5% and 95%). Sample-based CRPS, the
energy score and the variogram score are implemented and tested in `evaluation/metrics.py` for the
models that produce samples.

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
   - Spike result (§3.2): Open-Meteo's previous-runs archive has lead-specific forecasts from some
     point in 2021–22 onwards, so experiment B is feasible for validation and test. A forecast "1 day
     before" a target hour $h$ of $d+1$ was issued around hour $h$ of day $d$, which is after the 10:00
     issue time when $h \ge 10$. So use the 2-day-lead value for target hours $h \ge 10$ and the
     1-day-lead value only for the early hours (or 2-day lead everywhere, as the conservative choice).
     For training years without archived forecasts, train on reanalysis.
   - M1 result: because the forecast archive has a +1 °C bias before 2024 (§3.2), **train on
     reanalysis throughout** and feed forecast weather only at prediction time in validation and test,
     where the archive is homogeneous. Non-temperature forecast variables exist only from 2024-01-20.
2. **Demand lags.** At 10:00 on day $d$, demand for the rest of day $d$ is unknown. For a target hour
   $h$ of day $d+1$, the 24-hour lag $y_{t-24}$ is hour $h$ of day $d$, which is known only for
   $h \le 9$; for $h \ge 10$ the most recent same-hour value is $y_{t-48}$. Build lags relative to the
   *issue time*, not the target time.
3. **Scalers and encoders fitted on all data.** Fit standardisation on the training window only, inside each backtest fold.
4. **Revised data.** Check whether REE revises historical demand values. If it does, the history you download today is not what was known at the time.
5. **Benchmark timing.** Make sure the REE forecast compared against was issued at a comparable time.
6. **DST.** Index in UTC (§4). Day-level vectors for §7.5 need a convention for 23- and 25-hour days: drop,
   interpolate, or pad with a mask. Document the choice.

---

## 10. Structural breaks and special periods

| Period | What happened | Handling options |
|---|---|---|
| Mar–Jun 2020 | COVID-19 lockdown; demand collapsed | **Decided:** outside the data period (data starts in 2021), so no handling is needed |
| 2022 | Energy-price crisis; demand response to high prices | Indicator, or rely on trend term; discuss |
| 28 Apr 2025 | Iberian blackout | **Decided:** mask the affected days from training and from validation scoring (measurement artefact, not demand); the exact days are set in EDA from the recovery profile |
| **11 Dec 2025** | **REE changed the definition of published demand** to include estimated self-consumption; history not revised | **Decided (provisional, §16):** subtract REE's own estimate (`1941 − 545`) so the target keeps the pre-change definition. Verified: no jump against scheduled demand (mean gap 21 MW before, 24 MW after), estimate is zero at night and correlates 0.91 with radiation. REE's errors in MW are unchanged by the correction. The forecast for 11 Dec itself was issued under the old definition and is left uncorrected. |
| 11 Jun 2026 | Demand fell about 7 GW below REE's forecast in the evening (cause not investigated); in the test period | **Open:** mask or keep. Decide before any model is scored on the test period. |
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
│   └── models/                ← one YAML per model (hyper-parameters); lightgbm.yaml so far
├── data/                      ← git-ignored except README/CHANGELOG
│   ├── raw/  interim/  processed/
│   └── CHANGELOG.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_lightgbm.ipynb
│   └── …                      ← one per model, then a final results notebook
├── src/loadforecast/
│   ├── data/                  ← esios.py, weather.py, calendar.py, quality.py, build.py
│   ├── features/              ← build.py (all issue-time features in one place)
│   ├── models/                ← baselines.py, quantiles.py, gbm.py, bayes_numpyro.py, rnn.py, flow.py, copula.py
│   ├── evaluation/            ← backtest.py, metrics.py (CRPS, pinball, ES, VS), dm_test.py, scoring.py, run.py
│   └── viz/
├── scripts/                   ← spike_esios.py, spike_openmeteo.py (live API checks)
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
| Data | pandas, pyarrow (Parquet), requests, `holidays` |
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

## 16. Decisions

### 16.1 Decision log

| Date | Topic | Decision | Reason |
|---|---|---|---|
| 2026-11-12 | Issue time | 10:00 Europe/Madrid on day $d$ | Before the 12:00 day-ahead gate closure. Revisit only if REE's D+1 forecast is published at a very different time. |
| 2026-11-12 | Test period | Latest 12 months, 2025-11-01 → 2026-10-31 | Avoids the April 2025 blackout, has the best weather-forecast archive coverage, and is the most current result. |
| 2026-11-12 | Validation period | 2024-11-01 → 2025-10-31, blackout days masked | The 12 months before the test period. |
| 2026-11-12 | Data period | 2021-01-01 onwards; 2020 not used | Skips the lockdown and stays in the recent regime. Cost: features needing a year of history (rolling 365-day mean, climatology) are only valid from 2022; choose a shorter trend window or accept this in M1. |
| 2026-11-12 | Weather source | Open-Meteo only (reanalysis archive + forecast archive) | One free source without a key that covers both the oracle and the realistic experiment (§9). AEMET is dropped unless Open-Meteo falls short. |
| 2026-11-12 | Data frames | pandas | Best fit with LightGBM, statsmodels and `holidays`; the data is small. |
| 2026-11-12 | Deep-learning framework | PyTorch | Better ecosystem for flows (§7.5). |
| 2026-11-12 | ESIOS access | Own thin client on `requests` with a raw-response cache | Full control over caching and retries; one dependency fewer. |
| 2026-11-12 | Environment | `venv` + `pip` driven by `pyproject.toml` (`make setup`) | `uv` is not installed on the development machine; the `pyproject.toml` works with either. |
| 2026-11-12 | Disk budget | Estimate the size of the full dataset before any bulk download | The development machine is short on disk. |

| 2026-11-12 | REE benchmark | ESIOS indicator 1775 (D+1 forecast, published about 10:50 on day $d$) | Verified in the spike; 544 and 460 are revised intraday, 545 is scheduled demand (§3.1). |
| 2026-11-12 | Raw cache format | gzip-compressed JSON, native resolution | ESIOS hourly truncation sums instead of averaging; gzip shrinks responses about 16 times. |

| 2026-11-12 | Target definition | `demand_mw` = published demand minus REE's self-consumption estimate from 2025-12-11 (pre-change definition) | Keeps one definition across training, validation and test without moving the test period (§10). **Provisional:** taken during M1 without discussion; the alternative is to end the test period on 2025-12-10. |
| 2026-11-12 | Blackout mask | 28–29 April 2025 | 37% and 16% below REE's forecast; 30 April is 3% below, within normal range. |
| 2026-11-12 | Training weather | Reanalysis for training; forecast weather only at prediction time in validation and test | The forecast archive has a +1 °C bias before 2024 (§3.2). |
| 2026-11-12 | Trend features | Trailing means by hour over a few weeks plus lagged demand, not a 365-day mean | The level drifts under 1 GW per year but the daily shape changes a lot; also avoids a year of warm-up. |

| 2026-11-12 | Forecast output format | Point forecast plus 7 quantiles per hour; headline CRPS from those quantiles | One format every model family can produce, so scores are comparable (§8.3). |
| 2026-11-12 | Quantiles for point forecasts | Empirical quantiles by hour of the forecast's own errors over the previous 365 days | Simple, uses only information available at issue time, and applies equally to REE's forecast. Under-covers slightly (84–89% for a nominal 90%). |
| 2026-11-12 | Skill reference | `naive_weekly` | The standard seasonal-naive reference; the climatology is the stronger bar to beat. |

| 2026-11-12 | LightGBM quantiles | Point forecast (squared error) plus quantiles of its own past out-of-sample errors; no quantile regression | Quantile regression was overconfident out of sample: 68% coverage for a nominal 90%, CRPS 372 against 315 MW (§7.2). |
| 2026-11-12 | Forecast-weather inputs | Two-day-lead forecast for all target hours; one-day-lead for the rest of the issue day | Always issued before 10:00 on the issue day; the two leads are almost equally accurate (§3.2). |
| 2026-11-12 | Refit schedule | Every 28 days, on fixed calendar days | Calendar anchoring lets model variants share fitted models; retraining more often gave no gain. |
| 2026-11-12 | SHAP | LightGBM's built-in TreeSHAP (`pred_contrib`), no `shap` package | Same values, one dependency fewer. |

| 2026-11-13 | Bayesian error model | Direct day-ahead model of the 24-hour residual vector, fitted in a second stage | A one-step AR is not fitted on the horizon the forecast needs; a joint fit let the correction absorb the level (§7.3). |
| 2026-11-13 | Bayesian inference | One fixed NUTS fit for parameter intervals; SVI refitted every 28 days as the cheap forecasting variant | The two forecast equally well; SVI's posterior is far too narrow (§7.3). |

### 16.2 Data size estimate (2026-11-12)

Measured on the spike's sample weeks and extrapolated to 2021-01 → 2026-10 (about 304 weeks):

| Item | Basis | Estimate |
|---|---|---|
| ESIOS 1293, raw gzip | 23.5 kB per week at 5 min | about 7 MB |
| ESIOS 1775, raw gzip | about 2–3 kB per week, hourly | under 1 MB |
| Open-Meteo reanalysis, 10 cities, 5 variables | 6.8 kB per city-week uncompressed | about 21 MB raw, about 5 MB gzip |
| Open-Meteo previous runs, 10 cities | similar per request | about 5–10 MB gzip |
| Interim Parquet (all sources) | 610k five-minute rows + hourly tables | about 10 MB |
| Processed hourly table | about 51k rows × about 100 features | about 15–40 MB |
| **Data total** | | **under 100 MB** |

The data is not the disk risk; the Python environment is. The core `.venv` is 392 MB today. Expected
additions: LightGBM + SHAP about 150 MB, JAX + NumPyro + ArviZ about 400 MB, PyTorch **CPU-only**
about 800 MB (the default CUDA build is 5 GB or more, so install the CPU wheel), Streamlit + Plotly
about 200 MB. Total about 2 GB. Backtest outputs (predictive samples) need a budget of their own at
M2: 365 days × 24 hours × 1,000 samples as float32 is 35 MB per model run, so store quantiles or a
few hundred samples.

**Actual after M1:** raw 16 MB, interim 14 MB, processed 4.6 MB (51,120 rows × 46 columns), so
35 MB in total. The environment is 585 MB with the notebook tools.

**Actual after M3:** `data/` is about 50 MB (adding predictions and the cached feature table) and the
environment 738 MB with LightGBM.

**Actual after M4:** `data/` is 55 MB (adding the saved NUTS fit, 4 MB, and four prediction files)
and the environment 1.2 GB with JAX, NumPyro and ArviZ.

### 16.3 Still open

- [ ] Final city list and weights for the national temperature (§3.2).
- [ ] Gaussian vs. quantile head for the RNN (decide at M6).
- [ ] Licence (MIT suggested).
- [x] Trend window: resolved in M1, see the decision log.
- [ ] Confirm the target definition after the December 2025 change (provisional decision above).
- [ ] Mask 11 June 2026 in the test period or keep it (§10).
- [ ] Replace the approximate population figures in `configs/data.yaml` with INE values.

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
