# How much electricity will Spain use tomorrow?

**Forecasting Spain's hourly electricity demand one day ahead, with honest uncertainty.**

> 🚧 Work in progress. Preliminary results are [below](#results-at-a-glance) and in the
> [report](reports/report.md); final test-year results and a live dashboard link will follow.

---

## The problem

Electricity can't easily be stored at grid scale, so every moment the power plants feeding the grid have
to produce almost exactly what the country consumes. To plan which plants run tomorrow, Spain's grid
operator, **Red Eléctrica (REE)**, needs to know in advance how much electricity people and businesses
will use, hour by hour.

Get it wrong and someone pays: expensive backup plants are switched on at short notice, or energy is
bought and sold at bad prices.

This project asks: **using only public data, how well can we forecast tomorrow's hourly demand, and how
well can we say how uncertain that forecast is?**

## Why uncertainty matters

A forecast of "31,000 MW at 8 pm" is useful. A forecast of "31,000 MW, and we're 90% sure it will be
between 29,800 and 32,100" is much more useful. It tells the operator how much backup to keep ready.

But a range is only worth something if it's **honest**: across many days, a "90% range" should contain
what actually happened about 90% of the time. This project checks that explicitly instead of taking it on
trust.

## What drives demand

- **The clock and the calendar.** People sleep, work and eat at regular times, so demand follows strong
  daily and weekly rhythms. Weekends, holidays, "bridge" days and August all look different.
- **Temperature.** Demand rises when it's cold (heating) and, increasingly in Spain, when it's hot
  (air conditioning). In between there's a comfortable range where temperature hardly matters.
- **Long-term change.** Rooftop solar panels reduce the electricity people draw from the grid around
  midday. Electric cars and heat pumps add new demand.
- **Unusual events.** The 2020 lockdown, the 2022 energy-price crisis and the April 2025 blackout all leave
  marks in the data that a careful analysis has to recognise.

## The approach

1. **Collect** several years of hourly demand from REE, weather for Spain's major cities, and the
   holiday calendar.
2. **Build forecasting models**, from simple to sophisticated:
   - *simple rules*, such as "tomorrow will look like the same day last week", to set the bar;
   - *machine learning* (gradient-boosted trees), the industry standard;
   - a *statistical model built around how demand actually works* (daily rhythms, heating and cooling),
     whose parameters can be read directly, for example the temperature at which air conditioning kicks in;
   - *deep learning* (recurrent neural networks, and models that generate realistic whole-day scenarios).
3. **Test them fairly**, the way they'd be used in practice: each day, using only information available at
   the time, forecast the next day. Then repeat over a full year the models have never seen.
4. **Compare against the professionals.** REE publishes its own forecast. The project measures how close
   open methods get, and digs into *when and why* they do better or worse.

## What you'll be able to see

- An **interactive dashboard**: pick any day and compare what actually happened with each model's forecast
  and its uncertainty range, alongside REE's own forecast.
- A short **report** explaining the results in plain language.
- All the **code**, so every number can be reproduced.

## Results at a glance

Preliminary, on the **validation year** (1 November 2024 to 31 October 2025), which was also used to
choose and tune the models, so these scores are somewhat flattering. The final test year has not been
scored yet. Lower is better, except coverage, which should be 90%.

| Forecast | Mean error (MW) | Mean error (%) | Probabilistic score, CRPS (MW) | Coverage of the 90% range (%) |
|---|---:|---:|---:|---:|
| Same hour last week | 1,438 | 5.30 | 903 | 85.1 |
| Typical profile | 1,221 | 4.47 | 696 | 85.0 |
| Bayesian model | 586 | 2.15 | 359 | 84.7 |
| LightGBM | 538 | 1.99 | 315 | 88.4 |
| **REE** | **455** | **1.67** | **271** | 88.0 |

- Both models cut the error of the best simple rule by more than half.
- REE stays ahead, by 83 MW on average against LightGBM. The gap is statistically real but small, and the
  open model wins on 4 days out of 10, most clearly on unusually sunny days.
- The uncertainty ranges are close to honest: LightGBM's "90% range" contained the real value 88% of the
  time.

REE publishes no uncertainty range; the one scored here is built from its own past errors. The full story,
with the failure analysis, is in [`reports/report.md`](reports/report.md).

---

## For technical readers

The full design (data sources and API indicators, model specifications, evaluation metrics, leakage
controls and project plan) is in [`docs/DESIGN.md`](docs/DESIGN.md).

**Stack:** Python · pandas · LightGBM · NumPyro · PyTorch · Streamlit

**Data:** [Red Eléctrica (ESIOS / REData)](https://www.ree.es/en/apidatos), public weather data, Spanish
public holiday calendar.

## Author

**Gabriel D'Andrade Furlanetto**, physicist and data scientist ·
[LinkedIn](https://www.linkedin.com/in/gabriel-d-andrade-furlanetto-3477232b1) ·
[GitHub](https://github.com/BielUsal)
