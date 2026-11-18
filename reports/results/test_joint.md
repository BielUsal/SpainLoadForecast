# Joint (whole-day) scores: test

Scenarios of the 24 hours built on the RNN's forecasts (DESIGN 7.5). Daily means;
energy score, CRPS of each hour, of the daily peak (MW) and of the daily energy (GWh);
variogram score of order 0.5. Lower is better. Differences: mean daily difference with
a 95% block-bootstrap interval and the Diebold-Mariano p-value.

| model | days | energy | variogram | crps_hourly | crps_peak | crps_energy_gwh | energy_minus_independent | energy_minus_independent_ci | energy_minus_independent_p | variogram_minus_independent | variogram_minus_independent_ci | variogram_minus_independent_p | energy_minus_copula | energy_minus_copula_ci | energy_minus_copula_p | variogram_minus_copula | variogram_minus_copula_ci | variogram_minus_copula_p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flow | 362 | 1,930.6 | 23,245.3 | 324.2 | 354.9 | 5.6 | -66.9 | -115.7 to -21.5 | 0.00456 | -3,606.8 | -4906.1 to -2384.2 | 2.49e-08 | -17.9 | -65.4 to 27.6 | 0.44 | -577.4 | -1973.6 to 749.8 | 0.403 |
| copula | 362 | 1,948.5 | 23,822.6 | 327.5 | 370.3 | 5.5 | -49.0 | -56.1 to -43.9 | 3.53e-57 | -3,029.4 | -3395.5 to -2681.7 | 2.8e-61 |  |  |  |  |  |  |
| independent | 362 | 1,997.5 | 26,852.0 | 327.4 | 358.8 | 6.5 |  |  |  |  |  |  | 49.0 | 43.9 to 56.1 | 3.53e-57 | 3,029.4 | 2681.7 to 3395.5 | 2.8e-61 |
