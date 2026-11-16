# data/

Everything here except this file and `CHANGELOG.md` is git-ignored.

| Folder | Content |
|---|---|
| `raw/` | Exact API responses (gzip-compressed JSON), never edited. `raw/esios/<indicator>/`, `raw/openmeteo/<city>/` |
| `interim/` | Parsed, typed, UTC-indexed tables, one per source (Parquet) |
| `processed/` | The merged hourly modelling table with features (Parquet) |
| `features/` | Cached feature table for model training (observed weather), named by dataset hash; rebuilt automatically when missing |
| `predictions/<split>/` | One Parquet per model with its hourly point and quantile forecasts, written by `make evaluate` |

Rebuild with `make data`. Record every manual fix in `CHANGELOG.md`.

## Processed table: `processed/hourly.parquet`

One row per hour, indexed by `time_utc` (start of the hour). `hourly.meta.json` holds the content
hash and a quality summary.

| Columns | Meaning |
|---|---|
| `local_date`, `local_hour`, `utc_offset_h` | Europe/Madrid calendar position of the hour |
| `demand_mw` | **Target.** Hourly mean peninsular demand, pre-December-2025 definition |
| `demand_published_mw`, `selfcons_mw`, `post_definition_change` | Demand as published, REE's self-consumption estimate (from 2025-12-11), and the flag |
| `demand_n_obs`, `demand_spike` | Sub-hourly values behind each mean (6 or 12); robust spike flag |
| `ree_forecast_mw`, `ree_forecast_published_mw` | REE day-ahead forecast (ESIOS 1775), corrected and as published |
| `temp_obs`, `rh_obs`, `rad_obs`, `cloud_obs`, `wind_obs` | Population-weighted observed weather (reanalysis) |
| `temp_obs_<city>` | Observed temperature per city |
| `*_fc1`, `*_fc2` | Population-weighted forecast issued 1 and 2 days before the hour |
| `holiday_national`, `holiday_intensity`, `bridge_intensity`, `weekday`, `day_type`, `christmas`, `august`, `holy_week`, `local_hours` | Calendar, by local date |
| `is_masked` | Days excluded from training and scoring (blackout) |
