"""Features for one forecast, built only from what is known at the issue time.

``make_features(history, future, gap, weather)`` returns one row per target hour. It is the
single place where features are computed: the training set is made by calling it for past
target days (``build_dataset``), and the model calls it again at prediction time.

Inputs, as supplied by the backtest engine:
  * ``history``: every hourly row before the issue time (10:00 on day D-1);
  * ``future``: the target day D, calendar and weather columns only;
  * ``gap``: the same columns for the rest of day D-1.

``weather`` chooses which weather the target hours use: ``"oracle"`` reads the observed
values (an upper bound), ``"forecast"`` reads the archived forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loadforecast.evaluation.backtest import (
    CALENDAR_COLUMNS,
    gap_information,
    split_information,
)

WEATHER_VARS = ["temp", "rad", "rh", "cloud", "wind"]
SMOOTHING = 0.6  # daily weight decay of the smoothed temperature (EDA, section 7)
DAY_TYPE_CODE = {"working": 0, "saturday": 1, "sunday_holiday": 2}
HISTORY_DAYS = 40


def weather_sources(weather: str) -> tuple[dict[str, str], dict[str, str]]:
    """Columns that hold each weather variable for (target hours, gap hours)."""
    if weather == "oracle":
        obs = {v: f"{v}_obs" for v in WEATHER_VARS}
        return obs, obs
    if weather == "forecast":
        # Target hours: forecast issued two days before the hour, so always before the issue
        # time. Gap hours (the rest of day D-1): the one-day-lead forecast is early enough.
        return {v: f"{v}_fc2" for v in WEATHER_VARS}, {v: f"{v}_fc1" for v in WEATHER_VARS}
    raise ValueError(f"unknown weather mode: {weather}")


def future_columns(weather: str) -> list[str]:
    """Columns the engine must expose for the target day and the gap."""
    target, gap = weather_sources(weather)
    return CALENDAR_COLUMNS + sorted(set(target.values()) | set(gap.values()))


def _same_hour(known: pd.Series, dates: pd.Series, hours: pd.Series, lag: int) -> np.ndarray:
    keys = pd.MultiIndex.from_arrays([dates - pd.Timedelta(days=lag), hours])
    return known.reindex(keys).to_numpy()


def make_features(
    history: pd.DataFrame, future: pd.DataFrame, gap: pd.DataFrame, weather: str = "oracle"
) -> pd.DataFrame:
    target_src, gap_src = weather_sources(weather)
    recent = history.iloc[-24 * HISTORY_DAYS :]
    by_hour = recent.groupby(["local_date", "local_hour"])
    demand = by_hour["demand_mw"].mean()
    dates, hours = future["local_date"], future["local_hour"]
    target_day = dates.iloc[0]
    f = pd.DataFrame(index=future.index)

    # --- calendar of the target day ---
    f["hour"] = hours
    f["weekday"] = future["weekday"]
    f["day_type"] = future["day_type"].map(DAY_TYPE_CODE)
    angle = 2 * np.pi * dates.dt.day_of_year / 365.25
    for k in (1, 2):
        f[f"doy_sin{k}"] = np.sin(k * angle)
        f[f"doy_cos{k}"] = np.cos(k * angle)
    f["holiday"] = future["holiday_intensity"]
    f["holiday_national"] = future["holiday_national"].astype(int)
    f["bridge"] = future["bridge_intensity"]
    for flag in ("christmas", "august", "holy_week"):
        f[flag] = future[flag].astype(int)
    f["utc_offset"] = future["utc_offset_h"]

    # --- demand lags at the same local hour, and the calendar of the lagged days ---
    for lag in (1, 2, 7, 14, 21, 28):
        f[f"lag_d{lag}"] = _same_hour(demand, dates, hours, lag)  # lag 1 exists before 10:00 only
    f["lag_latest"] = f["lag_d1"].fillna(f["lag_d2"])
    f["ref_4w"] = f[["lag_d7", "lag_d14", "lag_d21", "lag_d28"]].mean(axis=1)
    f["lag_d2_vs_week_before"] = f["lag_d2"] - _same_hour(demand, dates, hours, 9)
    holiday_by_day = pd.concat([recent, gap]).groupby("local_date")["holiday_intensity"].first()
    for lag in (1, 2, 7, 14):
        f[f"holiday_d{lag}"] = holiday_by_day.reindex(dates - pd.Timedelta(days=lag)).to_numpy()

    # --- references that stay clean around holidays ---
    # ``ref_clean`` averages only the lagged weeks that were not holidays, and the three
    # ``lag_last_*`` features give the same hour on the most recent fully observed Sunday or
    # holiday, Saturday and ordinary working day, so a holiday can borrow a Sunday's shape.
    lag_cols = [f"lag_d{lag}" for lag in (7, 14, 21, 28)]
    lag_holiday = np.column_stack(
        [
            holiday_by_day.reindex(dates - pd.Timedelta(days=lag)).to_numpy()
            for lag in (7, 14, 21, 28)
        ]
    )
    clean = f[lag_cols].where(lag_holiday < 0.05)
    f["ref_clean"] = clean.mean(axis=1).fillna(f["ref_4w"])
    days = recent.groupby("local_date").agg(
        day_type=("day_type", "first"),
        holiday=("holiday_intensity", "first"),
        bridge=("bridge_intensity", "first"),
        hours=("demand_mw", "count"),
    )
    complete = days[(days.index < target_day - pd.Timedelta(days=1)) & (days["hours"] >= 23)]
    ordinary = (complete["holiday"] < 0.05) & (complete["bridge"] < 0.05)
    picks = {
        "sunday": complete.index[complete["day_type"] == "sunday_holiday"],
        "saturday": complete.index[(complete["day_type"] == "saturday") & ordinary],
        "working": complete.index[(complete["day_type"] == "working") & ordinary],
    }
    for name, candidates in picks.items():
        if len(candidates):
            keys = pd.MultiIndex.from_arrays([np.full(len(f), candidates[-1]), hours])
            f[f"lag_last_{name}"] = demand.reindex(keys).to_numpy()
        else:
            f[f"lag_last_{name}"] = np.nan

    # --- the morning of the issue day (hours observed before 10:00) against earlier days ---
    def morning_mean(column: str, lag: int) -> float:
        day = recent[(recent["local_date"] == target_day - pd.Timedelta(days=lag))]
        return day.loc[day["local_hour"] < 10, column].mean()

    f["morning_vs_d8"] = morning_mean("demand_mw", 1) - morning_mean("demand_mw", 8)
    f["morning_vs_d2"] = morning_mean("demand_mw", 1) - morning_mean("demand_mw", 2)
    f["morning_temp_vs_d8"] = morning_mean("temp_obs", 1) - morning_mean("temp_obs", 8)
    day_mean = recent.groupby("local_date")["demand_mw"].mean()
    f["day_d2_vs_d9"] = day_mean.get(target_day - pd.Timedelta(days=2), np.nan) - day_mean.get(
        target_day - pd.Timedelta(days=9), np.nan
    )

    # --- recent level: the last 24 observed hours against the same hours a week earlier ---
    last24 = recent["demand_mw"].iloc[-24:]
    week_before = recent["demand_mw"].iloc[-24 - 168 : -168]
    f["last24_mean"] = last24.mean()
    f["last24_vs_week_before"] = last24.mean() - week_before.mean()
    f["last_observed"] = recent["demand_mw"].iloc[-1]
    f["last_observed_vs_week_before"] = (
        recent["demand_mw"].iloc[-1] - recent["demand_mw"].iloc[-1 - 168]
    )

    # --- weather at the target hours ---
    for var in WEATHER_VARS:
        f[var] = future[target_src[var]].to_numpy()
    f["temp_day_mean"] = f["temp"].mean()
    f["temp_day_max"] = f["temp"].max()
    f["temp_day_min"] = f["temp"].min()
    f["rad_day_mean"] = f["rad"].mean()

    # --- weather relative to the lagged days (so the lags can be corrected) ---
    temp_obs, rad_obs = by_hour["temp_obs"].mean(), by_hour["rad_obs"].mean()
    for lag in (2, 7):
        f[f"temp_vs_d{lag}"] = f["temp"] - _same_hour(temp_obs, dates, hours, lag)
        f[f"rad_vs_d{lag}"] = f["rad"] - _same_hour(rad_obs, dates, hours, lag)

    # --- thermal inertia: exponentially smoothed daily temperature ---
    daily_temp = recent.groupby("local_date")["temp_obs"].mean()
    issue_day = target_day - pd.Timedelta(days=1)
    morning = recent.loc[recent["local_date"] == issue_day, "temp_obs"]
    issue_day_temp = np.concatenate([morning.to_numpy(), gap[gap_src["temp"]].to_numpy()]).mean()
    past = daily_temp.reindex([target_day - pd.Timedelta(days=k) for k in range(2, 11)]).to_numpy()
    series = np.concatenate([[f["temp_day_mean"].iloc[0], issue_day_temp], past])
    weights = (1 - SMOOTHING) * SMOOTHING ** np.arange(len(series))
    ok = ~np.isnan(series)
    f["temp_smooth"] = (series[ok] * weights[ok]).sum() / weights[ok].sum()
    f["temp_issue_day"] = issue_day_temp
    f["temp_day_mean_vs_d2"] = f["temp_day_mean"] - past[0]
    return f.astype("float64")


def build_dataset(
    table: pd.DataFrame,
    start: str,
    end: str,
    *,
    issue_time_local: str,
    tz: str,
    weather: str = "oracle",
) -> pd.DataFrame:
    """Features and target for every target day in [start, end], through the engine's split.

    Returns one row per target hour with ``local_date``, the features and ``y`` (NaN on
    masked days). ``table`` should come from ``prepare_table``.
    """
    columns = future_columns(weather)
    parts = []
    for target_date in pd.date_range(start, end, freq="D"):
        history, future = split_information(table, target_date, issue_time_local, tz, columns)
        if future.empty:
            continue
        gap = gap_information(table, target_date, issue_time_local, tz, columns)
        feats = make_features(history, future, gap, weather)
        feats.insert(0, "local_date", future["local_date"])
        parts.append(feats)
    out = pd.concat(parts)
    out["y"] = table["demand_mw"].reindex(out.index)
    return out
