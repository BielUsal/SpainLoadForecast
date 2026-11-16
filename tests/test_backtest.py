"""Backtest engine, baselines and residual quantiles, on a synthetic table (no data needed)."""

import numpy as np
import pandas as pd
import pytest

from loadforecast.evaluation.backtest import (
    CALENDAR_COLUMNS,
    issue_time_utc,
    prepare_table,
    run_backtest,
    split_information,
)
from loadforecast.models.baselines import (
    Climatology,
    ReeForecast,
    SeasonalNaiveDaily,
    SeasonalNaiveWeekly,
)
from loadforecast.models.quantiles import add_residual_quantiles

TZ = "Europe/Madrid"
ISSUE = "10:00"
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]


def make_table(start="2023-01-01", end="2023-12-31", seed=0) -> pd.DataFrame:
    local_start = pd.Timestamp(start, tz=TZ)
    local_end = pd.Timestamp(end, tz=TZ) + pd.Timedelta(days=1)
    idx = pd.date_range(
        local_start.tz_convert("UTC"), local_end.tz_convert("UTC"), freq="1h", inclusive="left"
    ).rename("time_utc")
    local = idx.tz_convert(TZ)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=idx)
    df["local_date"] = local.tz_localize(None).normalize()
    df["local_hour"] = local.hour
    df["utc_offset_h"] = 1.0
    df["weekday"] = local.day_of_week
    df["day_type"] = np.select(
        [df["weekday"] == 5, df["weekday"] == 6], ["saturday", "sunday_holiday"], "working"
    )
    for col in ("holiday_national", "christmas", "august", "holy_week"):
        df[col] = False
    df["holiday_intensity"] = 0.0
    df["bridge_intensity"] = 0.0
    df["local_hours"] = df.groupby("local_date")["local_hour"].transform("size")
    shape = 5000 * np.sin(2 * np.pi * (df["local_hour"] - 6) / 24) - 3000 * (df["weekday"] >= 5)
    df["demand_mw"] = 28000 + shape + rng.normal(0, 300, len(df))
    df["ree_forecast_mw"] = df["demand_mw"] + rng.normal(0, 200, len(df))
    df["temp_obs"] = 15 + rng.normal(0, 3, len(df))
    df["demand_spike"] = False
    df["is_masked"] = False
    return df


@pytest.fixture(scope="module")
def table():
    return prepare_table(make_table())


def models(table):
    return [
        SeasonalNaiveWeekly(),
        SeasonalNaiveDaily(),
        Climatology(),
        Climatology(level_adjust=True),
        ReeForecast(table),
    ]


def test_issue_time_follows_local_clock():
    assert issue_time_utc("2023-01-10", ISSUE, TZ) == pd.Timestamp("2023-01-09 09:00", tz="UTC")
    assert issue_time_utc("2023-07-10", ISSUE, TZ) == pd.Timestamp("2023-07-09 08:00", tz="UTC")


def test_split_information(table):
    history, future = split_information(table, "2023-06-15", ISSUE, TZ)
    issue = issue_time_utc("2023-06-15", ISSUE, TZ)
    assert history.index.max() == issue - pd.Timedelta(hours=1)
    assert list(future.columns) == CALENDAR_COLUMNS
    assert len(future) == 24 and (future["local_date"] == "2023-06-15").all()
    assert "demand_spike" not in history.columns


def test_dst_days_have_23_and_25_targets(table):
    _, spring = split_information(table, "2023-03-26", ISSUE, TZ)
    _, autumn = split_information(table, "2023-10-29", ISSUE, TZ)
    assert len(spring) == 23 and len(autumn) == 25


def test_weekly_naive_is_the_value_one_week_earlier(table):
    history, future = split_information(table, "2023-06-15", ISSUE, TZ)
    pred = SeasonalNaiveWeekly().predict(history, future)
    expected = table.loc[table["local_date"] == "2023-06-08", "demand_mw"].to_numpy()
    np.testing.assert_allclose(pred["point"].to_numpy(), expected)


def test_daily_naive_uses_the_latest_observed_same_hour(table):
    history, future = split_information(table, "2023-06-15", ISSUE, TZ)
    pred = SeasonalNaiveDaily().predict(history, future)["point"].to_numpy()
    day_before = table.loc[table["local_date"] == "2023-06-14", "demand_mw"].to_numpy()
    two_before = table.loc[table["local_date"] == "2023-06-13", "demand_mw"].to_numpy()
    np.testing.assert_allclose(pred[:10], day_before[:10])  # hours 0-9 were observed at 10:00
    np.testing.assert_allclose(pred[10:], two_before[10:])


def test_naive_falls_back_when_the_lag_day_is_masked():
    raw = make_table()
    raw.loc[raw["local_date"] == "2023-06-08", "is_masked"] = True
    table = prepare_table(raw)
    history, future = split_information(table, "2023-06-15", ISSUE, TZ)
    pred = SeasonalNaiveWeekly().predict(history, future)
    expected = table.loc[table["local_date"] == "2023-06-01", "demand_mw"].to_numpy()
    np.testing.assert_allclose(pred["point"].to_numpy(), expected)


def test_climatology_recovers_the_profile(table):
    history, future = split_information(table, "2023-11-15", ISSUE, TZ)
    for model in (Climatology(), Climatology(level_adjust=True)):
        model.fit(history)
        pred = model.predict(history, future)["point"]
        actual = table.loc[future.index, "demand_mw"]
        assert (pred - actual).abs().mean() < 400  # noise sd is 300


def test_no_leakage_from_after_the_issue_time(table):
    """Corrupt everything a model must not see; the forecasts must not change."""
    target = "2023-09-20"
    issue = issue_time_utc(target, ISSUE, TZ)
    corrupted = table.copy()
    secret = [
        c
        for c in table.columns
        if c not in CALENDAR_COLUMNS and c != "ree_forecast_mw" and table[c].dtype.kind == "f"
    ]
    after = corrupted.index >= issue
    corrupted.loc[after, secret] = -999.0
    assert (corrupted.loc[after, "demand_mw"] == -999.0).all()
    for clean_model, dirty_model in zip(models(table), models(corrupted), strict=True):
        kwargs = dict(issue_time_local=ISSUE, tz=TZ)
        clean = run_backtest(clean_model, table, target, target, **kwargs)
        dirty = run_backtest(dirty_model, corrupted, target, target, **kwargs)
        pd.testing.assert_frame_equal(clean, dirty)
        assert clean["point"].notna().all(), clean_model.name


def test_run_backtest_shape_and_refit_cadence(table):
    calls = []

    class Counting(SeasonalNaiveWeekly):
        def fit(self, history):
            calls.append(history.index.max())

    pred = run_backtest(
        Counting(), table, "2023-03-20", "2023-04-02", issue_time_local=ISSUE, tz=TZ, refit_every=7
    )
    days = pd.date_range("2023-03-20", "2023-04-02")
    anchored = [d for d in days[1:] if d.toordinal() % 7 == 0]
    assert len(calls) == 1 + len(anchored) == 3
    assert len(pred) == 14 * 24 - 1  # one 23-hour day
    assert pred.index.is_unique and pred["point"].notna().all()


def test_residual_quantiles_are_ordered_calibrated_and_causal(table):
    pred = run_backtest(
        SeasonalNaiveWeekly(), table, "2023-02-01", "2023-12-31", issue_time_local=ISSUE, tz=TZ
    )
    actual = table["demand_mw"]
    out = add_residual_quantiles(pred, actual, TAUS, window_days=120, min_days=28)
    cols = [f"q{t:g}" for t in TAUS]
    ready = out.dropna(subset=cols)
    assert out.loc[out["local_date"] < "2023-02-28", cols].isna().all().all()
    assert (np.diff(ready[cols].to_numpy(), axis=1) >= 0).all()
    y = actual.reindex(ready.index)
    coverage = ((y >= ready["q0.05"]) & (y <= ready["q0.95"])).mean()
    assert 0.85 < coverage < 0.95
    # Causality: changing actuals from day D-1 onwards must not change the quantiles of day D.
    day = pd.Timestamp("2023-08-10")
    tampered = actual.copy()
    tampered[table["local_date"] >= day - pd.Timedelta(days=1)] += 5000.0
    out2 = add_residual_quantiles(pred, tampered, TAUS, window_days=120, min_days=28)
    same_day = out["local_date"] == day
    pd.testing.assert_frame_equal(out.loc[same_day, cols], out2.loc[same_day, cols])
