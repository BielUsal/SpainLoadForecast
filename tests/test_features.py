"""Feature builder and LightGBM model on a synthetic table (no data needed)."""

import numpy as np
import pandas as pd
import pytest

from loadforecast.evaluation.backtest import (
    gap_information,
    issue_time_utc,
    prepare_table,
    run_backtest,
    split_information,
)
from loadforecast.features.build import (
    WEATHER_VARS,
    build_dataset,
    future_columns,
    make_features,
)
from test_backtest import ISSUE, TZ, make_table

KW = dict(issue_time_local=ISSUE, tz=TZ)


def weather_table() -> pd.DataFrame:
    table = make_table()
    rng = np.random.default_rng(11)
    for var, level in zip(WEATHER_VARS, [15.0, 200.0, 60.0, 50.0, 10.0], strict=True):
        table[f"{var}_obs"] = level + rng.normal(0, 3, len(table))
        table[f"{var}_fc1"] = table[f"{var}_obs"] + rng.normal(0, 1, len(table))
        table[f"{var}_fc2"] = table[f"{var}_obs"] + rng.normal(0, 1, len(table))
    return prepare_table(table)


@pytest.fixture(scope="module")
def table():
    return weather_table()


def features(table, target, weather):
    columns = future_columns(weather)
    history, future = split_information(table, target, ISSUE, TZ, columns)
    gap = gap_information(table, target, ISSUE, TZ, columns)
    return make_features(history, future, gap, weather), history, future, gap


def test_gap_covers_the_rest_of_the_issue_day(table):
    _, _, future, gap = features(table, "2023-06-15", "forecast")
    assert len(gap) == 14 and (gap["local_date"] == "2023-06-14").all()
    assert gap["local_hour"].tolist() == list(range(10, 24))
    assert "demand_mw" not in gap.columns and "demand_mw" not in future.columns


def test_lag_features_point_at_the_right_days(table):
    f, _, _, _ = features(table, "2023-06-15", "oracle")
    demand = table["demand_mw"]
    day = lambda d: demand[table["local_date"] == d].to_numpy()  # noqa: E731
    np.testing.assert_allclose(f["lag_d7"], day("2023-06-08"))
    np.testing.assert_allclose(f["lag_d2"], day("2023-06-13"))
    np.testing.assert_allclose(f["lag_d1"].iloc[:10], day("2023-06-14")[:10])
    assert f["lag_d1"].iloc[10:].isna().all()  # not yet observed at 10:00
    assert f.drop(columns="lag_d1").notna().all().all()
    # 2023-06-15 is a Thursday: the latest complete Sunday is 11 June, Saturday 10 June.
    np.testing.assert_allclose(f["lag_last_sunday"], day("2023-06-11"))
    np.testing.assert_allclose(f["lag_last_saturday"], day("2023-06-10"))
    np.testing.assert_allclose(f["lag_last_working"], day("2023-06-13"))


def test_weather_mode_selects_the_source(table):
    oracle, _, future_o, _ = features(table, "2023-06-15", "oracle")
    forecast, _, future_f, _ = features(table, "2023-06-15", "forecast")
    np.testing.assert_allclose(oracle["temp"], future_o["temp_obs"])
    np.testing.assert_allclose(forecast["temp"], future_f["temp_fc2"])
    assert not any(c.endswith("_obs") for c in future_f.columns)
    assert list(oracle.columns) == list(forecast.columns)


@pytest.mark.parametrize("weather", ["oracle", "forecast"])
def test_features_do_not_use_anything_after_the_issue_time(table, weather):
    target = "2023-09-20"
    issue = issue_time_utc(target, ISSUE, TZ)
    allowed = set(future_columns(weather))
    corrupted = table.copy()
    secret = [c for c in table.columns if c not in allowed and table[c].dtype.kind == "f"]
    corrupted.loc[corrupted.index >= issue, secret] = -999.0
    assert "demand_mw" in secret
    assert ("temp_obs" in secret) == (weather == "forecast")
    clean, *_ = features(table, target, weather)
    dirty, *_ = features(corrupted, target, weather)
    pd.testing.assert_frame_equal(clean, dirty)


def test_build_dataset_matches_single_calls(table):
    data = build_dataset(table, "2023-06-10", "2023-06-16", weather="oracle", **KW)
    single, *_ = features(table, "2023-06-15", "oracle")
    rows = data[data["local_date"] == "2023-06-15"]
    pd.testing.assert_frame_equal(rows[single.columns], single)
    np.testing.assert_allclose(rows["y"], table.loc[rows.index, "demand_mw"])
    assert len(data) == 7 * 24


def test_lightgbm_forecaster_runs_and_beats_the_naive(table):
    pytest.importorskip("lightgbm")
    from loadforecast.models.baselines import SeasonalNaiveWeekly
    from loadforecast.models.gbm import LightGBMForecaster

    cfg = {
        "half_life_years": 2.0,
        "num_boost_round": 150,
        "params": {
            "objective": "regression",
            "learning_rate": 0.1,
            "num_leaves": 7,
            "verbose": -1,
            "seed": 0,
        },
    }
    dataset = build_dataset(table, "2023-02-15", "2023-12-31", weather="oracle", **KW)
    actual = table["demand_mw"]
    errors = {}
    models = [LightGBMForecaster(dataset, w, cfg) for w in ("oracle", "forecast")]
    for model in [*models, SeasonalNaiveWeekly()]:
        pred = run_backtest(
            model,
            table,
            "2023-11-01",
            "2023-11-21",
            refit_every=7,
            future_columns=getattr(model, "future_columns", None),
            **KW,
        )
        assert pred["point"].notna().all()
        errors[model.name] = (pred["point"] - actual.reindex(pred.index)).abs().mean()
    assert errors["lgbm_oracle"] < errors["naive_weekly"]
    assert errors["lgbm_forecast"] < errors["naive_weekly"]
    shap = models[0].shap_values(dataset.tail(24))
    np.testing.assert_allclose(
        shap.sum(axis=1), models[0].booster.predict(dataset.tail(24)[models[0].features])
    )
