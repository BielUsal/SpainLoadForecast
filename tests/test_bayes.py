"""Bayesian structural model: grid, mean function, correction and the backtest wrapper."""

import numpy as np
import pandas as pd
import pytest

from test_backtest import ISSUE, TZ
from test_features import weather_table

pytest.importorskip("numpyro")

from loadforecast.evaluation.backtest import issue_time_utc, run_backtest  # noqa: E402
from loadforecast.models import bayes_numpyro as bm  # noqa: E402
from loadforecast.models.bayes import (  # noqa: E402
    BayesForecaster,
    training_frame,
    two_stage_fit,
)

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
CFG = {
    "train_start": "2023-01-01",
    "n_draws": 40,
    "draws_per_sample": 4,
    "svi": {"map_steps": 300, "refit_map_steps": 100, "structure_steps": 100, "error_steps": 100},
}


@pytest.fixture(scope="module")
def table():
    return weather_table()


@pytest.fixture(scope="module")
def grid(table):
    return bm.build_grid(training_frame(table, "2023-01-01", pd.Timestamp("2023-10-31")))


@pytest.fixture(scope="module")
def fit(grid):
    return two_stage_fit(grid, CFG, "svi")


def test_grid_shapes_and_clock_change_days(grid):
    n = grid.n_days
    assert n == 304
    assert grid.data["y"].shape == (n, 24) and grid.data["temp_lags"].shape == (n * 24, bm.N_LAGS)
    assert not np.isnan(grid.data["y"]).any()  # the missing spring hour is interpolated
    np.testing.assert_allclose(grid.data["blend"].sum(axis=1), 1.0)
    assert bm.default_weight(grid.data)[: bm.WARMUP_DAYS + bm.CARRY_DAYS].sum() == 0


def test_grid_rejects_gaps_in_the_days(table):
    frame = training_frame(table, "2023-01-01", pd.Timestamp("2023-03-31"))
    with pytest.raises(ValueError):
        bm.build_grid(frame[frame["local_date"] != "2023-02-10"])


def test_correction_only_uses_residuals_known_at_issue_time():
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    params = {"carry": jnp.asarray(rng.normal(size=(bm.N_CARRY, 5)))}
    resid = rng.normal(size=(30, 24))
    base = np.asarray(bm.correction(params, jnp.asarray(resid)))
    assert np.all(base[: bm.CARRY_DAYS] == 0)
    changed = resid.copy()
    changed[-1] += 5.0  # the target day itself
    changed[-2, bm.ISSUE_HOUR :] += 5.0  # the issue day from 10:00 onwards
    np.testing.assert_allclose(
        np.asarray(bm.correction(params, jnp.asarray(changed)))[-1], base[-1]
    )
    changed[-2, 3] += 1.0  # an hour of the issue day before 10:00 is allowed to matter
    assert not np.allclose(np.asarray(bm.correction(params, jnp.asarray(changed)))[-1], base[-1])


def test_error_covariance_is_valid():
    import jax.numpy as jnp

    params = {"log_scale": jnp.zeros(7), "common": 0.3, "rho": 0.9}
    tril = np.asarray(bm.error_scale_tril(params))
    cov = tril @ tril.T
    np.testing.assert_allclose(np.diag(cov), 1.0, rtol=1e-4)
    assert cov[0, 1] == pytest.approx(0.3 + 0.7 * 0.9, rel=1e-4)
    assert cov[0, 23] == pytest.approx(0.3 + 0.7 * 0.9**23, rel=1e-3)


def test_structural_mean_fits_the_synthetic_profile(grid, fit):
    import jax.numpy as jnp

    data = {k: jnp.asarray(v) for k, v in grid.data.items()}
    mean = np.asarray(
        bm.mean_fn({k: jnp.asarray(v) for k, v in fit["map"]["structure"].items()}, data)
    )
    error_mw = 1000 * np.abs(mean - grid.data["y"])[bm.WARMUP_DAYS :].mean()
    assert error_mw < 400  # the synthetic noise has a standard deviation of 300 MW


def test_forecaster_through_the_engine_without_leakage(table, fit):
    kwargs = dict(issue_time_local=ISSUE, tz=TZ)

    def forecast(tbl, weather, target):
        model = BayesForecaster(weather, TAUS, CFG, "svi", fixed_fit=fit)
        return run_backtest(
            model, tbl, target, target, future_columns=model.future_columns, **kwargs
        )

    target = "2023-11-15"
    pred = forecast(table, "forecast", target)
    quantiles = pred[[f"q{t:g}" for t in TAUS]].to_numpy()
    assert len(pred) == 24 and pred["point"].notna().all()
    assert (np.diff(quantiles, axis=1) > 0).all()
    actual = table["demand_mw"].reindex(pred.index)
    assert (pred["point"] - actual).abs().mean() < 800

    corrupted = table.copy()
    after = corrupted.index >= issue_time_utc(target, ISSUE, TZ)
    secret = ["demand_mw", "temp_obs", "rad_obs", "rh_obs", "cloud_obs", "wind_obs"]
    corrupted.loc[after, secret] = -999.0
    pd.testing.assert_frame_equal(pred, forecast(corrupted, "forecast", target))
    # with observed weather as input, corrupting demand alone must not matter either
    corrupted = table.copy()
    corrupted.loc[after, "demand_mw"] = -999.0
    pd.testing.assert_frame_equal(
        forecast(table, "oracle", target), forecast(corrupted, "oracle", target)
    )
