import numpy as np
import pytest

from loadforecast.evaluation import metrics
from loadforecast.evaluation.dm_test import block_bootstrap_ci, diebold_mariano, long_run_variance


def test_point_metrics():
    y, pred = np.array([100.0, 200.0, 400.0]), np.array([110.0, 180.0, 400.0])
    assert metrics.mae(y, pred) == pytest.approx(10.0)
    assert metrics.rmse(y, pred) == pytest.approx(np.sqrt(500 / 3))
    assert metrics.mape(y, pred) == pytest.approx(100 * (0.1 + 0.1 + 0) / 3)
    assert metrics.bias(y, pred) == pytest.approx(-10 / 3)
    assert metrics.skill_score(5.0, 10.0) == 0.5


def test_pinball_loss():
    assert metrics.pinball_loss(10.0, 8.0, 0.9) == pytest.approx(1.8)  # under-forecast
    assert metrics.pinball_loss(10.0, 12.0, 0.9) == pytest.approx(0.2)  # over-forecast
    assert metrics.pinball_loss(10.0, 10.0, 0.5) == 0.0


def test_crps_samples_matches_gaussian_closed_form():
    rng = np.random.default_rng(1)
    mu, sigma = np.array([0.0, 2.0, -1.0]), np.array([1.0, 0.5, 3.0])
    y = np.array([0.3, 3.0, -4.0])
    samples = rng.normal(mu[:, None], sigma[:, None], size=(3, 20000))
    np.testing.assert_allclose(
        metrics.crps_samples(y, samples), metrics.crps_gaussian(y, mu, sigma), rtol=0.03
    )


def test_crps_samples_equals_naive_double_sum():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(1, 50)), np.array([0.4])
    naive = np.abs(x[0] - y[0]).mean() - 0.5 * np.abs(x[0][:, None] - x[0][None, :]).mean()
    assert metrics.crps_samples(y, x)[0] == pytest.approx(naive)


def test_crps_from_dense_quantiles_approaches_gaussian():
    from statistics import NormalDist

    taus = np.linspace(0.005, 0.995, 199)
    quantiles = np.array([[NormalDist().inv_cdf(t) for t in taus]])
    approx = metrics.crps_from_quantiles(np.array([0.7]), quantiles, taus)[0]
    assert approx == pytest.approx(metrics.crps_gaussian(0.7, 0.0, 1.0), rel=0.03)


def test_crps_of_a_point_forecast_is_absolute_error():
    assert metrics.crps_samples(np.array([3.0]), np.full((1, 10), 5.0))[0] == pytest.approx(2.0)


def test_coverage_width_and_central_intervals():
    y = np.arange(10.0)
    assert metrics.interval_coverage(y, np.full(10, 2.0), np.full(10, 6.0)) == 0.5
    assert metrics.interval_width(np.zeros(4), np.full(4, 3.0)) == 3.0
    taus = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    assert metrics.central_intervals(taus) == [
        (0.5, 0.25, 0.75),
        (0.8, 0.1, 0.9),
        (0.9, 0.05, 0.95),
    ]


def test_pit_is_uniform_for_a_calibrated_forecast():
    from statistics import NormalDist

    rng = np.random.default_rng(3)
    taus = np.linspace(0.05, 0.95, 19)
    q = np.tile([NormalDist().inv_cdf(t) for t in taus], (5000, 1))
    pit = metrics.pit_from_quantiles(rng.normal(size=5000), q, taus)
    assert abs(np.mean(pit <= 0.5) - 0.5) < 0.03
    assert pit.min() >= 0.05 and pit.max() <= 0.95


def test_energy_score_reduces_to_crps_in_one_dimension():
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(400, 1))
    assert metrics.energy_score(np.array([0.5]), samples) == pytest.approx(
        metrics.crps_samples(np.array([0.5]), samples.T)[0]
    )


def test_energy_and_variogram_prefer_the_right_dependence():
    rng = np.random.default_rng(5)
    common = rng.normal(size=(2000, 1))
    correlated = common + 0.1 * rng.normal(size=(2000, 6))
    independent = rng.normal(size=(2000, 6)) * correlated.std()
    es_good = np.mean([metrics.energy_score(correlated[i], correlated[500:]) for i in range(100)])
    es_bad = np.mean([metrics.energy_score(correlated[i], independent[500:]) for i in range(100)])
    vs_good = np.mean(
        [metrics.variogram_score(correlated[i], correlated[500:]) for i in range(100)]
    )
    vs_bad = np.mean(
        [metrics.variogram_score(correlated[i], independent[500:]) for i in range(100)]
    )
    assert es_good < es_bad
    assert vs_good < vs_bad


def test_long_run_variance_of_white_noise_is_its_variance():
    x = np.random.default_rng(6).normal(size=20000)
    assert long_run_variance(x) == pytest.approx(1.0, rel=0.08)


def test_diebold_mariano():
    rng = np.random.default_rng(7)
    a = rng.normal(1.0, 1.0, 400)
    same = diebold_mariano(a, a)
    assert same["stat"] == 0.0 and same["p_value"] == 1.0
    better = diebold_mariano(a - 0.5, a + rng.normal(0, 0.2, 400))
    assert better["mean_diff"] < 0 and better["p_value"] < 1e-6
    noise = diebold_mariano(rng.normal(size=400), rng.normal(size=400))
    assert noise["p_value"] > 0.01
    low, high = block_bootstrap_ci(a - 0.5, a)
    assert low == pytest.approx(-0.5) and high == pytest.approx(-0.5)
