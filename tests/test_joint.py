import numpy as np
import pandas as pd
import pytest

pytest.importorskip("zuko")

from loadforecast.models.joint import (  # noqa: E402
    Days,
    GaussianCopula,
    Independent,
    Marginals,
    build_days,
    prefer_forecast,
)
from loadforecast.models.quantiles import quantile_column  # noqa: E402

TAUS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
Z = np.array([-1.645, -1.2816, -0.6745, 0.0, 0.6745, 1.2816, 1.645])
CFG = {
    "copula": {"window_days": 365},
    "flow": {
        "target": "residual",
        "transforms": 2,
        "hidden": [16, 16],
        "bins": 4,
        "lr": 0.003,
        "weight_decay": 0.0,
        "batch_size": 64,
        "max_epochs": 30,
        "patience": 5,
        "holdout": 0.2,
        "seeds": 1,
    },
}


def synthetic_days(n: int, corr: np.ndarray, seed: int = 0) -> Days:
    """Days whose outcomes are point + 100 * correlated standard normals, with the matching
    normal quantiles, so the marginals are exactly right."""
    rng = np.random.default_rng(seed)
    point = 25000 + 3000 * np.sin(np.linspace(0, 2 * np.pi, 24))[None].repeat(n, 0)
    eps = rng.standard_normal((n, 24)) @ np.linalg.cholesky(corr).T
    q = point[..., None] + 100.0 * Z
    return Days(
        dates=pd.date_range("2023-01-01", periods=n, freq="D"),
        y=point + 100.0 * eps,
        point=point,
        q=q,
        width=np.full((n, 24), 100.0 * 2 * 1.645),
        ctx=rng.standard_normal((n, 4)).astype(np.float32),
        source=np.ones(n),
    )


def ar1(rho: float) -> np.ndarray:
    i = np.arange(24)
    return rho ** np.abs(i[:, None] - i[None, :])


def test_marginals_round_trip_and_match_the_quantiles():
    m = Marginals(TAUS)
    q = np.array([[1.0, 2.0, 4.0, 5.0, 7.0, 9.0, 10.0]])
    s = np.linspace(-4, 4, 41)[None]
    values = m.values(q, s)[0]
    assert np.allclose(m.scores(np.repeat(q, len(values), 0), values), s[0])
    assert np.allclose(m.values(q, m.knots[None]), q)  # the knots give back the quantiles
    assert (np.diff(m.values(q, s)[0]) > 0).all()  # increasing, also beyond the end knots


def test_independent_marginals_are_calibrated_and_hours_uncorrelated():
    days = synthetic_days(1, np.eye(24))
    x = Independent(TAUS, CFG).sample(days, 20000, np.random.default_rng(0))
    assert x.shape == (20000, 24)
    assert np.allclose(np.quantile(x, TAUS, axis=0).T, days.q[0], atol=8.0)
    assert abs(np.corrcoef(x[:, 10], x[:, 11])[0, 1]) < 0.03


def test_copula_learns_the_correlation_between_hours():
    days = synthetic_days(400, ar1(0.9))
    model = GaussianCopula(TAUS, CFG)
    model.fit(days)
    assert np.abs(model.corr - ar1(0.9)).mean() < 0.05
    assert np.allclose(np.diag(model.corr, 1), 0.9, atol=0.03)
    x = model.sample(days.take([0]), 20000, np.random.default_rng(1))
    assert np.corrcoef(x[:, 10], x[:, 11])[0, 1] == pytest.approx(0.9, abs=0.03)
    assert np.allclose(np.quantile(x, TAUS, axis=0).T, days.q[0], atol=8.0)


def test_copula_beats_independent_on_the_energy_score_when_hours_move_together():
    from loadforecast.evaluation.metrics import energy_score

    train, test = synthetic_days(400, ar1(0.95)), synthetic_days(60, ar1(0.95), seed=5)
    copula, independent = GaussianCopula(TAUS, CFG), Independent(TAUS, CFG)
    copula.fit(train)
    rng = np.random.default_rng(2)
    es = {
        name: np.mean(
            [energy_score(test.y[i], m.sample(test.take([i]), 500, rng)) for i in range(60)]
        )
        for name, m in [("copula", copula), ("independent", independent)]
    }
    assert es["copula"] < es["independent"]


def test_flow_trains_and_samples_the_right_shape_and_scale():
    from loadforecast.models.joint import ConditionalFlow

    days = synthetic_days(300, ar1(0.9))
    flow = ConditionalFlow(TAUS, CFG, device="cpu")
    flow.fit(days)
    x = flow.sample(days.take([0]), 2000, np.random.default_rng(0))
    assert x.shape == (2000, 24) and np.isfinite(x).all()
    assert np.abs(np.median(x, axis=0) - days.point[0]).max() < 100.0
    assert 40.0 < x.std(axis=0).mean() < 250.0  # the truth is 100


def test_build_days_drops_clock_change_days_and_masks_unknown_demand():
    idx = pd.date_range("2025-03-28", "2025-04-01", freq="h", tz="Europe/Madrid", inclusive="left")
    idx = idx.tz_convert("UTC")
    local = idx.tz_convert("Europe/Madrid")
    table = pd.DataFrame(
        {
            "local_date": local.normalize().tz_localize(None),
            "local_hour": local.hour,
            "demand_mw": 25000.0,
            "is_masked": False,
            "weekday": local.day_of_week,
            "temp_fc2": 12.0,
            "temp_obs": 12.0,
            **{f: 0.0 for f in ["holiday_intensity", "bridge_intensity", "christmas"]},
            **{f: 0.0 for f in ["august", "holy_week"]},
        },
        index=idx,
    )
    table.loc[table["local_date"] == "2025-03-31", "is_masked"] = True
    pred = table[["local_date", "local_hour"]].copy()
    pred["point"] = 25000.0
    for tau, z in zip(TAUS, Z, strict=True):
        pred[quantile_column(tau)] = 25000.0 + 100.0 * z
    days = build_days(pred, table, TAUS, "forecast")
    assert pd.Timestamp("2025-03-30") not in days.dates  # 23 hours
    assert len(days) == 3
    assert days.complete.tolist() == [True, True, False]
    assert days.ctx.shape == (3, 7 + 5 + 2 + 3)

    oracle = build_days(pred, table, TAUS, "oracle")
    both = prefer_forecast(days.take([0]), oracle)
    assert len(both) == 3 and both.source.tolist() == [1.0, 0.0, 0.0]


def test_flow_inflation_widens_an_overconfident_flow():
    from loadforecast.models.joint import ConditionalFlow

    days = synthetic_days(300, ar1(0.9))
    days.width = days.width / 2  # the base model's raw ranges are half as wide as they should be
    cfg = {**CFG, "flow": {**CFG["flow"], "inflate": True, "max_epochs": 2, "patience": 1}}
    flow = ConditionalFlow(TAUS, cfg, device="cpu")
    flow.fit(days)  # two epochs: the flow is still far too narrow, so it must be widened
    assert flow.inflation > 1.2
    test = synthetic_days(80, ar1(0.9), seed=9)
    test.width = test.width / 2
    rng = np.random.default_rng(0)
    x = np.stack([flow.sample(test.take([i]), 400, rng) for i in range(80)])
    lo, hi = np.quantile(x, [0.05, 0.95], axis=1)
    assert 0.8 < ((test.y >= lo) & (test.y <= hi)).mean() < 0.97
