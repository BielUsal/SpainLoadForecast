"""The RNN with a flow head on the synthetic table of test_rnn (CPU, a few seconds)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zuko")

import test_rnn  # noqa: E402
from loadforecast.evaluation.backtest import prepare_table, run_backtest  # noqa: E402
from loadforecast.models import rnn_flow  # noqa: E402
from test_rnn import CFG, TAUS, TZ  # noqa: E402

FLOW_CFG = {
    **CFG,
    "seeds": 2,
    "flow": {
        "context": 8,
        "transforms": 2,
        "hidden": [16, 16],
        "bins": 4,
        "weight": 1.0,
        "n_samples": 300,
    },
}


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return test_rnn.table.__wrapped__()


def backtest(table: pd.DataFrame, start: str, end: str):
    torch.manual_seed(0)
    rnn_flow.RNNFlowForecaster._fitted = {}
    model = rnn_flow.RNNFlowForecaster("oracle", TAUS, FLOW_CFG)
    model.device = "cpu"
    pred = run_backtest(
        model,
        prepare_table(table),
        start,
        end,
        issue_time_local="10:00",
        tz=TZ,
        refit_every=10**6,
        future_columns=model.future_columns,
    )
    return model, pred


def test_scenarios_have_the_right_shape_and_follow_the_forecast(table, tmp_path):
    model, pred = backtest(table, "2023-04-18", "2023-04-20")
    assert pred["point"].notna().all()
    assert sorted(model.scenarios) == list(pd.date_range("2023-04-18", "2023-04-20"))
    point, offsets = model.scenarios[pd.Timestamp("2023-04-20")]
    assert point.shape == (24,) and offsets.shape == (300, 24)
    day = pred[pred["local_date"] == "2023-04-20"].sort_values("local_hour")
    assert np.allclose(point, day["point"], atol=1.0)
    x = point + offsets.astype(float)
    assert np.isfinite(x).all()
    # Barely trained, so judged against the quantile head's own range, not the truth.
    width = (day["q0.95"] - day["q0.05"]).to_numpy()
    lo, mid, hi = np.quantile(offsets.astype(float), [0.05, 0.5, 0.95], axis=0)
    assert 0.5 < (hi - lo).mean() / width.mean() < 1.5
    assert (np.abs(mid) < 0.25 * width).all()
    model.save_scenarios(tmp_path)
    z = np.load(tmp_path / "rnn_flow_oracle.npz")
    assert z["offsets"].shape == (3, 300, 24) and z["point"].shape == (3, 24)


def test_inflation_factor_restores_coverage():
    from loadforecast.evaluation.run_joint import inflation_factors

    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=200)
    y = rng.standard_normal((200, 24)) * 2.0  # outcomes twice as spread as the scenarios
    med, lo, hi = np.zeros((200, 24)), np.full((200, 24), -1.645), np.full((200, 24), 1.645)
    f = inflation_factors(dates, y, med, lo, hi, window=365, min_days=28)
    assert (f[:29] == 1.0).all() and f[29] != 1.0  # 28 past days needed
    assert f[-1] == pytest.approx(2.0, rel=0.1)
