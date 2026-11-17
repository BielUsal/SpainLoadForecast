"""The sequence model on a small synthetic table (CPU, a few seconds)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from loadforecast.evaluation.backtest import prepare_table, run_backtest  # noqa: E402
from loadforecast.models import rnn  # noqa: E402

TZ = "Europe/Madrid"
TAUS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
CFG = {
    "train_start": "2023-01-01",
    "encoder_hours": 168,
    "decoder_hours": 40,
    "hidden": 16,
    "layers": 1,
    "dropout": 0.0,
    "epochs": 3,
    "batch_size": 256,
    "lr": 0.005,
    "weight_decay": 0.0,
    "clip_norm": 1.0,
    "seeds": 2,
}


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    start = pd.Timestamp("2023-01-01", tz=TZ).tz_convert("UTC")
    end = pd.Timestamp("2023-05-01", tz=TZ).tz_convert("UTC")
    idx = pd.date_range(start, end, freq="1h", inclusive="left").rename("time_utc")
    local = idx.tz_convert(TZ)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(index=idx)
    df["local_date"] = local.tz_localize(None).normalize()
    df["local_hour"] = local.hour
    df["utc_offset_h"] = 1.0
    df["weekday"] = local.day_of_week
    df["day_type"] = "working"
    df["local_hours"] = 24
    df["holiday_national"] = False
    for flag in ("holiday_intensity", "bridge_intensity"):
        df[flag] = 0.0
    for flag in ("christmas", "august", "holy_week"):
        df[flag] = False
    df["temp_obs"] = 12 + 5 * np.sin(2 * np.pi * local.hour / 24) + rng.normal(0, 1, len(idx))
    df["rad_obs"] = np.clip(500 * np.sin(np.pi * (local.hour - 6) / 12), 0, None)
    for var in ("rh", "cloud", "wind"):
        df[f"{var}_obs"] = 0.0
    weekend = np.where(local.day_of_week >= 5, -3000.0, 0.0)
    shape = 4000 * np.sin(2 * np.pi * (local.hour - 9) / 24)
    df["demand_mw"] = 27000 + shape + weekend + rng.normal(0, 200, len(idx))
    df["is_masked"] = df["local_date"] == pd.Timestamp("2023-04-10")
    return df


def backtest(table: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    torch.manual_seed(0)
    rnn.RNNForecaster._fitted = {}
    model = rnn.RNNForecaster("oracle", TAUS, CFG)
    model.device = "cpu"
    return run_backtest(
        model,
        prepare_table(table),
        start,
        end,
        issue_time_local="10:00",
        tz=TZ,
        refit_every=10**6,
        future_columns=model.future_columns,
    )


def test_valid_origins_skip_incomplete_windows():
    y = np.arange(700, dtype=float)
    y[250] = np.nan
    origins = rnn.valid_origins(y, 168, 40)
    assert origins.min() == 168 and origins.max() == 660
    assert not ((origins > 250) & (origins <= 250 + 168)).any()  # encoder would hold the gap
    assert 250 in origins  # a missing target hour is allowed: the loss skips it


def test_quantiles_cannot_cross():
    net = rnn.Seq2Quantiles(len(TAUS), 8, 1, 0.0)
    out = net(torch.randn(4, 168, rnn.N_COVARIATES + 1), torch.randn(4, 38, rnn.N_COVARIATES + 1))
    assert out.shape == (4, 38, len(TAUS))
    assert (out.diff(dim=-1) > 0).all()


def test_forecasts_beat_last_week_and_survive_a_masked_day(table):
    pred = backtest(table, "2023-04-08", "2023-04-20")  # the encoder crosses the masked day
    columns = [f"q{tau:g}" for tau in TAUS]
    assert pred[["point", *columns]].notna().all().all()
    assert (np.diff(pred[columns].to_numpy(), axis=1) > 0).all()
    y = table["demand_mw"].reindex(pred.index)
    keep = ~table["is_masked"].reindex(pred.index)
    mae = (pred["point"] - y)[keep].abs().mean()
    assert mae < 600  # noise is 200 MW; the daily swing is 8,000 MW


def test_forecasts_do_not_depend_on_data_after_the_issue_time(table):
    day = "2023-04-20"
    first = backtest(table, day, day)
    tampered = table.copy()
    issue = pd.Timestamp("2023-04-19 10:00", tz=TZ).tz_convert("UTC")
    tampered.loc[tampered.index >= issue, "demand_mw"] += 5000.0
    second = backtest(tampered, day, day)
    assert np.allclose(first["point"], second["point"])
