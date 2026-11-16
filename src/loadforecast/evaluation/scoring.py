"""Score a prediction frame against realised demand."""

from __future__ import annotations

import numpy as np
import pandas as pd

from loadforecast.evaluation import metrics
from loadforecast.models.quantiles import quantile_column


def join_actual(pred: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Add ``y`` and drop hours that cannot be scored (masked days, missing forecast or actual)."""
    out = pred.join(table[["demand_mw", "is_masked"]])
    out = out[~out["is_masked"]].rename(columns={"demand_mw": "y"}).drop(columns="is_masked")
    return out.dropna(subset=["y", "point"])


def summary(scored: pd.DataFrame, taus: list[float]) -> dict:
    """All headline metrics for a scored frame (output of ``join_actual``)."""
    y, point = scored["y"].to_numpy(), scored["point"].to_numpy()
    out = {
        "hours": len(scored),
        "MAE": metrics.mae(y, point),
        "RMSE": metrics.rmse(y, point),
        "MAPE": metrics.mape(y, point),
        "bias": metrics.bias(y, point),
    }
    columns = [quantile_column(t) for t in taus]
    if all(c in scored.columns for c in columns):
        prob = scored.dropna(subset=columns)
        yq = prob["y"].to_numpy()
        out["CRPS"] = float(metrics.crps_from_quantiles(yq, prob[columns].to_numpy(), taus).mean())
        for nominal, lo, hi in metrics.central_intervals(taus):
            lower, upper = prob[quantile_column(lo)], prob[quantile_column(hi)]
            out[f"cov{round(100 * nominal)}"] = 100 * metrics.interval_coverage(yq, lower, upper)
            out[f"width{round(100 * nominal)}"] = metrics.interval_width(lower, upper)
    return out


def hourly_losses(scored: pd.DataFrame, taus: list[float]) -> pd.DataFrame:
    """Per-hour absolute error and CRPS (NaN where quantiles are missing)."""
    out = scored[["local_date", "local_hour"]].copy()
    out["abs_error"] = (scored["point"] - scored["y"]).abs()
    columns = [quantile_column(t) for t in taus]
    if all(c in scored.columns for c in columns):
        out["crps"] = metrics.crps_from_quantiles(
            scored["y"].to_numpy(), scored[columns].to_numpy(), taus
        )
        out.loc[scored[columns].isna().any(axis=1), "crps"] = np.nan
    return out


def daily_losses(scored: pd.DataFrame, taus: list[float]) -> pd.DataFrame:
    """Mean loss per target day: the unit of the Diebold-Mariano test."""
    return hourly_losses(scored, taus).groupby("local_date")[["abs_error", "crps"]].mean()
