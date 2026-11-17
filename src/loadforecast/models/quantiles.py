"""Turn point forecasts into quantile forecasts from the forecast's own past errors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_column(tau: float) -> str:
    return f"q{tau:g}"


def add_residual_quantiles(
    pred: pd.DataFrame,
    actual: pd.Series,
    taus: list[float],
    *,
    window_days: int = 365,
    min_days: int = 28,
) -> pd.DataFrame:
    """Add quantile columns: point forecast plus empirical quantiles of past errors by hour.

    For target day D (issued on D-1) only errors from target days D-2 and earlier are used,
    because day D-1 is not fully observed at issue time. The window is the last
    ``window_days`` such days; days with fewer than ``min_days`` of error history get NaN
    quantiles. ``pred`` needs ``local_date``, ``local_hour`` and ``point``; ``actual`` is
    aligned on the same index and may contain NaN (masked days are ignored).
    """
    frame = pred[["local_date", "local_hour", "point"]].copy()
    frame["error"] = actual.reindex(frame.index) - frame["point"]
    errors = frame.pivot_table(
        index="local_date", columns="local_hour", values="error", aggfunc="mean", dropna=False
    ).reindex(columns=range(24))
    dates = errors.index
    values = errors.to_numpy()
    offsets = np.full((len(dates), 24, len(taus)), np.nan)
    for i, date in enumerate(dates):
        hi = dates.searchsorted(date - pd.Timedelta(days=1))  # rows strictly before D-1
        lo = dates.searchsorted(date - pd.Timedelta(days=1 + window_days))
        window = values[lo:hi]
        enough = (~np.isnan(window)).sum(axis=0) >= min_days
        if enough.any():
            with np.errstate(all="ignore"):
                q = np.nanquantile(window[:, enough], taus, axis=0)  # (taus, hours)
            offsets[i][enough] = q.T
    row = dates.get_indexer(frame["local_date"])
    hour = frame["local_hour"].to_numpy()
    out = pred.copy()
    for j, tau in enumerate(taus):
        out[quantile_column(tau)] = frame["point"].to_numpy() + offsets[row, hour, j]
    return out


def recalibrate_quantiles(
    pred: pd.DataFrame,
    actual: pd.Series,
    taus: list[float],
    *,
    window_days: int = 365,
    min_days: int = 28,
) -> pd.DataFrame:
    """Recalibrate a model's own quantiles from its past standardised errors.

    The model keeps its point forecast and the width it assigns to each hour; only the
    multipliers change. With ``s`` the width of the model's widest central interval, the
    past errors are standardised as ``z = (actual - point) / s`` and the new quantile is
    ``point + s * quantile(z, tau)``, with the quantiles of ``z`` pooled over all hours of the
    window. So a model whose ranges are 20% too narrow gets them widened by 20% everywhere,
    and a constant bias is shifted out, but a day the model considers uncertain stays wider
    than a day it considers easy.

    Causality is as in ``add_residual_quantiles``: for target day D only target days D-2 and
    earlier are used. Days with fewer than ``min_days`` of history keep the model's quantiles.
    The model's original quantiles are kept in ``model_q*`` columns.
    """
    columns = [quantile_column(t) for t in taus]
    out = pred.copy()
    for column in columns:
        out[f"model_{column}"] = pred[column]
    scale = (pred[columns[-1]] - pred[columns[0]]).to_numpy()
    frame = pred[["local_date", "local_hour"]].copy()
    frame["z"] = (actual.reindex(frame.index) - pred["point"]).to_numpy() / scale
    z = frame.pivot_table(
        index="local_date", columns="local_hour", values="z", aggfunc="mean", dropna=False
    ).reindex(columns=range(24))
    dates = z.index
    values = z.to_numpy()
    multipliers = np.full((len(dates), len(taus)), np.nan)
    for i, date in enumerate(dates):
        hi = dates.searchsorted(date - pd.Timedelta(days=1))  # rows strictly before D-1
        lo = dates.searchsorted(date - pd.Timedelta(days=1 + window_days))
        window = values[lo:hi]
        if (~np.isnan(window)).any(axis=1).sum() >= min_days:
            multipliers[i] = np.nanquantile(window, taus)
    row = dates.get_indexer(frame["local_date"])
    for j, column in enumerate(columns):
        new = pred["point"].to_numpy() + scale * multipliers[row, j]
        out[column] = np.where(np.isnan(new), pred[column].to_numpy(), new)
    return out


def point_from_median(pred: pd.DataFrame) -> pd.DataFrame:
    """Use the (recalibrated) median as the point forecast; the old point goes to ``model_point``.

    Run after ``recalibrate_quantiles``: the recalibrated median is the model's point shifted by
    the median of its own past standardised errors, so this is a causal bias correction.
    Hours without a median keep their point.
    """
    out = pred.copy()
    out["model_point"] = pred["point"]
    median = pred[quantile_column(0.5)]
    out["point"] = median.where(median.notna(), pred["point"])
    return out
