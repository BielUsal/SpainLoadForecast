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
