"""Quality checks for hourly series. Each returns data, so tests and reports can both use them."""

from __future__ import annotations

import numpy as np
import pandas as pd


def missing_hours(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Timestamps absent from an hourly UTC index between its first and last value."""
    full = pd.date_range(index.min(), index.max(), freq="1h")
    return full.difference(index)


def gaps(series: pd.Series) -> pd.DataFrame:
    """Runs of consecutive NaN values: start, end and length in hours."""
    isna = series.isna()
    if not isna.any():
        return pd.DataFrame(columns=["start", "end", "hours"])
    run = (isna != isna.shift()).cumsum()[isna]
    out = pd.DataFrame(
        {
            "start": run.groupby(run).apply(lambda g: g.index[0]),
            "end": run.groupby(run).apply(lambda g: g.index[-1]),
            "hours": run.groupby(run).size(),
        }
    )
    return out.reset_index(drop=True)


def out_of_range(series: pd.Series, low: float, high: float) -> pd.Series:
    return series[(series < low) | (series > high)]


def spike_zscores(series: pd.Series, local_index: pd.DatetimeIndex) -> pd.Series:
    """Robust z-score of the hourly change, relative to the usual change at that hour of the week.

    The hourly difference is compared with the median difference for the same local
    (weekday, hour), and scaled by 1.4826 times the median absolute deviation.
    """
    diff = series.diff()
    key = local_index.day_of_week * 24 + local_index.hour
    med = diff.groupby(key).transform("median")
    mad = (diff - med).abs().groupby(key).transform("median")
    return (diff - med) / (1.4826 * mad.replace(0, np.nan))


def flag_spikes(
    series: pd.Series, local_index: pd.DatetimeIndex, threshold: float = 8.0
) -> pd.Series:
    return spike_zscores(series, local_index).abs() > threshold
