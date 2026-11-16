"""Baseline point forecasts. Every result in the project is reported against these.

All three follow the ``Model`` protocol of ``loadforecast.evaluation.backtest``. They are
turned into probabilistic forecasts by ``loadforecast.models.quantiles.add_residual_quantiles``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY_TYPES = ["working", "saturday", "sunday_holiday"]


def _lookup(history: pd.DataFrame, future: pd.DataFrame, day_lags: list[int]) -> np.ndarray:
    """Demand at the same local hour ``lag`` days before each future row.

    Lags are tried in order and the first available value wins. A value is unavailable when
    the hour had not started at issue time, when the day is masked, or when the local hour
    does not exist on that day (clock change).
    """
    recent = history.iloc[-24 * (max(day_lags) + 3) :]
    known = recent.groupby(["local_date", "local_hour"])["demand_mw"].mean()
    out = np.full(len(future), np.nan)
    for lag in day_lags:
        keys = pd.MultiIndex.from_arrays(
            [future["local_date"] - pd.Timedelta(days=lag), future["local_hour"]]
        )
        values = known.reindex(keys).to_numpy()
        out = np.where(np.isnan(out), values, out)
    return out


class SeasonalNaiveWeekly:
    """Same local hour one week earlier (two or three weeks if that value is unavailable)."""

    name = "naive_weekly"

    def fit(self, history: pd.DataFrame) -> None:
        pass

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"point": _lookup(history, future, [7, 14, 21, 28])}, index=future.index
        )


class SeasonalNaiveDaily:
    """Most recent observed value of the same local hour.

    For a target day D issued at 10:00 on D-1, that is D-1 for hours before 10:00 and D-2
    for the rest.
    """

    name = "naive_daily"

    def fit(self, history: pd.DataFrame) -> None:
        pass

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"point": _lookup(history, future, [1, 2, 3, 4])}, index=future.index)


class Climatology:
    """Mean demand by (day type, week of year, local hour) over all history.

    Weeks are smoothed with their two neighbours. With ``level_adjust`` the profile is
    rescaled by the ratio of actual to climatological demand over the last ``adjust_days``
    fully observed days, which follows slow changes in the level.
    """

    def __init__(self, level_adjust: bool = False, adjust_days: int = 28):
        self.level_adjust = level_adjust
        self.adjust_days = adjust_days
        self.name = "climatology_adjusted" if level_adjust else "climatology"
        self._table: np.ndarray | None = None

    @staticmethod
    def _keys(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        day_type = pd.Categorical(frame["day_type"], categories=DAY_TYPES).codes
        week = np.minimum((frame["local_date"].dt.day_of_year.to_numpy() - 1) // 7, 51)
        return day_type, week, frame["local_hour"].to_numpy()

    def fit(self, history: pd.DataFrame) -> None:
        data = history.dropna(subset=["demand_mw"])
        day_type, week, hour = self._keys(data)
        sums = np.zeros((len(DAY_TYPES), 52, 24))
        counts = np.zeros_like(sums)
        np.add.at(sums, (day_type, week, hour), data["demand_mw"].to_numpy())
        np.add.at(counts, (day_type, week, hour), 1.0)
        # smooth over neighbouring weeks (the year wraps around)
        sums = sums + np.roll(sums, 1, axis=1) + np.roll(sums, -1, axis=1)
        counts = counts + np.roll(counts, 1, axis=1) + np.roll(counts, -1, axis=1)
        with np.errstate(invalid="ignore"):
            table = sums / counts
        # cells never observed (short history): fall back to the mean of that day type and hour
        fallback = np.nanmean(table, axis=1, keepdims=True)
        self._table = np.where(np.isnan(table), fallback, table)

    def _profile(self, frame: pd.DataFrame) -> np.ndarray:
        return self._table[self._keys(frame)]

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        point = self._profile(future)
        if self.level_adjust:
            last_day = history["local_date"].iloc[-1]  # the issue day, only partly observed
            window = history[
                (history["local_date"] < last_day)
                & (history["local_date"] >= last_day - pd.Timedelta(days=self.adjust_days))
            ].dropna(subset=["demand_mw"])
            if len(window):
                point = point * window["demand_mw"].sum() / self._profile(window).sum()
        return pd.DataFrame({"point": point}, index=future.index)


class ReeForecast:
    """REE's published day-ahead forecast (ESIOS 1775): the professional benchmark.

    It is published at about 10:50, after our issue time, so it is read directly from the
    table and is not something our models could use as an input.
    """

    name = "ree"

    def __init__(self, table: pd.DataFrame):
        self._forecast = table["ree_forecast_mw"]

    def fit(self, history: pd.DataFrame) -> None:
        pass

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"point": self._forecast.reindex(future.index)}, index=future.index)
