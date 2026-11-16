"""Rolling-origin backtest engine.

For every target day D the forecast is issued on day D-1 at the issue time (10:00 local).
The engine, not the model, decides what the model may see:

* ``history``: every row of the hourly table whose hour started before the issue time;
* ``future``: the rows of day D restricted to columns known in advance (calendar, and
  weather forecasts or observed weather when an experiment asks for them).

A model therefore cannot read tomorrow's demand even by mistake.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

CALENDAR_COLUMNS = [
    "local_date",
    "local_hour",
    "utc_offset_h",
    "holiday_national",
    "holiday_intensity",
    "bridge_intensity",
    "weekday",
    "day_type",
    "christmas",
    "august",
    "holy_week",
    "local_hours",
]
# Columns computed with statistics of the whole sample: never shown to a model.
HIDDEN_COLUMNS = ["demand_spike"]


class Model(Protocol):
    name: str

    def fit(self, history: pd.DataFrame) -> None: ...

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        """Return a frame indexed like ``future`` with at least a ``point`` column."""
        ...


def issue_time_utc(target_date, issue_time_local: str, tz: str) -> pd.Timestamp:
    """UTC instant at which the forecast for local day ``target_date`` is issued."""
    origin = pd.Timestamp(target_date).normalize() - pd.Timedelta(days=1)
    hour, minute = (int(part) for part in issue_time_local.split(":"))
    local = (origin + pd.Timedelta(hours=hour, minutes=minute)).tz_localize(tz)
    return local.tz_convert("UTC")


def prepare_table(table: pd.DataFrame) -> pd.DataFrame:
    """Hide what no model should use: demand on masked days and whole-sample columns."""
    out = table.drop(columns=[c for c in HIDDEN_COLUMNS if c in table.columns]).copy()
    out.loc[out["is_masked"], "demand_mw"] = float("nan")
    return out


def split_information(
    table: pd.DataFrame,
    target_date,
    issue_time_local: str,
    tz: str,
    future_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The (history, future) pair for one target day. ``table`` must be sorted by time."""
    issue = issue_time_utc(target_date, issue_time_local, tz)
    history = table.iloc[: table.index.searchsorted(issue)]
    columns = future_columns or CALENDAR_COLUMNS
    future = table.loc[table["local_date"] == pd.Timestamp(target_date), columns]
    return history, future


def gap_information(
    table: pd.DataFrame, target_date, issue_time_local: str, tz: str, columns: list[str]
) -> pd.DataFrame:
    """Known-in-advance columns for the hours between the issue time and the target day.

    These are the remaining hours of the issue day. Demand there is unknown, but calendar
    and weather-forecast columns are available, and features such as a smoothed temperature
    need them.
    """
    issue = issue_time_utc(target_date, issue_time_local, tz)
    start = table.index.searchsorted(issue)
    rows = table.iloc[start : start + 48]
    return rows.loc[rows["local_date"] < pd.Timestamp(target_date), columns]


def run_backtest(
    model: Model,
    table: pd.DataFrame,
    start: str,
    end: str,
    *,
    issue_time_local: str,
    tz: str,
    refit_every: int = 1,
    future_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Forecast every local day in [start, end]. ``table`` should come from ``prepare_table``.

    ``refit_every`` is the number of days between calls to ``model.fit`` (on fixed calendar
    days, plus once at the start); ``predict`` always
    receives the up-to-date history. Models with ``needs_gap = True`` also receive the
    known-in-advance columns for the hours between the issue time and the target day.
    """
    parts = []
    for i, target_date in enumerate(pd.date_range(start, end, freq="D")):
        history, future = split_information(
            table, target_date, issue_time_local, tz, future_columns
        )
        if future.empty:
            continue
        # Refits are anchored to the calendar, so runs that start on different days share them.
        if i == 0 or target_date.toordinal() % refit_every == 0:
            model.fit(history)
        if getattr(model, "needs_gap", False):
            gap = gap_information(table, target_date, issue_time_local, tz, list(future.columns))
            pred = model.predict(history, future, gap)
        else:
            pred = model.predict(history, future)
        pred = pred.reindex(future.index)
        pred.insert(0, "local_date", future["local_date"])
        pred.insert(1, "local_hour", future["local_hour"])
        parts.append(pred)
    out = pd.concat(parts)
    out.attrs["model"] = model.name
    return out
