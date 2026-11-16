"""Calendar table by local date: holidays, bridge days, special periods, DST days.

Regional holidays are combined into a population-weighted intensity in [0, 1] over the
peninsular autonomous communities. A national holiday has intensity 1. Municipal holidays
are not covered by the ``holidays`` package and are ignored.
"""

from __future__ import annotations

import holidays
import pandas as pd
from dateutil.easter import easter

HOLIDAY_THRESHOLD = 0.5  # intensity from which a day counts as a holiday for the day type


def holiday_intensity(dates: pd.DatetimeIndex, region_population: dict[str, float]) -> pd.DataFrame:
    years = range(dates[0].year - 1, dates[-1].year + 2)
    pop = pd.Series(region_population, dtype="float64")
    weights = pop / pop.sum()
    national = holidays.country_holidays("ES", years=years)
    intensity = pd.Series(0.0, index=dates)
    for region, w in weights.items():
        regional = holidays.country_holidays("ES", subdiv=region, years=years)
        intensity += w * pd.Series(
            [d.date() in regional for d in dates], index=dates, dtype="float64"
        )
    out = pd.DataFrame(index=dates)
    out["holiday_national"] = [d.date() in national for d in dates]
    out["holiday_intensity"] = intensity.clip(upper=1.0).round(6)
    out.loc[out["holiday_national"], "holiday_intensity"] = 1.0
    return out


def build_calendar(
    start: str, end: str, tz: str, region_population: dict[str, float]
) -> pd.DataFrame:
    """One row per local date in [start, end]."""
    dates = pd.date_range(start, end, freq="D")
    # two days of margin so that bridge days at the edges see their neighbours
    padded = pd.date_range(
        dates[0] - pd.Timedelta(days=2), dates[-1] + pd.Timedelta(days=2), freq="D"
    )
    cal = holiday_intensity(padded, region_population)
    h = cal["holiday_intensity"]
    weekday = pd.Series(padded.day_of_week, index=padded)

    # Bridge: a Monday before a holiday Tuesday, or a Friday after a holiday Thursday.
    bridge = pd.Series(0.0, index=padded)
    bridge[weekday == 0] = h.shift(-1)[weekday == 0]
    bridge[weekday == 4] = h.shift(1)[weekday == 4]
    cal["bridge_intensity"] = (bridge.fillna(0.0) * (1.0 - h)).round(6)

    cal["weekday"] = weekday
    is_holiday = h >= HOLIDAY_THRESHOLD
    cal["day_type"] = "working"
    cal.loc[weekday == 5, "day_type"] = "saturday"
    cal.loc[(weekday == 6) | is_holiday, "day_type"] = "sunday_holiday"

    md = padded.month * 100 + padded.day
    cal["christmas"] = (md >= 1224) | (md <= 106)
    cal["august"] = padded.month == 8
    holy = pd.Series(False, index=padded)
    for year in range(padded[0].year, padded[-1].year + 1):
        sunday = pd.Timestamp(easter(year))
        holy |= (padded >= sunday - pd.Timedelta(days=7)) & (
            padded <= sunday + pd.Timedelta(days=1)
        )
    cal["holy_week"] = holy.to_numpy()

    # Days with 23 or 25 local hours.
    local_midnights = pd.DatetimeIndex(padded).tz_localize(tz)
    next_midnights = (pd.DatetimeIndex(padded) + pd.Timedelta(days=1)).tz_localize(tz)
    cal["local_hours"] = ((next_midnights - local_midnights) / pd.Timedelta(hours=1)).astype(int)

    cal = cal.loc[dates]
    cal.index.name = "local_date"
    return cal
