import pandas as pd
import pytest

from loadforecast.config import load_config
from loadforecast.data.calendar import build_calendar


@pytest.fixture(scope="module")
def cal():
    cfg = load_config("data")
    regions = cfg["calendar"]["region_population"]
    return build_calendar("2021-01-01", "2026-10-31", cfg["timezone"], regions), regions


def test_one_row_per_day(cal):
    table, _ = cal
    assert table.index.is_unique
    assert len(table) == (pd.Timestamp("2026-10-31") - pd.Timestamp("2021-01-01")).days + 1


def test_national_holiday(cal):
    table, _ = cal
    row = table.loc["2024-12-25"]
    assert row["holiday_national"] and row["holiday_intensity"] == 1.0
    assert row["day_type"] == "sunday_holiday"
    assert row["christmas"]


def test_regional_holiday_is_weighted(cal):
    table, regions = cal
    # 2 May is a holiday only in the Comunidad de Madrid.
    row = table.loc["2024-05-02"]
    assert not row["holiday_national"]
    assert row["holiday_intensity"] == pytest.approx(
        regions["MD"] / sum(regions.values()), abs=1e-4
    )
    assert row["day_type"] == "working"


def test_bridge_day(cal):
    table, _ = cal
    # Monday 31 Oct 2022 sits between a weekend and All Saints (Tuesday 1 Nov).
    assert table.loc["2022-10-31", "bridge_intensity"] == 1.0
    assert table.loc["2022-10-24", "bridge_intensity"] == 0.0


def test_holy_week_and_dst(cal):
    table, _ = cal
    assert table.loc["2024-03-29", "holy_week"]  # Good Friday
    assert not table.loc["2024-04-10", "holy_week"]
    assert table.loc["2024-03-31", "local_hours"] == 23
    assert table.loc["2024-10-27", "local_hours"] == 25
    assert table.loc["2024-06-01", "local_hours"] == 24
