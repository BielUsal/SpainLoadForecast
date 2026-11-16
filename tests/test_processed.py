"""Quality checks on the built dataset. Skipped until `make data` has been run."""

import pandas as pd
import pytest

from loadforecast.config import PROCESSED_DIR, load_config
from loadforecast.data import quality
from loadforecast.data.build import hourly_index

TABLE = PROCESSED_DIR / "hourly.parquet"
pytestmark = pytest.mark.skipif(not TABLE.exists(), reason="processed table not built")


@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(TABLE)


@pytest.fixture(scope="module")
def cfg():
    return load_config("data")


def test_index_is_complete_hourly_utc(df, cfg):
    assert str(df.index.tz) == "UTC"
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert len(quality.missing_hours(df.index)) == 0
    assert df.index.equals(hourly_index(cfg))


def test_local_days_have_23_24_or_25_hours(df):
    counts = df.groupby("local_date").size()
    assert set(counts.unique()) <= {23, 24, 25}
    assert (counts == df.groupby("local_date")["local_hours"].first()).all()
    assert (counts == 23).sum() == (counts == 25).sum() + (1 if counts.index[-1].month < 10 else 0)


def test_demand_mostly_present_and_in_range(df, cfg):
    low, high = cfg["quality"]["demand_range_mw"]
    demand = df.loc[~df["is_masked"], "demand_mw"]
    assert demand.isna().mean() < 0.001
    assert len(quality.out_of_range(demand, low, high)) == 0


def test_ree_forecast_mostly_present_and_sane(df):
    ok = df.loc[~df["is_masked"]].dropna(subset=["demand_mw", "ree_forecast_mw"])
    assert df["ree_forecast_mw"].isna().mean() < 0.01
    mape = ((ok["ree_forecast_mw"] - ok["demand_mw"]).abs() / ok["demand_mw"]).mean()
    assert 0.005 < mape < 0.03


def test_observed_weather_complete(df):
    assert df["temp_obs"].notna().all()
    assert df["temp_obs"].between(-15, 45).all()


def test_forecast_weather_covers_validation_and_test(df):
    recent = df.loc[df["local_date"] >= "2024-11-01"]
    assert recent["temp_fc2"].isna().mean() < 0.01


def test_definition_change_is_removed(df, cfg):
    """After the change, published demand minus the self-consumption estimate is the target."""
    change = pd.Timestamp(cfg["esios"]["definition_change"])
    before, after = df[df["local_date"] < change], df[df["local_date"] >= change]
    assert before["selfcons_mw"].isna().all()
    assert (before["demand_mw"] == before["demand_published_mw"]).all()
    assert after["selfcons_mw"].notna().all() and (after["selfcons_mw"] >= 0).all()
    night = after[after["local_hour"].isin([0, 1, 2, 3])]
    assert night["selfcons_mw"].max() < 1.0

    # The midday share of working-day demand must not jump across the change.
    def midday_share(frame):
        work = frame[frame["day_type"] == "working"]
        return (
            work.loc[work["local_hour"].between(12, 15), "demand_mw"].mean()
            / work["demand_mw"].mean()
        )

    month = pd.Timedelta(days=30)
    share_before = midday_share(before[before["local_date"] >= change - month])
    share_after = midday_share(after[after["local_date"] < change + month])
    assert abs(share_after - share_before) < 0.03
