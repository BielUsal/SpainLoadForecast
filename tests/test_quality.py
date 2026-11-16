import numpy as np
import pandas as pd

from loadforecast.data import quality
from loadforecast.data.esios import month_chunks
from loadforecast.data.weather import weighted_mean, year_chunks


def _series(hours=24 * 7 * 20):
    idx = pd.date_range("2023-01-02", periods=hours, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    daily = 5000 * np.sin(2 * np.pi * idx.hour / 24)
    return pd.Series(28000 + daily + rng.normal(0, 150, hours), index=idx)


def test_missing_hours():
    s = _series()
    assert len(quality.missing_hours(s.index)) == 0
    assert len(quality.missing_hours(s.drop(s.index[[10, 11, 500]]).index)) == 3


def test_gaps():
    s = _series()
    s.iloc[100:105] = np.nan
    s.iloc[300] = np.nan
    found = quality.gaps(s)
    assert sorted(found["hours"]) == [1, 5]
    assert found["start"].min() == s.index[100]


def test_spike_is_flagged_and_normal_data_is_not():
    s = _series()
    assert not quality.flag_spikes(s, s.index).any()
    s.iloc[1000] += 6000
    flags = quality.flag_spikes(s, s.index)
    assert flags.iloc[1000] and flags.iloc[1001]
    assert flags.sum() == 2


def test_out_of_range():
    s = _series()
    s.iloc[5] = 0.0
    assert list(quality.out_of_range(s, 15000, 45000).index) == [s.index[5]]


def test_month_chunks_cover_period_without_overlap():
    chunks = month_chunks("2021-01-01", "2021-04-15", "Europe/Madrid")
    assert len(chunks) == 4
    assert chunks[0][0] == "2021-01-01T00:00:00+01:00"
    assert chunks[2][1] == "2021-03-31T23:59:59+02:00"  # DST started on 28 March
    assert chunks[3][1] == "2021-04-15T23:59:59+02:00"
    for (_, end), (start, _) in zip(chunks, chunks[1:], strict=False):
        assert pd.Timestamp(start) - pd.Timestamp(end) == pd.Timedelta(seconds=1)


def test_year_chunks():
    assert year_chunks("2020-12-31", "2022-03-01") == [
        ("2020-12-31", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
        ("2022-01-01", "2022-03-01"),
    ]


def test_weighted_mean_renormalises_missing_cities():
    wide = pd.DataFrame({"a": [10.0, 10.0, np.nan], "b": [20.0, np.nan, np.nan]})
    out = weighted_mean(wide, pd.Series({"a": 0.75, "b": 0.25}))
    assert out.iloc[0] == 12.5
    assert out.iloc[1] == 10.0
    assert np.isnan(out.iloc[2])
