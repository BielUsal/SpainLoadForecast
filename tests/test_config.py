"""Smoke tests: the package imports and the configs are consistent."""

from datetime import date, timedelta

import loadforecast
from loadforecast.config import load_config


def test_version():
    assert loadforecast.__version__


def test_splits_are_contiguous_and_ordered():
    splits = load_config("backtest")["splits"]
    names = ["train", "validation", "test"]
    for name in names:
        start = date.fromisoformat(splits[name]["start"])
        end = date.fromisoformat(splits[name]["end"])
        assert start < end
    for a, b in zip(names, names[1:], strict=False):
        end_a = date.fromisoformat(splits[a]["end"])
        start_b = date.fromisoformat(splits[b]["start"])
        assert start_b == end_a + timedelta(days=1)


def test_splits_inside_data_period():
    period = load_config("data")["period"]
    splits = load_config("backtest")["splits"]
    assert splits["train"]["start"] >= period["start"]
    assert splits["test"]["end"] <= period["end"]


def test_city_list():
    cities = load_config("data")["weather"]["cities"]
    assert len({c["name"] for c in cities}) == len(cities)
    assert all(c["population"] > 0 for c in cities)
