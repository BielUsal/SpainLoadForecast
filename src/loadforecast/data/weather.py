"""Open-Meteo downloads (no API key) with a gzip raw-response cache.

Two products per city, both hourly and in UTC:
  * ``archive``: reanalysis, i.e. observed weather (the "oracle" experiment);
  * ``previous_runs``: the forecast as issued 1 and 2 days before each valid hour.

Raw responses are cached under ``data/raw/openmeteo/<product>/<city>/<year>.json.gz``.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import pandas as pd
import requests

from loadforecast.config import RAW_DIR

CACHE_DIR = RAW_DIR / "openmeteo"
LEADS = (1, 2)


class WeatherError(RuntimeError):
    pass


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def year_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end] (dates) into calendar-year pieces."""
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    chunks = []
    for year in range(first.year, last.year + 1):
        a = max(first, pd.Timestamp(year=year, month=1, day=1))
        b = min(last, pd.Timestamp(year=year, month=12, day=31))
        chunks.append((a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")))
    return chunks


def _get(url: str, params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(2**attempt * 5)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(60 if resp.status_code == 429 else 2**attempt * 2)
            continue
        raise WeatherError(f"Open-Meteo HTTP {resp.status_code}: {resp.text[:200]}")
    raise WeatherError(f"Open-Meteo request failed after {retries} attempts")


def fetch_city(
    product: str,
    url: str,
    city: dict,
    start: str,
    end: str,
    hourly: list[str],
    *,
    cache_dir: Path = CACHE_DIR,
    pause: float = 3.0,
) -> pd.DataFrame:
    """Hourly variables for one city over [start, end] (UTC dates), as a UTC-indexed frame."""
    frames = []
    for chunk_start, chunk_end in year_chunks(start, end):
        cache_file = cache_dir / product / slug(city["name"]) / f"{chunk_start}_{chunk_end}.json.gz"
        if cache_file.exists():
            with gzip.open(cache_file, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
        else:
            params = {
                "latitude": city["lat"],
                "longitude": city["lon"],
                "start_date": chunk_start,
                "end_date": chunk_end,
                "hourly": ",".join(hourly),
                "timezone": "UTC",
            }
            payload = _get(url, params)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(cache_file, "wt", encoding="utf-8") as fh:
                json.dump(payload, fh)
            time.sleep(pause)
        data = payload["hourly"]
        frame = pd.DataFrame({k: v for k, v in data.items() if k != "time"}, dtype="float64")
        frame.index = pd.to_datetime(data["time"], utc=True)
        frames.append(frame)
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def forecast_variables(variables: list[str]) -> list[str]:
    return [f"{v}_previous_day{lead}" for v in variables for lead in LEADS]


def population_weights(cities: list[dict]) -> pd.Series:
    pop = pd.Series({slug(c["name"]): float(c["population"]) for c in cities})
    return pop / pop.sum()


def weighted_mean(wide: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Population-weighted mean across city columns, renormalised where a city is missing."""
    w = weights.reindex(wide.columns)
    num = wide.mul(w, axis=1).sum(axis=1, min_count=1)
    den = wide.notna().mul(w, axis=1).sum(axis=1)
    return num / den.where(den > 0)
