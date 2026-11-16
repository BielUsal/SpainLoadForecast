"""Thin client for the ESIOS API (api.esios.ree.es) with a gzip raw-response cache.

The token is read from the ``ESIOS_API_KEY`` environment variable (loaded from ``.env``).
Raw responses are cached under ``data/raw/esios/<indicator>/`` and never edited.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from loadforecast.config import RAW_DIR, ROOT

BASE_URL = "https://api.esios.ree.es"
CACHE_DIR = RAW_DIR / "esios"


class EsiosError(RuntimeError):
    pass


class EsiosClient:
    def __init__(self, token: str | None = None, cache_dir: Path = CACHE_DIR, pause: float = 0.5):
        load_dotenv(ROOT / ".env")
        token = token or os.environ.get("ESIOS_API_KEY", "").strip()
        if not token:
            raise EsiosError("ESIOS_API_KEY is empty. Put your token in .env (see .env.example).")
        self.cache_dir = cache_dir
        self.pause = pause
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json; application/vnd.esios-api-v1+json",
                "Content-Type": "application/json",
                "x-api-key": token,
            }
        )

    def _get(self, path: str, params: dict[str, Any] | None = None, retries: int = 4) -> dict:
        url = f"{BASE_URL}{path}"
        for attempt in range(retries):
            resp = self.session.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                time.sleep(self.pause)
                return resp.json()
            if resp.status_code in (401, 403):
                raise EsiosError(f"ESIOS rejected the token (HTTP {resp.status_code}).")
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2**attempt * 2)
                continue
            raise EsiosError(f"GET {path} failed: HTTP {resp.status_code} {resp.text[:200]}")
        raise EsiosError(f"GET {path} failed after {retries} attempts")

    def list_indicators(self) -> list[dict]:
        """The full indicator catalogue (id, name, short_name, description)."""
        return self._get("/indicators")["indicators"]

    def get_indicator(
        self,
        indicator_id: int,
        start: str,
        end: str,
        *,
        geo_ids: list[int] | None = None,
        time_trunc: str | None = None,
        use_cache: bool = True,
    ) -> dict:
        """Fetch one indicator between ``start`` and ``end`` (ISO 8601 strings with offset).

        Returns the ``indicator`` object of the response, whose ``values`` list holds the data.
        """
        key = f"{start}_{end}_{time_trunc or 'native'}_{'-'.join(map(str, geo_ids or []))}"
        key = key.replace(":", "").replace("+", "p")
        cache_file = self.cache_dir / str(indicator_id) / f"{key}.json.gz"
        if use_cache and cache_file.exists():
            with gzip.open(cache_file, "rt", encoding="utf-8") as fh:
                return json.load(fh)["indicator"]

        params: dict[str, Any] = {"start_date": start, "end_date": end}
        if time_trunc:
            params["time_trunc"] = time_trunc
        if geo_ids:
            params["geo_ids[]"] = geo_ids
        payload = self._get(f"/indicators/{indicator_id}", params)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache_file, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return payload["indicator"]


def month_chunks(start: str, end: str, tz: str) -> list[tuple[str, str]]:
    """Split the local-date range [start, end] into calendar months as ISO strings with offset."""
    first = pd.Timestamp(start, tz=tz)
    last = pd.Timestamp(end, tz=tz) + pd.Timedelta(days=1)
    chunks = []
    cur = first
    while cur < last:
        nxt = min((cur + pd.offsets.MonthBegin(1)).normalize(), last)
        chunks.append((cur.isoformat(), (nxt - pd.Timedelta(seconds=1)).isoformat()))
        cur = nxt
    return chunks


def values_to_series(indicator: dict, geo_id: int) -> pd.Series:
    """The values of one API response as a UTC-indexed float series (interval-start labels)."""
    rows = [(v["datetime_utc"], v["value"]) for v in indicator["values"] if v["geo_id"] == geo_id]
    if not rows:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    times, vals = zip(*rows, strict=True)
    return pd.Series(vals, index=pd.to_datetime(list(times), utc=True), dtype="float64")


def download_series(
    client: EsiosClient, indicator_id: int, start: str, end: str, tz: str, geo_id: int
) -> pd.Series:
    """Download (or read from cache) one indicator month by month over local dates [start, end]."""
    parts = []
    for chunk_start, chunk_end in month_chunks(start, end, tz):
        parts.append(
            values_to_series(client.get_indicator(indicator_id, chunk_start, chunk_end), geo_id)
        )
    series = pd.concat(parts).sort_index()
    return series[~series.index.duplicated(keep="last")]
