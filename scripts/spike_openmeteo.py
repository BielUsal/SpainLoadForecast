"""Spike: what does Open-Meteo cover, and how big are the responses?

Checks, for Madrid and one week in each of several periods:
  * the reanalysis archive (observed weather, "oracle" experiment),
  * the historical forecast archive,
  * the previous-runs archive (forecasts as issued 1 and 2 days before the valid time).

No API key is needed. Run: python scripts/spike_openmeteo.py
"""

from __future__ import annotations

import requests

from loadforecast.config import load_config

WEEKS = [
    ("2021-01-11", "2021-01-17"),
    ("2022-07-11", "2022-07-17"),
    ("2023-07-10", "2023-07-16"),
    ("2024-01-15", "2024-01-21"),
    ("2024-11-04", "2024-11-10"),
    ("2025-07-14", "2025-07-20"),
    ("2026-10-19", "2026-10-25"),
]
PREV = ["temperature_2m", "temperature_2m_previous_day1", "temperature_2m_previous_day2"]


def fetch(url: str, lat: float, lon: float, start: str, end: str, hourly: list[str]):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(hourly),
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code != 200:
        return None, len(resp.content), f"HTTP {resp.status_code}: {resp.text[:120]}"
    return resp.json().get("hourly", {}), len(resp.content), ""


def coverage(hourly: dict, var: str) -> str:
    vals = hourly.get(var)
    if vals is None:
        return "absent"
    ok = sum(v is not None for v in vals)
    return f"{ok}/{len(vals)}"


def mae(a: list, b: list) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b, strict=False) if x is not None and y is not None]
    return sum(abs(x - y) for x, y in pairs) / len(pairs) if pairs else None


def main() -> None:
    cfg = load_config("data")["weather"]
    city = cfg["cities"][0]
    lat, lon = city["lat"], city["lon"]
    variables = cfg["variables"]
    print(f"City: {city['name']}  variables: {variables}\n")

    sizes = []
    for start, end in WEEKS:
        print(f"== {start} .. {end}")
        arch, n_arch, err = fetch(cfg["archive_url"], lat, lon, start, end, variables)
        if arch is None:
            print(f"  archive            : {err}")
        else:
            sizes.append(n_arch)
            cov = {v: coverage(arch, v) for v in variables}
            print(f"  archive            : {n_arch} bytes  {cov}")

        hist, n_hist, err = fetch(cfg["historical_forecast_url"], lat, lon, start, end, variables)
        if hist is None:
            print(f"  historical forecast: {err}")
        else:
            print(f"  historical forecast: {n_hist} bytes  temp {coverage(hist, 'temperature_2m')}")

        prev, n_prev, err = fetch(cfg["previous_runs_url"], lat, lon, start, end, PREV)
        if prev is None:
            print(f"  previous runs      : {err}")
        else:
            cov = {v.replace("temperature_2m", "T"): coverage(prev, v) for v in PREV}
            line = f"  previous runs      : {n_prev} bytes  {cov}"
            if arch is not None:
                for lead in (1, 2):
                    m = mae(
                        prev.get(f"temperature_2m_previous_day{lead}", []), arch["temperature_2m"]
                    )
                    if m is not None:
                        line += f"  MAE(day{lead} vs archive)={m:.2f}C"
            print(line)

    if sizes:
        per_week = sum(sizes) / len(sizes)
        n_vars = len(variables)
        print(f"\nArchive JSON: about {per_week / 1024:.1f} kB per city-week, {n_vars} variables")


if __name__ == "__main__":
    main()
