"""Build the data layers: raw (API cache) -> interim (one table per source) -> processed.

Usage:
    python -m loadforecast.data.build            # everything
    python -m loadforecast.data.build interim    # download what is missing and parse
    python -m loadforecast.data.build processed  # merge the interim tables

All tables are indexed by the UTC start of each hour. Local time (Europe/Madrid) appears
only as the ``local_date`` and ``local_hour`` columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime

import pandas as pd

from loadforecast.config import INTERIM_DIR, PROCESSED_DIR, load_config
from loadforecast.data import quality
from loadforecast.data.calendar import build_calendar
from loadforecast.data.esios import EsiosClient, download_series
from loadforecast.data.weather import (
    LEADS,
    fetch_city,
    forecast_variables,
    population_weights,
    slug,
    weighted_mean,
)

SHORT = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "rh",
    "shortwave_radiation": "rad",
    "cloud_cover": "cloud",
    "wind_speed_10m": "wind",
}


def hourly_index(cfg: dict) -> pd.DatetimeIndex:
    """Every hour of the local-date period, as UTC hour starts."""
    tz = cfg["timezone"]
    start = pd.Timestamp(cfg["period"]["start"], tz=tz)
    end = pd.Timestamp(cfg["period"]["end"], tz=tz) + pd.Timedelta(days=1)
    idx = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="1h", inclusive="left")
    return idx.rename("time_utc")


def build_interim(cfg: dict) -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    tz, period, es = cfg["timezone"], cfg["period"], cfg["esios"]
    geo = es["geo_id_peninsula"]
    client = EsiosClient()

    print("ESIOS demand (native resolution)...")
    native = download_series(
        client, es["indicators"]["demand_real"], period["start"], period["end"], tz, geo
    )
    native.rename("demand_mw").rename_axis("time_utc").to_frame().to_parquet(
        INTERIM_DIR / "demand_native.parquet"
    )
    hourly = native.resample("1h").agg(["mean", "count"])
    hourly.columns = ["demand_mw", "demand_n_obs"]
    hourly.rename_axis("time_utc").to_parquet(INTERIM_DIR / "demand_hourly.parquet")

    print("ESIOS REE day-ahead forecast...")
    ree = download_series(
        client, es["indicators"]["demand_forecast_d1"], period["start"], period["end"], tz, geo
    )
    ree = ree.resample("1h").mean().rename("ree_forecast_mw").rename_axis("time_utc")
    ree.to_frame().to_parquet(INTERIM_DIR / "ree_forecast_d1.parquet")

    print("ESIOS self-consumption estimate (scheduled total minus scheduled)...")
    change = pd.Timestamp(es["definition_change"])
    first = change.replace(day=1).strftime("%Y-%m-%d")
    parts = {
        name: download_series(client, es["indicators"][name], first, period["end"], tz, geo)
        .resample("1h")
        .mean()
        for name in ("demand_scheduled_total", "demand_scheduled")
    }
    selfcons = (parts["demand_scheduled_total"] - parts["demand_scheduled"]).dropna()
    selfcons = selfcons.clip(lower=0).rename("selfcons_mw").rename_axis("time_utc")
    selfcons.to_frame().to_parquet(INTERIM_DIR / "selfcons_hourly.parquet")

    wx = cfg["weather"]
    utc_start = (pd.Timestamp(period["start"]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    utc_end = period["end"]
    products = {
        "archive": (wx["archive_url"], wx["variables"], "weather_observed.parquet"),
        "previous_runs": (
            wx["previous_runs_url"],
            forecast_variables(wx["variables"]),
            "weather_forecast.parquet",
        ),
    }
    for product, (url, variables, filename) in products.items():
        frames = []
        for city in wx["cities"]:
            print(f"Open-Meteo {product}: {city['name']}...")
            frame = fetch_city(product, url, city, utc_start, utc_end, variables)
            frame.insert(0, "city", slug(city["name"]))
            frames.append(frame.rename_axis("time_utc").reset_index())
        pd.concat(frames, ignore_index=True).to_parquet(INTERIM_DIR / filename, index=False)

    cal = build_calendar(period["start"], period["end"], tz, cfg["calendar"]["region_population"])
    cal.to_parquet(INTERIM_DIR / "calendar.parquet")
    print(f"Interim tables written to {INTERIM_DIR}")


def _national(
    long: pd.DataFrame, column: str, weights: pd.Series
) -> tuple[pd.Series, pd.DataFrame]:
    wide = long.pivot(index="time_utc", columns="city", values=column)
    return weighted_mean(wide, weights), wide


def build_processed(cfg: dict) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tz = cfg["timezone"]
    idx = hourly_index(cfg)
    local = idx.tz_convert(tz)

    df = pd.DataFrame(index=idx)
    df["local_date"] = local.tz_localize(None).normalize()
    df["local_hour"] = local.hour
    df["utc_offset_h"] = (local.tz_localize(None) - idx.tz_localize(None)) / pd.Timedelta(hours=1)

    # Definition change: from `definition_change` REE's published demand includes estimated
    # self-consumption, and from the following day so does its day-ahead forecast. Subtracting
    # the estimate keeps `demand_mw` and `ree_forecast_mw` on the pre-change definition.
    change = pd.Timestamp(cfg["esios"]["definition_change"])
    df["post_definition_change"] = df["local_date"] >= change
    selfcons = pd.read_parquet(INTERIM_DIR / "selfcons_hourly.parquet")["selfcons_mw"].reindex(idx)
    df["selfcons_mw"] = selfcons.where(df["post_definition_change"])

    demand = pd.read_parquet(INTERIM_DIR / "demand_hourly.parquet")
    df["demand_published_mw"] = demand["demand_mw"].reindex(idx)
    df["demand_mw"] = df["demand_published_mw"] - df["selfcons_mw"].where(
        df["post_definition_change"], 0.0
    )
    df["demand_n_obs"] = demand["demand_n_obs"].reindex(idx).fillna(0).astype(int)
    q = cfg["quality"]
    df["demand_spike"] = quality.flag_spikes(df["demand_mw"], local, q["spike_threshold"])

    ree = pd.read_parquet(INTERIM_DIR / "ree_forecast_d1.parquet")
    df["ree_forecast_published_mw"] = ree["ree_forecast_mw"].reindex(idx)
    forecast_includes = df["local_date"] > change
    df["ree_forecast_mw"] = df["ree_forecast_published_mw"] - df["selfcons_mw"].where(
        forecast_includes, 0.0
    )

    weights = population_weights(cfg["weather"]["cities"])
    observed = pd.read_parquet(INTERIM_DIR / "weather_observed.parquet")
    forecast = pd.read_parquet(INTERIM_DIR / "weather_forecast.parquet")
    city_temps = None
    for variable in cfg["weather"]["variables"]:
        short = SHORT[variable]
        national, wide = _national(observed, variable, weights)
        df[f"{short}_obs"] = national.reindex(idx)
        if short == "temp":
            city_temps = wide.reindex(idx).add_prefix("temp_obs_")
        for lead in LEADS:
            national, _ = _national(forecast, f"{variable}_previous_day{lead}", weights)
            df[f"{short}_fc{lead}"] = national.reindex(idx)
    df = df.join(city_temps)

    cal = pd.read_parquet(INTERIM_DIR / "calendar.parquet")
    df = df.join(cal, on="local_date")

    backtest = load_config("backtest")
    df["is_masked"] = False
    for item in backtest.get("masked_days", []):
        inside = (df["local_date"] >= item["start"]) & (df["local_date"] <= item["end"])
        df.loc[inside, "is_masked"] = True

    out = PROCESSED_DIR / "hourly.parquet"
    df.to_parquet(out)
    meta = describe(df, cfg)
    with open(PROCESSED_DIR / "hourly.meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"Processed table: {out}  rows={len(df)}  columns={df.shape[1]}  hash={meta['hash']}")
    return df


def dataset_hash(df: pd.DataFrame) -> str:
    """Content hash of the table (values and index), logged with every experiment."""
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:16]


def describe(df: pd.DataFrame, cfg: dict) -> dict:
    low, high = cfg["quality"]["demand_range_mw"]
    demand_gaps = quality.gaps(df["demand_mw"])
    return {
        "hash": dataset_hash(df),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": len(df),
        "columns": list(df.columns),
        "first_hour_utc": df.index[0],
        "last_hour_utc": df.index[-1],
        "demand_missing_hours": int(df["demand_mw"].isna().sum()),
        "demand_gaps": demand_gaps.to_dict("records"),
        "demand_out_of_range_hours": int(len(quality.out_of_range(df["demand_mw"], low, high))),
        "demand_spike_hours": int(df["demand_spike"].sum()),
        "ree_forecast_missing_hours": int(df["ree_forecast_mw"].isna().sum()),
        "selfcons_missing_hours_after_change": int(
            df.loc[df["post_definition_change"], "selfcons_mw"].isna().sum()
        ),
        "missing_by_column": {c: int(n) for c, n in df.isna().sum().items() if n},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", nargs="?", default="all", choices=["all", "interim", "processed"])
    args = parser.parse_args()
    cfg = load_config("data")
    if args.step in ("all", "interim"):
        build_interim(cfg)
    if args.step in ("all", "processed"):
        build_processed(cfg)


if __name__ == "__main__":
    main()
