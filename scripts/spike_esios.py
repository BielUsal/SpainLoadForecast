"""Spike: verify the ESIOS indicators before the pipeline depends on them.

1. Searches the indicator catalogue for demand-related entries.
2. For each candidate ID, fetches one week in 2021 and one recent week and reports the name,
   time step, geographies, coverage, metadata fields and response size.

Needs ESIOS_API_KEY in .env. Run: python scripts/spike_esios.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime

from loadforecast.config import load_config
from loadforecast.data.esios import EsiosClient, EsiosError

WEEKS = [
    ("2021-01-11T00:00:00+01:00", "2021-01-17T23:59:59+01:00"),
    ("2026-10-19T00:00:00+02:00", "2026-10-25T23:59:59+01:00"),
]
META_KEYS = ["name", "short_name", "step_type", "tiempo", "magnitud", "values_updated_at"]


def strip_html(text: str | None) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").strip()


def summarise(ind: dict) -> None:
    values = ind.get("values", [])
    print(f"    n_values={len(values)}  json_bytes={len(json.dumps(ind))}")
    if not values:
        return
    print(f"    value keys: {sorted(values[0].keys())}")
    geos = Counter((v.get("geo_id"), v.get("geo_name")) for v in values)
    print(f"    geographies: {dict(geos)}")
    top_geo = geos.most_common(1)[0][0][0]
    times = sorted(
        datetime.fromisoformat(v["datetime_utc"].replace("Z", "+00:00"))
        for v in values
        if v.get("geo_id") == top_geo
    )
    steps = Counter((b - a).total_seconds() / 60 for a, b in zip(times, times[1:], strict=False))
    print(f"    first={times[0]}  last={times[-1]}")
    print(f"    step minutes (count): {dict(steps.most_common(4))}")
    print(f"    sample: {values[0]}")


def main() -> None:
    cfg = load_config("data")["esios"]
    client = EsiosClient()

    print("== Catalogue entries mentioning demand ==")
    catalogue = client.list_indicators()
    print(f"catalogue size: {len(catalogue)}")
    for ind in catalogue:
        name = ind.get("name", "")
        if re.search(r"demanda", name, re.I) and re.search(
            r"previs|real|program|peninsul", name, re.I
        ):
            print(f"  {ind['id']:>6}  {name}")

    print("\n== Candidate indicators ==")
    candidates = {**cfg["indicators"], **cfg.get("rejected_indicators", {})}
    for label, ind_id in candidates.items():
        print(f"\n[{ind_id}] {label}")
        for i, (start, end) in enumerate(WEEKS):
            try:
                ind = client.get_indicator(ind_id, start, end)
            except EsiosError as exc:
                print(f"  {start[:10]}: ERROR {exc}")
                continue
            if i == 0:
                for key in META_KEYS:
                    print(f"  {key}: {ind.get(key)}")
                print(f"  description: {strip_html(ind.get('description'))[:400]}")
            print(f"  week {start[:10]}:")
            summarise(ind)


if __name__ == "__main__":
    main()
