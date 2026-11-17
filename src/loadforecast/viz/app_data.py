"""Build the small data bundle that the dashboard reads.

``data/`` is not in the repository, so the deployed app cannot read it. This module copies
what the app needs for one split into ``app/data/`` (a few MB, committed with the code):

    app/data/<split>_predictions.parquet   one row per model and hour: point and quantiles
    app/data/<split>_actuals.parquet       one row per hour: demand, weather, kind of day
    app/data/<split>_scores.csv            the score table of reports/results
    app/data/<split>_meta.json             period, dataset hash, masked days

Usage:
    python -m loadforecast.viz.app_data --split validation
"""

from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import pandas as pd

from loadforecast.config import DATA_DIR, PROCESSED_DIR, ROOT, load_config
from loadforecast.evaluation.backtest import prepare_table
from loadforecast.models.quantiles import quantile_column

APP_DATA_DIR = ROOT / "app" / "data"
RESULTS_DIR = ROOT / "reports" / "results"
# Models shown in the app. The others (SVI variant, daily naive, adjusted climatology) add
# nothing a visitor would want to compare.
APP_MODELS = [
    "ree",
    "lgbm_forecast",
    "bayes_nuts_forecast",
    "rnn_forecast",
    "climatology",
    "naive_weekly",
    "lgbm_oracle",
    "bayes_nuts_oracle",
    "rnn_oracle",
]


def day_kind(table: pd.DataFrame) -> pd.Series:
    """One label per hour for the kind of day, special days first."""
    kind = np.select(
        [
            table["holiday_intensity"] >= 0.3,
            table["bridge_intensity"] >= 0.5,
            table["christmas"].astype(bool),
            table["holy_week"].astype(bool),
            table["day_type"] == "saturday",
            table["day_type"] == "sunday_holiday",
        ],
        ["Holiday", "Bridge day", "Christmas period", "Holy Week", "Saturday", "Sunday"],
        "Working day",
    )
    return pd.Series(kind, index=table.index)


def build(split: str) -> dict:
    cfg = load_config("backtest")
    taus = cfg["quantiles"]
    start, end = cfg["splits"][split]["start"], cfg["splits"][split]["end"]
    raw = pd.read_parquet(PROCESSED_DIR / "hourly.parquet")
    meta = json.loads((PROCESSED_DIR / "hourly.meta.json").read_text())
    table = prepare_table(raw)
    in_split = (table["local_date"] >= start) & (table["local_date"] <= end)

    actuals = table.loc[in_split, ["local_date", "local_hour", "is_masked"]].copy()
    actuals["demand_mw"] = raw.loc[actuals.index, "demand_mw"].round().astype("float32")
    actuals["temp_c"] = raw.loc[actuals.index, "temp_obs"].round(1).astype("float32")
    actuals["rad_wm2"] = raw.loc[actuals.index, "rad_obs"].round().astype("float32")
    actuals["day_kind"] = day_kind(table.loc[actuals.index]).astype("category")
    actuals["local_hour"] = actuals["local_hour"].astype("int8")

    columns = ["point", *[quantile_column(t) for t in taus]]
    parts = []
    for name in APP_MODELS:
        path = DATA_DIR / "predictions" / split / f"{name}.parquet"
        if not path.exists():
            print(f"skipping {name}: no predictions for {split}")
            continue
        pred = pd.read_parquet(path)[columns].reindex(actuals.index)
        parts.append(pred.round().astype("float32").assign(model=name))
    predictions = pd.concat(parts)
    predictions["model"] = predictions["model"].astype("category")

    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    actuals.to_parquet(APP_DATA_DIR / f"{split}_actuals.parquet")
    predictions.to_parquet(APP_DATA_DIR / f"{split}_predictions.parquet")
    shutil.copyfile(RESULTS_DIR / f"{split}.csv", APP_DATA_DIR / f"{split}_scores.csv")
    info = {
        "split": split,
        "start": start,
        "end": end,
        "dataset_hash": meta["hash"],
        "issue_time_local": cfg["issue_time_local"],
        "quantiles": taus,
        "masked_days": cfg.get("masked_days", []),
        "models": [str(m) for m in predictions["model"].cat.categories],
    }
    (APP_DATA_DIR / f"{split}_meta.json").write_text(json.dumps(info, indent=2))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    args = parser.parse_args()
    info = build(args.split)
    size = sum(p.stat().st_size for p in APP_DATA_DIR.glob(f"{args.split}_*"))
    print(f"{args.split}: {len(info['models'])} models, {size / 1e6:.1f} MB in {APP_DATA_DIR}")


if __name__ == "__main__":
    main()
