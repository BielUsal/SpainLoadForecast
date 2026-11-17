"""Run the backtest for a set of models on one split and write predictions and scores.

Usage:
    python -m loadforecast.evaluation.run --split validation
    python -m loadforecast.evaluation.run --split validation --models lgbm   # one group only
    python -m loadforecast.evaluation.run --split test --final   # once, at the end

Outputs:
    data/predictions/<split>/<model>.parquet   hourly predictions (point and quantiles)
    reports/results/<split>.csv / .md / .json  score table and run metadata
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import pandas as pd

from loadforecast.config import DATA_DIR, PROCESSED_DIR, ROOT, load_config
from loadforecast.evaluation import scoring
from loadforecast.evaluation.backtest import prepare_table, run_backtest
from loadforecast.evaluation.dm_test import block_bootstrap_ci, diebold_mariano
from loadforecast.evaluation.metrics import skill_score
from loadforecast.models.baselines import (
    Climatology,
    ReeForecast,
    SeasonalNaiveDaily,
    SeasonalNaiveWeekly,
)
from loadforecast.models.quantiles import (
    add_residual_quantiles,
    point_from_median,
    recalibrate_quantiles,
)

PREDICTIONS_DIR = DATA_DIR / "predictions"
RESULTS_DIR = ROOT / "reports" / "results"
WARMUP_DAYS = 400  # error history needed before the split for the residual quantiles
REFERENCE = "naive_weekly"  # skill scores are relative to this model
BENCHMARK = "ree"


def baseline_models(table: pd.DataFrame) -> list[tuple[object, int]]:
    """(model, refit cadence in days)."""
    return [
        (SeasonalNaiveWeekly(), 1),
        (SeasonalNaiveDaily(), 1),
        (Climatology(), 28),
        (Climatology(level_adjust=True), 28),
        (ReeForecast(table), 1),
    ]


MODEL_GROUPS = ["baselines", "lgbm", "bayes", "rnn"]


def build_models(
    groups: list[str], table: pd.DataFrame, dataset_hash: str, cfg: dict, split: str = "validation"
):
    models = []
    if "baselines" in groups:
        models += baseline_models(table)
    if "lgbm" in groups:
        from loadforecast.models.gbm import lightgbm_models  # imported late: optional dependency

        models += lightgbm_models(table, dataset_hash, cfg)
    if "bayes" in groups:
        from loadforecast.models.bayes import bayes_models

        models += bayes_models(cfg["quantiles"], dataset_hash, split=split)
    if "rnn" in groups:
        from loadforecast.models.rnn import rnn_models

        models += rnn_models(cfg["quantiles"])
    return models


def run_models(split: str, groups: list[str], only: str | None = None) -> None:
    """Backtest the chosen model groups and save their predictions for the split."""
    data_cfg, cfg = load_config("data"), load_config("backtest")
    raw = pd.read_parquet(PROCESSED_DIR / "hourly.parquet")
    meta = json.loads((PROCESSED_DIR / "hourly.meta.json").read_text())
    table = prepare_table(raw)
    taus = cfg["quantiles"]
    start, end = cfg["splits"][split]["start"], cfg["splits"][split]["end"]
    first_possible = pd.Timestamp(data_cfg["period"]["start"]) + pd.Timedelta(days=35)
    warm_start = max(pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS), first_possible)

    out_dir = PREDICTIONS_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, refit_every in build_models(groups, table, meta["hash"], cfg, split):
        if only and only not in model.name:
            continue
        native = getattr(model, "native_quantiles", False)
        recalibrate = getattr(model, "recalibrate", False)
        needs_history = recalibrate or not native  # past errors from before the split
        model_start = warm_start if needs_history else pd.Timestamp(start)
        if getattr(model, "min_start", None):
            model_start = max(model_start, pd.Timestamp(model.min_start))
        print(f"{model.name}: backtest {model_start.date()} -> {end}", flush=True)
        pred = run_backtest(
            model,
            table,
            model_start,
            end,
            issue_time_local=cfg["issue_time_local"],
            tz=cfg["timezone"],
            refit_every=refit_every,
            future_columns=getattr(model, "future_columns", None),
        )
        if not native:
            pred = add_residual_quantiles(pred, table["demand_mw"], taus)
        elif recalibrate:
            pred = recalibrate_quantiles(pred, table["demand_mw"], taus)
        if getattr(model, "point_from_median", False):
            pred = point_from_median(pred)
        pred = pred[pred["local_date"] >= start]
        pred.attrs = {}
        pred.to_parquet(out_dir / f"{model.name}.parquet")


def score_split(split: str) -> pd.DataFrame:
    """Score every saved prediction file of the split and write the results table."""
    cfg = load_config("backtest")
    raw = pd.read_parquet(PROCESSED_DIR / "hourly.parquet")
    meta = json.loads((PROCESSED_DIR / "hourly.meta.json").read_text())
    taus = cfg["quantiles"]
    start, end = cfg["splits"][split]["start"], cfg["splits"][split]["end"]
    rows, daily = {}, {}
    for path in sorted((PREDICTIONS_DIR / split).glob("*.parquet")):
        scored = scoring.join_actual(pd.read_parquet(path), raw)
        rows[path.stem] = scoring.summary(scored, taus)
        daily[path.stem] = scoring.daily_losses(scored, taus)

    results = pd.DataFrame(rows).T.sort_values("MAE", ascending=False)
    results.index.name = "model"
    if REFERENCE in results.index:
        for metric in ("MAE", "CRPS"):
            results[f"skill_{metric}"] = [
                skill_score(v, results.loc[REFERENCE, metric]) for v in results[metric]
            ]
    if BENCHMARK in daily:
        for name, losses in daily.items():
            if name == BENCHMARK:
                continue
            both = losses.add_suffix("_m").join(daily[BENCHMARK].add_suffix("_b")).dropna()
            test = diebold_mariano(both["abs_error_m"], both["abs_error_b"])
            low, high = block_bootstrap_ci(both["abs_error_m"], both["abs_error_b"])
            results.loc[name, "MAE_minus_ree"] = test["mean_diff"]
            results.loc[name, "ci_low"] = low
            results.loc[name, "ci_high"] = high
            results.loc[name, "dm_p"] = test["p_value"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / f"{split}.csv")
    (RESULTS_DIR / f"{split}.md").write_text(to_markdown(results, split, meta["hash"]))
    run_meta = {
        "split": split,
        "start": start,
        "end": end,
        "dataset_hash": meta["hash"],
        "issue_time_local": cfg["issue_time_local"],
        "quantiles": taus,
        "masked_days": cfg.get("masked_days", []),
        "models": list(results.index),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (RESULTS_DIR / f"{split}.json").write_text(json.dumps(run_meta, indent=2))
    return results


def to_markdown(results: pd.DataFrame, split: str, dataset_hash: str) -> str:
    formats = {"hours": "{:,.0f}", "MAPE": "{:.2f}", "skill_MAE": "{:.3f}", "skill_CRPS": "{:.3f}"}
    formats |= {c: "{:.1f}" for c in results.columns if c.startswith("cov")}
    formats |= {"dm_p": "{:.3g}"}
    lines = [
        f"# Results: {split}",
        "",
        f"Dataset `{dataset_hash}`. MAE, RMSE, bias, CRPS and widths in MW; MAPE, coverage in %.",
        f"Skill is relative to `{REFERENCE}`. `MAE_minus_ree` is the mean daily MAE difference to",
        "REE's forecast with a 95% block-bootstrap interval and the Diebold-Mariano p-value.",
        "",
        "| model | " + " | ".join(results.columns) + " |",
        "|---|" + "---:|" * len(results.columns),
    ]
    for name, row in results.iterrows():
        cells = ["" if pd.isna(v) else formats.get(c, "{:,.0f}").format(v) for c, v in row.items()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument(
        "--models",
        default=",".join(MODEL_GROUPS),
        help=f"comma-separated groups ({', '.join(MODEL_GROUPS)}); 'none' only re-scores",
    )
    parser.add_argument("--only", default=None, help="keep models whose name contains this text")
    parser.add_argument("--final", action="store_true", help="required to score the test split")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("the test split is scored once, at the end: pass --final to confirm")
    groups = [] if args.models == "none" else args.models.split(",")
    unknown = set(groups) - set(MODEL_GROUPS)
    if unknown:
        parser.error(f"unknown model groups: {sorted(unknown)}")
    run_models(args.split, groups, args.only)
    results = score_split(args.split)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(results.round(3).to_string())


if __name__ == "__main__":
    main()
