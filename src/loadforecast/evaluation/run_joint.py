"""Backtest the joint (whole-day) models of DESIGN 7.5 on one split and score their scenarios.

Usage:
    python -m loadforecast.evaluation.run_joint --split validation
    python -m loadforecast.evaluation.run_joint --split validation --only flow \\
        --set flow.target=score --tag score          # a variant, saved as flow_score
    python -m loadforecast.evaluation.run_joint --split test --final   # once, at the end
    python -m loadforecast.evaluation.run_joint --split validation --e2e  # end-to-end variants

Reads the RNN predictions in data/predictions/{history,validation,test}. For target day D each
model learns only from days up to D-2, refitted every ``refit_every_days`` on fixed calendar
days. Outputs:
    data/scenarios/<split>/<model>.npz            saved scenarios (dates, samples in MW)
    data/scenarios/<split>/<model>_daily.parquet  daily scores
    reports/results/<split>_joint.csv / .md       score table of every saved model
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from loadforecast.config import DATA_DIR, PROCESSED_DIR, load_config
from loadforecast.evaluation import metrics
from loadforecast.evaluation.backtest import prepare_table
from loadforecast.evaluation.dm_test import block_bootstrap_ci, diebold_mariano
from loadforecast.evaluation.run import PREDICTIONS_DIR, RESULTS_DIR
from loadforecast.models.joint import (
    Days,
    Marginals,
    build_days,
    concat_days,
    joint_models,
    load_model_config,
    prefer_forecast,
)

SCENARIOS_DIR = DATA_DIR / "scenarios"
POOL_SPLITS = ["history", "validation", "test"]
REFERENCE = "independent"


def load_days(split: str, base: str, weather: str, table, taus) -> Days | None:
    path = PREDICTIONS_DIR / split / f"{base}_{weather}.parquet"
    if not path.exists():
        return None
    return build_days(pd.read_parquet(path), table, taus, weather)


def error_pool(base: str, table, taus) -> Days:
    """Every day with an out-of-sample forecast, forecast-weather runs preferred."""
    parts = []
    for split in POOL_SPLITS:
        fc = load_days(split, base, "forecast", table, taus)
        obs = load_days(split, base, "oracle", table, taus)
        if fc is not None and obs is not None:
            parts.append(prefer_forecast(fc, obs))
        elif fc is not None or obs is not None:
            parts.append(fc if fc is not None else obs)
    pool = concat_days(parts)
    return pool.take(np.argsort(pool.dates.to_numpy()))


def score_day(y: np.ndarray, x: np.ndarray) -> dict:
    """Scores of one day's scenarios ``x`` (M, 24) against the outcome ``y`` (24,)."""
    return {
        "energy": metrics.energy_score(y, x),
        "variogram": metrics.variogram_score(y, x, p=0.5),
        "crps_hourly": float(metrics.crps_samples(y, x.T).mean()),
        "crps_peak": float(metrics.crps_samples([y.max()], x.max(axis=1)[None])[0]),
        "crps_energy_gwh": float(
            metrics.crps_samples([y.sum() / 1e3], x.sum(axis=1)[None] / 1e3)[0]
        ),
    }


def run(split: str, cfg: dict, only: str | None, tag: str | None) -> None:
    bt = load_config("backtest")
    taus = bt["quantiles"]
    start, end = bt["splits"][split]["start"], bt["splits"][split]["end"]
    table = prepare_table(pd.read_parquet(PROCESSED_DIR / "hourly.parquet"))
    pool = error_pool(cfg["base_model"], table, taus)
    target = load_days(split, cfg["base_model"], "forecast", table, taus)
    target = target.take((target.dates >= start) & (target.dates <= end))
    out_dir = SCENARIOS_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in joint_models(taus, cfg):
        if only and only != model.name:
            continue
        name = f"{model.name}_{tag}" if tag else model.name
        rng = np.random.default_rng(0)
        rows, saved = [], []
        tic = time.time()
        for i, date in enumerate(target.dates):
            if i == 0 or date.toordinal() % cfg["refit_every_days"] == 0:
                known = (pool.dates <= date - pd.Timedelta(days=2)) & pool.complete
                train = pool.take(known)
                if len(train) < cfg["min_train_days"]:
                    raise RuntimeError(f"{date.date()}: only {len(train)} days of past errors")
                model.fit(train)
            day = target.take([i])
            x = model.sample(day, cfg["n_samples"], rng)
            keep = rng.permutation(len(x))[: cfg["n_saved"]]  # mixes the seeds of a flow
            saved.append(x[keep].astype(np.float32))
            row = {"local_date": date}
            if day.complete[0]:
                row |= score_day(day.y[0], x)
            rows.append(row)
        daily = pd.DataFrame(rows).set_index("local_date")
        daily.to_parquet(out_dir / f"{name}_daily.parquet")
        np.savez_compressed(
            out_dir / f"{name}.npz", dates=target.dates.to_numpy(), samples=np.stack(saved)
        )
        print(f"{name}: {len(target)} days in {time.time() - tic:.0f} s", flush=True)


E2E_VARIANTS = ["e2e_raw", "e2e_inflate", "e2e_marginals"]


def inflation_factors(dates, y, x_med, x_lo, x_hi, window: int, min_days: int) -> np.ndarray:
    """For each day D, the factor that makes the 90% ranges of the model's own scenarios on
    days D-2 and earlier (over ``window`` days) cover 90% of the hours. 1 with too few days."""
    ratio = np.abs(y - x_med) / np.maximum((x_hi - x_lo) / 2, 1e-6)  # (D, 24)
    ok = ~np.isnan(ratio).any(axis=1)
    out = np.ones(len(dates))
    for i, date in enumerate(dates):
        past = ok & (dates <= date - pd.Timedelta(days=2))
        past &= dates > date - pd.Timedelta(days=2 + window)
        if past.sum() >= min_days:
            out[i] = np.quantile(ratio[past], 0.9)
    return out


def run_e2e(split: str, cfg: dict, only: str | None, tag: str | None) -> None:
    """Score the end-to-end scenarios of ``rnn_flow`` under the three calibrations."""
    bt = load_config("backtest")
    taus = bt["quantiles"]
    start, end = bt["splits"][split]["start"], bt["splits"][split]["end"]
    e2e = cfg["e2e"]
    table = prepare_table(pd.read_parquet(PROCESSED_DIR / "hourly.parquet"))
    target = load_days(split, e2e["base_model"], "forecast", table, taus)
    target = target.take((target.dates >= start) & (target.dates <= end))
    raw = np.load(SCENARIOS_DIR / split / "raw" / f"{e2e['base_model']}_forecast.npz")
    dates = pd.DatetimeIndex(raw["dates"])
    x_all = raw["point"][:, None, :] + raw["offsets"].astype(np.float32)  # (D, M, 24)

    # Outcomes for every day with scenarios, warm-up days included, for the inflation factor.
    wide = table.pivot_table(index="local_date", columns="local_hour", values="demand_mw")
    y_all = wide.reindex(index=dates, columns=range(24)).to_numpy()
    lo, med, hi = np.quantile(x_all, [0.05, 0.5, 0.95], axis=1)
    factor = pd.Series(
        inflation_factors(dates, y_all, med, lo, hi, e2e["window_days"], e2e["min_days"]),
        index=dates,
    )
    marginals = Marginals(taus)
    position = pd.Series(range(len(dates)), index=dates)
    out_dir = SCENARIOS_DIR / split
    for variant in E2E_VARIANTS:
        if only and only != variant:
            continue
        name = f"{variant}_{tag}" if tag else variant
        rng = np.random.default_rng(0)
        rows, saved = [], []
        for i, date in enumerate(target.dates):
            x = x_all[position[date]].astype(float)
            if variant == "e2e_inflate":
                m = np.median(x, axis=0)
                x = m + factor[date] * (x - m)
            elif variant == "e2e_marginals":
                ranks = x.argsort(axis=0).argsort(axis=0)
                scores = norm.ppf((ranks + 0.5) / len(x))
                x = marginals.values(target.q[i], scores.T).T
            keep = rng.permutation(len(x))[: cfg["n_saved"]]
            saved.append(x[keep].astype(np.float32))
            row = {"local_date": date}
            if target.complete[i]:
                row |= score_day(target.y[i], x)
            rows.append(row)
        daily = pd.DataFrame(rows).set_index("local_date")
        daily.to_parquet(out_dir / f"{name}_daily.parquet")
        np.savez_compressed(
            out_dir / f"{name}.npz", dates=target.dates.to_numpy(), samples=np.stack(saved)
        )
        f = factor.reindex(target.dates)
        print(f"{name}: {len(target)} days; inflation {f.min():.2f} to {f.max():.2f}", flush=True)


def summarise(split: str) -> pd.DataFrame:
    daily = {
        p.name.removesuffix("_daily.parquet"): pd.read_parquet(p).dropna()
        for p in sorted((SCENARIOS_DIR / split).glob("*_daily.parquet"))
    }
    table = pd.DataFrame({name: d.mean() for name, d in daily.items()}).T
    table.insert(0, "days", [len(d) for d in daily.values()])
    for ref in ("independent", "copula"):
        if ref not in daily:
            continue
        for name, d in daily.items():
            if name == ref:
                continue
            both = d.join(daily[ref].add_suffix("_ref")).dropna()
            for metric in ("energy", "variogram"):
                diff = diebold_mariano(both[metric], both[f"{metric}_ref"])
                low, high = block_bootstrap_ci(both[metric], both[f"{metric}_ref"])
                table.loc[name, f"{metric}_minus_{ref}"] = diff["mean_diff"]
                table.loc[name, f"{metric}_minus_{ref}_ci"] = f"{low:.1f} to {high:.1f}"
                table.loc[name, f"{metric}_minus_{ref}_p"] = diff["p_value"]
    table = table.sort_values("energy")
    table.index.name = "model"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / f"{split}_joint.csv")
    lines = [
        f"# Joint (whole-day) scores: {split}",
        "",
        "Scenarios of the 24 hours built on the RNN's forecasts (DESIGN 7.5). Daily means;",
        "energy score, CRPS of each hour, of the daily peak (MW) and of the daily energy (GWh);",
        "variogram score of order 0.5. Lower is better. Differences: mean daily difference with",
        "a 95% block-bootstrap interval and the Diebold-Mariano p-value.",
        "",
        "| model | " + " | ".join(table.columns) + " |",
        "|---|" + "---:|" * len(table.columns),
    ]
    for name, row in table.iterrows():
        cells = []
        for c, v in row.items():
            if isinstance(v, str):
                cells.append(v)
            elif pd.isna(v):
                cells.append("")
            elif c.endswith("_p"):
                cells.append(f"{v:.3g}")
            elif c == "days":
                cells.append(f"{v:.0f}")
            else:
                cells.append(f"{v:,.1f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (RESULTS_DIR / f"{split}_joint.md").write_text("\n".join(lines) + "\n")
    return table


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, value = item.split("=", 1)
        *path, last = key.split(".")
        node = cfg
        for part in path:
            node = node[part]
        node[last] = yaml.safe_load(value)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--only", default=None, help="one model: independent, copula or flow")
    parser.add_argument("--set", action="append", default=[], help="config override, a.b=value")
    parser.add_argument("--tag", default=None, help="suffix for the saved model name")
    parser.add_argument("--e2e", action="store_true", help="the end-to-end variants instead")
    parser.add_argument("--score-only", action="store_true", help="only rebuild the table")
    parser.add_argument("--final", action="store_true", help="required to score the test split")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("the test split is scored once, at the end: pass --final to confirm")
    if not args.score_only:
        runner = run_e2e if args.e2e else run
        runner(args.split, apply_overrides(load_model_config(), args.set), args.only, args.tag)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(summarise(args.split).round(3).to_string())


if __name__ == "__main__":
    main()
