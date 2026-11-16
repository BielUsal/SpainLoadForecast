"""Backtest wrapper and fitting routines for the Bayesian structural model.

The model itself is in ``bayes_numpyro``. This module turns the engine's hourly frames into
the model's day-by-hour grid, runs the two fitting stages, and produces quantile forecasts
from the posterior predictive.

Two ways of getting the posterior are supported:
  * ``inference="svi"``: refit at every backtest refit, by posterior mode plus a low-rank
    Gaussian approximation (about a minute per refit);
  * ``inference="nuts"``: one NUTS fit made beforehand with
    ``python -m loadforecast.models.bayes``, kept fixed through the backtest. The forecast
    still adapts every day, because it conditions on the latest residuals.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

from loadforecast.config import DATA_DIR, PROCESSED_DIR, load_config
from loadforecast.features.build import future_columns, weather_sources
from loadforecast.models.quantiles import quantile_column

MODELS_DIR = DATA_DIR / "models"
GRID_CALENDAR = [
    "weekday",
    "holiday_intensity",
    "bridge_intensity",
    "christmas",
    "august",
    "holy_week",
]
GRID_COLUMNS = ["local_date", "local_hour", "demand_mw", "temp", "rad", *GRID_CALENDAR]


def load_model_config() -> dict:
    return load_config("models/bayes")


def nuts_path(dataset_hash: str):
    return MODELS_DIR / f"bayes_nuts_{dataset_hash}.pkl"


def training_frame(history: pd.DataFrame, start: str, end=None) -> pd.DataFrame:
    """Observed-weather rows of whole days in [start, end]; ``end`` defaults to the last
    fully observed day of ``history``."""
    if end is None:
        end = history["local_date"].iloc[-1] - pd.Timedelta(days=1)
    rows = history[(history["local_date"] >= start) & (history["local_date"] <= end)]
    frame = rows.assign(temp=rows["temp_obs"], rad=rows["rad_obs"])
    return frame[GRID_COLUMNS]


def two_stage_fit(grid, cfg: dict, inference: str, init: dict | None = None, seed: int = 0) -> dict:
    """Fit structure, then the error model on its residuals. Returns draws and extras."""
    import jax.numpy as jnp

    from loadforecast.models import bayes_numpyro as bm

    svi = cfg["svi"]
    args = bm.stage_args(grid)
    started = time.time()
    first = init is None
    steps = svi["map_steps"] if first else svi["refit_map_steps"]
    structure_map = bm.fit_map(
        bm.structure_model, args, steps, seed, None if first else init["structure"]
    )
    resid = bm.residuals(structure_map, args[0])
    error_args = (resid, args[1])
    error_map = bm.fit_map(
        bm.error_model, error_args, steps // 2, seed, None if first else init["errors"]
    )
    out = {"map": {"structure": structure_map, "errors": error_map}, "n_days": grid.n_days}

    if inference == "svi":
        n = cfg["n_draws"]
        structure = bm.fit_svi(
            bm.structure_model, args, structure_map, svi["structure_steps"], seed=seed, n_samples=n
        )
        errors = bm.fit_svi(
            bm.error_model, error_args, error_map, svi["error_steps"], seed=seed, n_samples=n
        )
    elif inference == "nuts":
        nuts = cfg["nuts"]
        kwargs = dict(
            warmup=nuts["warmup"],
            samples=nuts["samples"],
            chains=nuts["chains"],
            seed=seed,
            max_tree_depth=nuts["max_tree_depth"],
            dense_mass=nuts.get("dense_mass", False),
        )
        mcmc_structure = bm.fit_nuts(bm.structure_model, args, structure_map, **kwargs)
        by_chain = mcmc_structure.get_samples(group_by_chain=True)
        structure = mcmc_structure.get_samples()
        # Residuals of the posterior-mean surface feed the error model.
        thin = {k: v[:: max(1, len(v) // 200)] for k, v in structure.items()}
        import jax

        mean = jax.vmap(lambda p: bm.mean_fn(p, args[0]))(thin).mean(axis=0)
        resid = jnp.where(jnp.isnan(args[0]["y"]), 0.0, args[0]["y"] - mean)
        mcmc_errors = bm.fit_nuts(bm.error_model, (resid, args[1]), error_map, **kwargs)
        errors = mcmc_errors.get_samples()
        out["chains"] = {
            "structure": {k: np.asarray(v) for k, v in by_chain.items()},
            "errors": {
                k: np.asarray(v) for k, v in mcmc_errors.get_samples(group_by_chain=True).items()
            },
        }
        out["extra_fields"] = {
            "structure": {
                k: np.asarray(v)
                for k, v in mcmc_structure.get_extra_fields(group_by_chain=True).items()
            },
            "errors": {
                k: np.asarray(v)
                for k, v in mcmc_errors.get_extra_fields(group_by_chain=True).items()
            },
        }
    else:
        raise ValueError(f"unknown inference: {inference}")

    out["structure"] = {k: np.asarray(v) for k, v in structure.items()}
    out["errors"] = {k: np.asarray(v) for k, v in errors.items()}
    out["map"] = {
        stage: {k: np.asarray(v) for k, v in values.items()} for stage, values in out["map"].items()
    }
    out["seconds"] = time.time() - started
    return out


def paired_draws(fit: dict, n: int, seed: int = 0) -> tuple[dict, dict]:
    """``n`` draws of each stage, paired at random (the stages are fitted separately)."""
    rng = np.random.default_rng(seed)
    out = []
    for stage in ("structure", "errors"):
        total = len(next(iter(fit[stage].values())))
        pick = rng.choice(total, size=n, replace=total < n)
        out.append({k: v[pick] for k, v in fit[stage].items()})
    return out[0], out[1]


class BayesForecaster:
    needs_gap = True
    native_quantiles = True
    _fits: dict[str, dict] = {}  # shared between weather variants, keyed by training cutoff

    def __init__(
        self,
        weather: str,
        taus: list[float],
        cfg: dict | None = None,
        inference: str = "svi",
        fixed_fit: dict | None = None,
    ):
        from loadforecast.models import bayes_numpyro as bm

        self.bm = bm
        self.cfg = cfg or load_model_config()
        self.weather = weather
        self.taus = taus
        self.inference = inference
        self.name = f"bayes_{inference}_{weather}"
        self.future_columns = future_columns(weather)
        self.target_src, self.gap_src = weather_sources(weather)
        self.fixed_fit = fixed_fit
        self.draws: tuple[dict, dict] | None = None
        self.last_fit: dict | None = None
        self.rng = np.random.default_rng(0)
        self.context_days = bm.CARRY_DAYS + bm.WARMUP_DAYS + 2
        self.min_start = "2024-01-21" if weather == "forecast" else None

    def fit(self, history: pd.DataFrame) -> None:
        if self.fixed_fit is not None:
            if self.draws is None:
                self.draws = paired_draws(self.fixed_fit, self.cfg["n_draws"])
            return
        cutoff = history["local_date"].iloc[-1]
        key = json.dumps([str(cutoff), self.inference, self.cfg], sort_keys=True, default=str)
        if key not in self._fits:
            grid = self.bm.build_grid(training_frame(history, self.cfg["train_start"]))
            previous = self._latest_fit()
            init = previous["map"] if previous else None
            self._fits[key] = two_stage_fit(grid, self.cfg, self.inference, init) | {
                "cutoff": cutoff
            }
        self.last_fit = self._fits[key]
        self.draws = paired_draws(self.last_fit, self.cfg["n_draws"])

    def _latest_fit(self) -> dict | None:
        fits = [f for f in self._fits.values() if "cutoff" in f]
        return max(fits, key=lambda f: f["cutoff"]) if fits else None

    def prediction_grid(self, history: pd.DataFrame, future: pd.DataFrame, gap: pd.DataFrame):
        target_day = future["local_date"].iloc[0]
        first = target_day - pd.Timedelta(days=self.context_days - 1)
        past = history[history["local_date"] >= first]
        parts = [
            past.assign(temp=past["temp_obs"], rad=past["rad_obs"]),
            gap.assign(
                temp=gap[self.gap_src["temp"]], rad=gap[self.gap_src["rad"]], demand_mw=np.nan
            ),
            future.assign(
                temp=future[self.target_src["temp"]],
                rad=future[self.target_src["rad"]],
                demand_mw=np.nan,
            ),
        ]
        return self.bm.build_grid(pd.concat([p[GRID_COLUMNS] for p in parts]))

    def predict(self, history: pd.DataFrame, future: pd.DataFrame, gap: pd.DataFrame):
        grid = self.prediction_grid(history, future, gap)
        structure, errors = self.draws
        out = self.bm.predict_last_day(
            structure, errors, grid, self.rng, self.cfg["draws_per_sample"]
        )
        quantiles = np.quantile(out["samples"], self.taus, axis=0)  # (taus, 24)
        hours = future["local_hour"].to_numpy()
        pred = pd.DataFrame(index=future.index)
        pred["point"] = np.median(out["samples"], axis=0)[hours]
        for tau, values in zip(self.taus, quantiles, strict=True):
            pred[quantile_column(tau)] = values[hours]
        pred["structural_mean"] = out["structural_mean"].mean(axis=0)[hours]
        return pred


def bayes_models(
    taus: list[float], dataset_hash: str, variants: tuple[str, ...] = ("svi", "nuts")
) -> list[tuple[object, int]]:
    cfg = load_model_config()
    models = []
    if "svi" in variants:
        for weather in ("oracle", "forecast"):
            models.append((BayesForecaster(weather, taus, cfg, "svi"), cfg["refit_every_days"]))
    if "nuts" in variants and nuts_path(dataset_hash).exists():
        with open(nuts_path(dataset_hash), "rb") as fh:
            fixed = pickle.load(fh)
        for weather in ("oracle", "forecast"):
            models.append((BayesForecaster(weather, taus, cfg, "nuts", fixed), 10**6))
    return models


def main() -> None:
    """Run the single NUTS fit on the training period and save it."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--chains", type=int, default=None)
    args = parser.parse_args()

    import numpyro

    cfg = load_model_config()
    if args.chains:
        cfg["nuts"]["chains"] = args.chains
    numpyro.set_host_device_count(cfg["nuts"]["chains"])

    from loadforecast.evaluation.backtest import prepare_table
    from loadforecast.models import bayes_numpyro as bm

    raw = pd.read_parquet(PROCESSED_DIR / "hourly.parquet")
    meta = json.loads((PROCESSED_DIR / "hourly.meta.json").read_text())
    table = prepare_table(raw)
    frame = training_frame(table, cfg["train_start"], cfg["nuts"]["train_end"])
    grid = bm.build_grid(frame)
    print(f"NUTS on {grid.n_days} days, {cfg['nuts']['chains']} chains", flush=True)
    fit = two_stage_fit(grid, cfg, "nuts")
    fit |= {"train_start": cfg["train_start"], "train_end": cfg["nuts"]["train_end"]}
    fit |= {"dataset_hash": meta["hash"], "config": cfg["nuts"]}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(nuts_path(meta["hash"]), "wb") as fh:
        pickle.dump(fit, fh)
    for stage in ("structure", "errors"):
        extra = fit["extra_fields"][stage]
        print(
            f"{stage}: divergences {int(extra['diverging'].sum())}, "
            f"mean leapfrog steps {extra['num_steps'].mean():.0f}"
        )
    print(f"saved {nuts_path(meta['hash'])} after {fit['seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
