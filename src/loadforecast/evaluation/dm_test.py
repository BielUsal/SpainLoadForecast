"""Diebold-Mariano test and block bootstrap for comparing two forecasts."""

from __future__ import annotations

import math

import numpy as np


def long_run_variance(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (Bartlett kernel) estimate of the long-run variance of a series."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if lags is None:
        lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
    d = x - x.mean()
    lrv = (d @ d) / n
    for k in range(1, lags + 1):
        gamma = (d[k:] @ d[:-k]) / n
        lrv += 2 * (1 - k / (lags + 1)) * gamma
    return float(lrv)


def diebold_mariano(loss_a, loss_b, lags: int | None = None) -> dict:
    """Test of equal expected loss from per-period losses (here: one value per day).

    Returns the mean differential (A minus B, negative means A is better), the DM statistic
    and the two-sided p-value from the normal approximation.
    """
    delta = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    delta = delta[~np.isnan(delta)]
    n = len(delta)
    lrv = long_run_variance(delta, lags)
    if lrv <= 0:
        return {"mean_diff": float(delta.mean()), "stat": 0.0, "p_value": 1.0, "n": n}
    stat = delta.mean() / math.sqrt(lrv / n)
    p_value = math.erfc(abs(stat) / math.sqrt(2))
    return {"mean_diff": float(delta.mean()), "stat": float(stat), "p_value": p_value, "n": n}


def block_bootstrap_ci(
    loss_a, loss_b, block: int = 7, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Moving-block bootstrap confidence interval for the mean loss differential (A minus B)."""
    delta = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    delta = delta[~np.isnan(delta)]
    n = len(delta)
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = delta[idx].mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))
