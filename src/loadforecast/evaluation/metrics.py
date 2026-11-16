"""Forecast metrics: point, quantile, sample-based and multivariate scores.

Conventions: ``y`` is the realised value, errors are forecast minus actual, and every score
is negatively oriented (smaller is better) unless stated. Functions return per-observation
values where that is useful, so callers can average them by hour, by day or overall.
"""

from __future__ import annotations

import math

import numpy as np

SQRT_PI = math.sqrt(math.pi)


# --- point forecasts ---------------------------------------------------------------------


def mae(y, pred) -> float:
    return float(np.nanmean(np.abs(np.asarray(pred) - np.asarray(y))))


def rmse(y, pred) -> float:
    return float(np.sqrt(np.nanmean((np.asarray(pred) - np.asarray(y)) ** 2)))


def mape(y, pred) -> float:
    """Mean absolute percentage error, in percent. Safe here: demand is far from zero."""
    y = np.asarray(y)
    return float(100 * np.nanmean(np.abs(np.asarray(pred) - y) / np.abs(y)))


def bias(y, pred) -> float:
    return float(np.nanmean(np.asarray(pred) - np.asarray(y)))


def skill_score(loss_model: float, loss_baseline: float) -> float:
    """1 - L_model / L_baseline: 1 is perfect, 0 matches the baseline, negative is worse."""
    return 1.0 - loss_model / loss_baseline


# --- quantile forecasts ------------------------------------------------------------------


def pinball_loss(y, q, tau: float) -> np.ndarray:
    """Pinball (quantile) loss of the tau-quantile forecast ``q``, per observation."""
    diff = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return np.where(diff >= 0, tau * diff, (tau - 1) * diff)


def crps_from_quantiles(y, quantile_forecasts, taus) -> np.ndarray:
    """CRPS approximated by twice the average pinball loss over the quantile levels.

    ``quantile_forecasts`` has shape (n, len(taus)). With few levels this is a coarse
    approximation, but it is the same for every model, so comparisons are fair.
    """
    q = np.asarray(quantile_forecasts, dtype=float)
    y = np.asarray(y, dtype=float)[:, None]
    taus = np.asarray(taus, dtype=float)[None, :]
    diff = y - q
    losses = np.where(diff >= 0, taus * diff, (taus - 1) * diff)
    return 2.0 * losses.mean(axis=1)


def interval_coverage(y, lower, upper) -> float:
    """Share of observations inside [lower, upper]."""
    y = np.asarray(y)
    ok = ~np.isnan(y)
    return float(np.mean((y[ok] >= np.asarray(lower)[ok]) & (y[ok] <= np.asarray(upper)[ok])))


def interval_width(lower, upper) -> float:
    """Mean width of the interval (sharpness)."""
    return float(np.nanmean(np.asarray(upper) - np.asarray(lower)))


def central_intervals(taus) -> list[tuple[float, float, float]]:
    """Pairs of levels that form central intervals: (nominal coverage, lower tau, upper tau)."""
    taus = sorted(float(t) for t in taus)
    out = []
    for lo in taus:
        hi = round(1 - lo, 10)
        if lo < 0.5 and any(abs(hi - t) < 1e-9 for t in taus):
            out.append((round(hi - lo, 10), lo, hi))
    return sorted(out)


def pit_from_quantiles(y, quantile_forecasts, taus) -> np.ndarray:
    """Probability integral transform F(y), by linear interpolation between the quantiles.

    Values below the lowest or above the highest quantile are clipped to that level, so the
    histogram is only informative between the extreme levels.
    """
    q = np.asarray(quantile_forecasts, dtype=float)
    y = np.asarray(y, dtype=float)
    taus = np.asarray(taus, dtype=float)
    return np.array([np.interp(yi, qi, taus) for yi, qi in zip(y, q, strict=True)])


# --- sample-based (ensemble) forecasts ---------------------------------------------------


def crps_samples(y, samples) -> np.ndarray:
    """CRPS estimated from predictive samples, per observation.

    ``samples`` has shape (n, M). Uses the sorted form of
    mean|x - y| - 0.5 * mean|x - x'|, which costs O(M log M) instead of O(M^2).
    """
    x = np.sort(np.asarray(samples, dtype=float), axis=1)
    y = np.asarray(y, dtype=float)[:, None]
    m = x.shape[1]
    term1 = np.abs(x - y).mean(axis=1)
    weights = 2 * np.arange(1, m + 1) - m - 1
    term2 = (x * weights).sum(axis=1) / (m * m)
    return term1 - term2


def crps_gaussian(y, mu, sigma) -> np.ndarray:
    """Closed-form CRPS of a normal predictive distribution."""
    y, mu, sigma = (np.asarray(a, dtype=float) for a in (y, mu, sigma))
    z = (y - mu) / sigma
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / SQRT_PI)


# --- multivariate (whole-day) forecasts --------------------------------------------------


def energy_score(y, samples) -> float:
    """Energy score of one multivariate observation ``y`` (d,) given samples (M, d)."""
    x = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float)
    term1 = np.linalg.norm(x - y, axis=1).mean()
    gram = x @ x.T
    sq = np.diag(gram)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * gram, 0.0))
    return float(term1 - 0.5 * dist.mean())


def variogram_score(y, samples, p: float = 0.5, weights=None) -> float:
    """Variogram score of order ``p`` for one observation ``y`` (d,) given samples (M, d)."""
    x = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float)
    obs = np.abs(y[:, None] - y[None, :]) ** p
    exp = (np.abs(x[:, :, None] - x[:, None, :]) ** p).mean(axis=0)
    w = np.ones_like(obs) if weights is None else np.asarray(weights, dtype=float)
    return float((w * (obs - exp) ** 2).sum())
