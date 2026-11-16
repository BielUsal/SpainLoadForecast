"""Bayesian structural model of day-ahead demand (NumPyro).

The model works on a regular grid of local days by 24 hours, in GW.

Mean of hour h on day d::

    m[d, h] = weekday profile (blended towards Sunday on holidays)
            + profiles for holidays, bridge days, Christmas, August and Holy Week
            + yearly cycle, and its interaction with the shape of the day
            + trend: piecewise-linear level, plus a linear drift of the shape of the day
            + solar radiation effect (drifting in time)
            + heating and cooling: smooth hinges on an exponentially smoothed temperature

Day-ahead error model. The forecast for day d is issued at 10:00 on day d-1, so the model
conditions on the residuals r = y - m that are known by then::

    y[d, :] ~ MultivariateStudentT(nu, m[d, :] + c[d, :], Sigma)
    c[d, h] = k1[h] * mean(r[d-1, 0:10])     this morning
            + k2[h] * mean(r[d-2, :])        level of the last complete day
            + k3[h] * r[d-1, h]              same hour this morning (h < 10 only)
            + sum over L in (2, 7, 14, 21) of kL[h] * r[d-L, h]   same hour, L days back
    Sigma = diag(s) [a 11' + (1 - a) AR1(rho)] diag(s)

This is a direct multi-step error model: it is fitted on exactly the conditional
distribution that the forecast needs, 14 to 38 hours ahead, which a one-step AR likelihood
is not (DESIGN section 7.3).

The two parts are fitted in two stages (``structure_model``, then ``error_model`` on its
residuals). Fitted jointly, the correction absorbs the level of demand, because day-to-day
errors are very persistent, and the structural parameters stop meaning anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

Y_CENTRE = 27.0  # GW, subtracted from demand so intercepts are near zero
N_LAGS = 168  # hours of temperature history in the smoothing kernel
ISSUE_HOUR = 10
SAME_HOUR_LAGS = (2, 7, 14, 21)  # days back whose same-hour residual enters the correction
CARRY_DAYS = max(SAME_HOUR_LAGS)  # the day-ahead correction looks back this many days
N_CARRY = 3 + len(SAME_HOUR_LAGS)
K_DAY, K_EXTRA, K_SHAPE, K_YEAR = 5, 2, 3, 3
KNOTS = np.arange(0.5, 7.0, 0.5)  # trend knots, years since TIME_ORIGIN
TIME_ORIGIN = pd.Timestamp("2021-01-01")
EXTRAS = ["holiday_intensity", "bridge_intensity", "christmas", "august", "holy_week"]
WARMUP_DAYS = N_LAGS // 24  # first days of a grid have incomplete temperature lags
HOURS = np.arange(24)


def fourier(x: np.ndarray, period: float, order: int, intercept: bool) -> np.ndarray:
    cols = [np.ones_like(x, dtype=float)] if intercept else []
    for k in range(1, order + 1):
        cols += [np.cos(2 * np.pi * k * x / period), np.sin(2 * np.pi * k * x / period)]
    return np.column_stack(cols)


H_DAY = fourier(HOURS, 24, K_DAY, True)  # (24, 11) weekday profiles
H_EXTRA = fourier(
    HOURS, 24, K_EXTRA, True
)  # (24, 5) special-day profiles, hour-varying coefficients
H_SHAPE = fourier(HOURS, 24, K_SHAPE, False)  # (24, 6) shape-only terms (no level)
H_SEASON = fourier(
    HOURS, 24, K_DAY, False
)  # (24, 10) how the shape of the day moves with the season
H_SCALE = fourier(HOURS, 24, 3, True)  # (24, 7) log error scale


@dataclass
class Grid:
    """Model inputs for a run of consecutive local days."""

    dates: pd.DatetimeIndex
    data: dict[str, np.ndarray]

    @property
    def n_days(self) -> int:
        return len(self.dates)


def build_grid(frame: pd.DataFrame) -> Grid:
    """Turn hourly rows into the day-by-hour arrays the model needs.

    ``frame`` needs ``local_date``, ``local_hour``, ``demand_mw`` (NaN where unknown),
    ``temp``, ``rad`` and the calendar columns, for consecutive local days. On clock-change
    days the missing local hour is interpolated and the repeated one averaged.
    """

    def grid_of(column: str, fill: bool) -> np.ndarray:
        wide = frame.pivot_table(
            index="local_date", columns="local_hour", values=column, aggfunc="mean", dropna=False
        ).reindex(columns=range(24))
        if fill:
            wide = wide.interpolate(axis=1, limit_direction="both")
        return wide.to_numpy(dtype="float32")

    days = frame.groupby("local_date").first()
    dates = pd.DatetimeIndex(days.index)
    if not (np.diff(dates.values) == np.timedelta64(1, "D")).all():
        raise ValueError("the grid needs consecutive local days")
    n = len(dates)

    y = grid_of("demand_mw", fill=False) / 1000.0
    # Fill only the clock-change gap (one missing hour inside an otherwise complete day).
    one_missing = np.isnan(y).sum(axis=1) == 1
    y[one_missing] = (
        pd.DataFrame(y[one_missing]).interpolate(axis=1, limit_direction="both").to_numpy()
    )
    temp = grid_of("temp", fill=True)
    lag_index = np.maximum(np.arange(n * 24)[:, None] - np.arange(N_LAGS)[None, :], 0)
    temp_lags = np.nan_to_num(temp.ravel()[lag_index], nan=15.0)

    weekday = days["weekday"].to_numpy(dtype=int)
    holiday = days["holiday_intensity"].to_numpy(dtype="float32")
    blend = np.zeros((n, 7), dtype="float32")
    blend[np.arange(n), weekday] = 1.0 - holiday
    blend[:, 6] += holiday  # a holiday looks like a Sunday
    tau = ((dates - TIME_ORIGIN).days / 365.25).to_numpy(dtype="float32")
    day_of_year = dates.day_of_year.to_numpy(dtype=float)
    year_basis = fourier(day_of_year, 365.25, K_YEAR, False)

    data = {
        "y": y - Y_CENTRE,
        "temp_lags": temp_lags.astype("float32"),
        "rad": np.nan_to_num(grid_of("rad", fill=True), nan=0.0) / 1000.0,  # kW/m2
        "blend": blend,
        "extras": days[EXTRAS].to_numpy(dtype="float32"),
        "year_basis": year_basis.astype("float32"),
        "tau": tau,
        "trend_basis": np.maximum(tau[:, None] - KNOTS[None, :], 0.0).astype("float32"),
    }
    return Grid(dates, data)


def mean_fn(p: dict, d: dict) -> jnp.ndarray:
    """Mean demand (GW, centred) on the grid: shape (n_days, 24)."""
    n = d["blend"].shape[0]
    if "profile" in p:
        profiles = p["profile"]  # (7, 11)
    else:  # fits saved with the earlier non-centred parametrisation
        profiles = p["profile_mean"] + p["profile_sd"] * p["profile_z"]
    mean = d["blend"] @ profiles @ H_DAY.T
    mean += jnp.einsum("dj,jk,hk->dh", d["extras"], p["extra"], H_EXTRA)
    mean += (d["year_basis"] @ p["yearly"])[:, None]
    mean += d["year_basis"][:, :4] @ p["year_shape"] @ H_SEASON.T
    mean += (p["slope"] * d["tau"] + d["trend_basis"] @ p["slope_change"])[:, None]
    mean += d["tau"][:, None] * (H_SHAPE @ p["shape_drift"])[None, :]
    mean += d["rad"] * (p["rad"] + p["rad_drift"] * d["tau"])[:, None]

    decay = 0.5 ** (jnp.arange(N_LAGS) / p["half_life"])
    smooth = (d["temp_lags"] @ (decay / decay.sum())).reshape(n, 24)
    sharp = p["sharpness"]
    heat = jax.nn.softplus(sharp * (p["t_heat"] - smooth)) / sharp
    cool = jax.nn.softplus(sharp * (smooth - p["t_heat"] - p["comfort_width"])) / sharp
    mean += jnp.exp(H_EXTRA @ p["log_heat"])[None, :] * heat
    mean += jnp.exp(H_EXTRA @ p["log_cool"])[None, :] * cool
    return mean


def correction(p: dict, resid: jnp.ndarray) -> jnp.ndarray:
    """Day-ahead correction c[d, :] from residuals known at issue time.

    ``resid`` is (n_days, 24) with zeros where the residual is unknown. The first
    ``CARRY_DAYS`` rows are zero, because their predictors are outside the grid.
    """
    k = p["carry"] @ H_EXTRA.T  # (3 + number of same-hour lags, 24)
    n = resid.shape[0]

    def lagged(days: int) -> jnp.ndarray:
        return resid[CARRY_DAYS - days : n - days]

    issue_day = lagged(1)  # known before 10:00 only
    c = (
        issue_day[:, :ISSUE_HOUR].mean(axis=1, keepdims=True) * k[0]
        + lagged(2).mean(axis=1, keepdims=True) * k[1]
        + issue_day * (HOURS < ISSUE_HOUR) * k[2]
    )
    for i, days in enumerate(SAME_HOUR_LAGS):
        c += lagged(days) * k[3 + i]
    return jnp.concatenate([jnp.zeros((CARRY_DAYS, 24)), c], axis=0)


def error_scale_tril(p: dict) -> jnp.ndarray:
    """Cholesky factor of the within-day error covariance (GW)."""
    s = jnp.exp(H_SCALE @ p["log_scale"])
    lag = jnp.abs(HOURS[:, None] - HOURS[None, :])
    corr = p["common"] + (1.0 - p["common"]) * p["rho"] ** lag
    cov = s[:, None] * corr * s[None, :] + 1e-6 * jnp.eye(24)
    return jnp.linalg.cholesky(cov)


def sample_structure() -> dict:
    """Priors of the structural mean."""
    k_sd = jnp.concatenate([jnp.array([5.0]), jnp.repeat(3.0 / jnp.arange(1, K_DAY + 1), 2)])
    hinge_loc = jnp.array([jnp.log(0.6), 0, 0, 0, 0])
    hinge_sd = jnp.array([0.7, 0.5, 0.5, 0.5, 0.5])
    # Centred: the data pin down each weekday's profile, so sampling the profiles themselves
    # mixes far better than sampling standardised deviations from the common profile.
    profile_mean = numpyro.sample("profile_mean", dist.Normal(0.0, k_sd).to_event(1))
    profile_sd = numpyro.sample("profile_sd", dist.HalfNormal(jnp.ones(11)).to_event(1))
    return {
        "profile": numpyro.sample(
            "profile", dist.Normal(jnp.tile(profile_mean, (7, 1)), profile_sd).to_event(2)
        ),
        "extra": numpyro.sample("extra", dist.Normal(jnp.zeros((5, 5)), 2.0).to_event(2)),
        "yearly": numpyro.sample("yearly", dist.Normal(jnp.zeros(2 * K_YEAR), 2.0).to_event(1)),
        "year_shape": numpyro.sample(
            "year_shape", dist.Normal(jnp.zeros((4, 2 * K_DAY)), 1.0).to_event(2)
        ),
        "slope": numpyro.sample("slope", dist.Normal(0.0, 1.0)),
        "slope_change": numpyro.sample(
            "slope_change", dist.Laplace(jnp.zeros(len(KNOTS)), 0.3).to_event(1)
        ),
        "shape_drift": numpyro.sample(
            "shape_drift", dist.Normal(jnp.zeros(2 * K_SHAPE), 0.5).to_event(1)
        ),
        "rad": numpyro.sample("rad", dist.Normal(0.0, 3.0)),
        "rad_drift": numpyro.sample("rad_drift", dist.Normal(0.0, 1.0)),
        "half_life": numpyro.sample("half_life", dist.LogNormal(jnp.log(24.0), 0.7)),
        "sharpness": numpyro.sample("sharpness", dist.LogNormal(0.0, 0.5)),
        "t_heat": numpyro.sample("t_heat", dist.Normal(15.5, 2.0)),
        "comfort_width": numpyro.sample("comfort_width", dist.LogNormal(jnp.log(5.0), 0.4)),
        "log_heat": numpyro.sample("log_heat", dist.Normal(hinge_loc, hinge_sd).to_event(1)),
        "log_cool": numpyro.sample("log_cool", dist.Normal(hinge_loc, hinge_sd).to_event(1)),
    }


def sample_noise() -> dict:
    """Priors of a within-day error distribution (used by both stages)."""
    scale_loc = jnp.array([jnp.log(0.5), 0, 0, 0, 0, 0, 0])
    scale_sd = jnp.array([1.0] + [0.5] * 6)
    return {
        "log_scale": numpyro.sample("log_scale", dist.Normal(scale_loc, scale_sd).to_event(1)),
        "common": numpyro.sample("common", dist.Beta(2.0, 2.0)),
        "rho": numpyro.sample("rho", dist.Beta(5.0, 2.0)),
        "dof": numpyro.sample("dof", dist.Gamma(2.0, 0.1)),
    }


def structure_model(d: dict, weight: jnp.ndarray) -> None:
    """Stage 1: the structural mean, with errors correlated within a day.

    ``weight`` (n_days,) is 1 for days that enter the likelihood. Errors on different days
    are treated as independent here, which they are not: credible intervals of the
    structural parameters are therefore too narrow.
    """
    p = sample_structure() | sample_noise()
    y = d["y"]
    likelihood = dist.MultivariateStudentT(2.0 + p["dof"], mean_fn(p, d), error_scale_tril(p))
    with numpyro.plate("days", y.shape[0]):
        numpyro.sample("obs", likelihood.mask(weight > 0), obs=jnp.nan_to_num(y))


def error_model(resid: jnp.ndarray, weight: jnp.ndarray) -> None:
    """Stage 2: the day-ahead distribution of the residuals given those known at issue time.

    ``resid`` is (n_days, 24) with zeros where unknown; ``weight`` marks the target days.
    """
    carry_loc = jnp.tile(jnp.array([0.2, 0, 0, 0, 0]), (N_CARRY, 1))
    carry_sd = jnp.tile(jnp.array([0.5, 0.3, 0.3, 0.3, 0.3]), (N_CARRY, 1))
    p = {"carry": numpyro.sample("carry", dist.Normal(carry_loc, carry_sd).to_event(2))}
    p |= sample_noise()
    likelihood = dist.MultivariateStudentT(
        2.0 + p["dof"], correction(p, resid), error_scale_tril(p)
    )
    with numpyro.plate("days", resid.shape[0]):
        numpyro.sample("obs", likelihood.mask(weight > 0), obs=resid)


def default_weight(d: dict) -> np.ndarray:
    """Days that can be targets: fully observed, past the lag warm-up."""
    complete = ~np.isnan(np.asarray(d["y"])).any(axis=1)
    complete[: WARMUP_DAYS + CARRY_DAYS] = False
    return complete.astype("float32")


def residuals(structure: dict, grid_data: dict) -> jnp.ndarray:
    """y - mean on the grid for one set of structural parameters; zero where y is unknown."""
    y = grid_data["y"]
    return jnp.where(jnp.isnan(y), 0.0, y - mean_fn(structure, grid_data))


@jax.jit
def _predictive_parts(structure: dict, errors: dict, d: dict):
    """For one posterior draw of each stage: location of the last grid day, scale factor, dof."""
    mean = mean_fn(structure, d)
    resid = jnp.where(jnp.isnan(d["y"]), 0.0, d["y"] - mean)
    loc = mean + correction(errors, resid)
    return loc[-1], error_scale_tril(errors), 2.0 + errors["dof"], mean[-1]


_predictive_batch = jax.jit(jax.vmap(_predictive_parts, in_axes=(0, 0, None)))


def predict_last_day(
    structure: dict,
    errors: dict,
    grid: Grid,
    rng: np.random.Generator,
    draws_per_sample: int = 4,
) -> dict[str, np.ndarray]:
    """Posterior predictive for the last day of ``grid`` (the target day).

    ``structure`` and ``errors`` hold the same number of posterior draws. Returns
    ``samples`` (n, 24) in MW, and the structural mean per draw in MW.
    """
    data = {k: jnp.asarray(v) for k, v in grid.data.items()}
    loc, tril, dof, mean = (np.asarray(a) for a in _predictive_batch(structure, errors, data))
    n = loc.shape[0]
    z = rng.standard_normal((n, draws_per_sample, 24))
    chi2 = rng.chisquare(dof[:, None], size=(n, draws_per_sample))
    scale = np.sqrt(dof[:, None] / chi2)[:, :, None]
    draws = loc[:, None, :] + scale * np.einsum("nij,nkj->nki", tril, z)
    return {
        "samples": (draws.reshape(-1, 24) + Y_CENTRE) * 1000.0,
        "structural_mean": (mean + Y_CENTRE) * 1000.0,
    }


# --- inference ---------------------------------------------------------------------------
# Each fitting function takes a model and its arguments, so both stages share them.


def fit_map(
    target, args: tuple, steps: int = 4000, seed: int = 0, init: dict | None = None
) -> dict:
    """Posterior mode by gradient ascent."""
    from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_median, init_to_value

    init_fn = init_to_median if init is None else init_to_value(values=init)
    guide = autoguide.AutoDelta(target, init_loc_fn=init_fn)
    schedule = numpyro.optim.ClippedAdam(lambda i: 0.02 * 0.05 ** (i / steps))
    result = SVI(target, guide, schedule, Trace_ELBO()).run(
        jax.random.PRNGKey(seed), steps, *args, progress_bar=False
    )
    return guide.median(result.params)


def fit_svi(
    target,
    args: tuple,
    init: dict,
    steps: int = 4000,
    rank: int = 20,
    seed: int = 0,
    n_samples: int = 300,
) -> dict:
    """Low-rank Gaussian approximation of the posterior, started at ``init`` (a MAP estimate).

    Returns posterior draws as a dict of arrays with leading dimension ``n_samples``.
    """
    from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value

    guide = autoguide.AutoLowRankMultivariateNormal(
        target, rank=rank, init_loc_fn=init_to_value(values=init), init_scale=0.01
    )
    schedule = numpyro.optim.ClippedAdam(lambda i: 0.005 * 0.1 ** (i / steps))
    result = SVI(target, guide, schedule, Trace_ELBO()).run(
        jax.random.PRNGKey(seed), steps, *args, progress_bar=False
    )
    key = jax.random.PRNGKey(seed + 1)
    return guide.sample_posterior(key, result.params, sample_shape=(n_samples,))


def fit_nuts(
    target,
    args: tuple,
    init: dict,
    warmup: int = 500,
    samples: int = 500,
    chains: int = 4,
    seed: int = 0,
    max_tree_depth: int = 8,
    dense_mass: bool = False,
):
    """NUTS started at ``init``. Returns the fitted ``MCMC`` object."""
    from numpyro.infer import MCMC, NUTS, init_to_value

    kernel = NUTS(
        target,
        init_strategy=init_to_value(values=init),
        max_tree_depth=max_tree_depth,
        dense_mass=dense_mass,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        chain_method="parallel",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), *args, extra_fields=("diverging", "num_steps"))
    return mcmc


def stage_args(grid: Grid) -> tuple[dict, jnp.ndarray]:
    """Arguments of ``structure_model`` for a grid."""
    return {k: jnp.asarray(v) for k, v in grid.data.items()}, jnp.asarray(default_weight(grid.data))
