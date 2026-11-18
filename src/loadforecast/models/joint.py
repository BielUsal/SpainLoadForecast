"""Joint forecasts of the 24 hours of a day, built on the RNN's forecasts. DESIGN section 7.5.

The RNN gives a calibrated distribution for each hour on its own (seven quantiles). A planner
who needs whole-day scenarios also needs the dependence between hours: if 18:00 comes in high,
19:00 will too. The models here turn one day's RNN forecast into scenarios of the 24 hours:

- ``independent``: each hour drawn from its RNN marginal on its own. The reference that a
  marginal-only forecast amounts to.
- ``copula``: the same marginals tied together by a Gaussian copula, with the correlation of the
  hours' normal scores over the past year.
- ``flow``: a conditional neural spline flow (``zuko``) over the day's 24-vector of errors,
  conditioned on the calendar, the season, the temperature and the RNN's own range widths.

All of them learn only from the RNN's out-of-sample errors on days up to D-2 for target day D,
as the residual quantiles do (``models/quantiles.py``).

The marginals are continuous versions of the seven quantiles: piecewise linear in normal-score
space (the quantile levels mapped through the inverse normal CDF), extended linearly beyond the
5% and 95% quantiles, so the tails are Gaussian-like.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
from scipy.stats import norm

from loadforecast.config import load_config
from loadforecast.models.quantiles import quantile_column

FLAGS = ["holiday_intensity", "bridge_intensity", "christmas", "august", "holy_week"]
TEMP_CENTRE, TEMP_SCALE = 15.0, 8.0
WIDTH_REF = 2000.0  # MW, a typical raw 5-95% width of the RNN
Z90 = 2 * norm.ppf(0.95)  # width of a standard normal's central 90% interval


def load_model_config() -> dict:
    return load_config("models/joint")


# --- day arrays -------------------------------------------------------------------------


@dataclass
class Days:
    """One row per local day with 24 hours. ``y`` is NaN where demand is unknown or masked."""

    dates: pd.DatetimeIndex
    y: np.ndarray  # (D, 24)
    point: np.ndarray  # (D, 24)
    q: np.ndarray  # (D, 24, K) the quantiles that define the marginals
    width: np.ndarray  # (D, 24) the model's own 5-95% width, before recalibration
    ctx: np.ndarray  # (D, C)
    source: np.ndarray  # (D,) 1 if the forecast used forecast weather, 0 if observed

    def __len__(self) -> int:
        return len(self.dates)

    def take(self, rows) -> Days:
        return Days(**{f.name: getattr(self, f.name)[rows] for f in fields(self)})

    @property
    def complete(self) -> np.ndarray:
        return ~np.isnan(self.y).any(axis=1)


def concat_days(parts: list[Days]) -> Days:
    parts = [p for p in parts if len(p)]
    out = {}
    for f in fields(Days):
        values = [getattr(p, f.name) for p in parts]
        out[f.name] = values[0].append(values[1:]) if f.name == "dates" else np.concatenate(values)
    return Days(**out)


def build_days(pred: pd.DataFrame, table: pd.DataFrame, taus: list[float], weather: str) -> Days:
    """Reshape hourly predictions to days. Days without exactly 24 hours (clock changes) are
    left out."""
    temp_col = "temp_fc2" if weather == "forecast" else "temp_obs"
    frame = pred.join(table[["demand_mw", "is_masked", "weekday", temp_col, *FLAGS]])
    hours = frame.groupby("local_date")["local_hour"].agg(["size", "nunique"])
    keep = hours.index[(hours["size"] == 24) & (hours["nunique"] == 24)]
    frame = frame[frame["local_date"].isin(keep)].sort_values(["local_date", "local_hour"])
    n = len(keep)

    def grid(values) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(n, 24)

    y = grid(frame["demand_mw"].where(~frame["is_masked"].astype(bool)))
    columns = [quantile_column(t) for t in taus]
    q = np.asarray(frame[columns], dtype=float).reshape(n, 24, len(taus))
    lo, hi = columns[0], columns[-1]
    if f"model_{lo}" in frame:
        width = grid(frame[f"model_{hi}"] - frame[f"model_{lo}"])
    else:
        width = grid(frame[hi] - frame[lo])
    first = frame.groupby("local_date").first()
    flags = frame.groupby("local_date")[FLAGS].max().to_numpy(dtype=float)
    doy = 2 * np.pi * pd.DatetimeIndex(keep).day_of_year.to_numpy() / 365.25
    temp = (grid(frame[temp_col]).mean(axis=1) - TEMP_CENTRE) / TEMP_SCALE
    log_width = np.log(width.mean(axis=1) / WIDTH_REF)
    source = np.full(n, 1.0 if weather == "forecast" else 0.0)
    ctx = np.column_stack(
        [
            np.eye(7)[first["weekday"].to_numpy(dtype=int)],
            flags,
            np.sin(doy),
            np.cos(doy),
            temp,
            log_width,
            source,
        ]
    )
    return Days(
        dates=pd.DatetimeIndex(keep),
        y=y,
        point=grid(frame["point"]),
        q=q,
        width=width,
        ctx=ctx.astype(np.float32),
        source=source,
    )


def prefer_forecast(forecast: Days, oracle: Days) -> Days:
    """All days, taken from the forecast-weather run where it exists and from the observed-
    weather run otherwise (forecast radiation only exists from 2024)."""
    extra = oracle.take(~oracle.dates.isin(forecast.dates))
    both = concat_days([forecast, extra])
    return both.take(np.argsort(both.dates.to_numpy()))


# --- marginals ----------------------------------------------------------------------------


class Marginals:
    """Continuous marginals from K quantiles: piecewise linear in normal-score space."""

    def __init__(self, taus: list[float]):
        self.knots = norm.ppf(np.asarray(taus))

    def values(self, q: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Demand at normal scores ``scores`` (..., M) given quantiles ``q`` (..., K)."""
        k = self.knots
        idx = np.clip(np.searchsorted(k, scores), 1, len(k) - 1)
        lo = np.take_along_axis(q, idx - 1, axis=-1)
        hi = np.take_along_axis(q, idx, axis=-1)
        slope = (hi - lo) / (k[idx] - k[idx - 1])
        return lo + slope * (scores - k[idx - 1])

    def scores(self, q: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Normal score of each value ``y`` (...) under quantiles ``q`` (..., K)."""
        k = self.knots
        idx = 1 + (y[..., None] > q[..., 1:-1]).sum(axis=-1)
        lo = np.take_along_axis(q, (idx - 1)[..., None], axis=-1)[..., 0]
        hi = np.take_along_axis(q, idx[..., None], axis=-1)[..., 0]
        slope = (k[idx] - k[idx - 1]) / np.maximum(hi - lo, 1e-6)
        return k[idx - 1] + (y - lo) * slope


# --- joint models -------------------------------------------------------------------------


class Independent:
    name = "independent"

    def __init__(self, taus: list[float], cfg: dict):
        self.marginals = Marginals(taus)

    def fit(self, train: Days) -> None:
        pass

    def sample(self, day: Days, m: int, rng: np.random.Generator) -> np.ndarray:
        """(m, 24) scenarios for a one-day ``Days``."""
        scores = rng.standard_normal((24, m))
        return self.marginals.values(day.q[0], scores).T


class GaussianCopula(Independent):
    name = "copula"

    def __init__(self, taus: list[float], cfg: dict):
        super().__init__(taus, cfg)
        self.window = cfg["copula"]["window_days"]
        self.chol = np.eye(24)

    def fit(self, train: Days) -> None:
        recent = train.take(train.dates > train.dates[-1] - pd.Timedelta(days=self.window))
        s = self.marginals.scores(recent.q, recent.y)
        corr = np.corrcoef(s.T)
        values, vectors = np.linalg.eigh(corr)
        corr = (vectors * np.maximum(values, 1e-6)) @ vectors.T
        d = np.sqrt(np.diag(corr))
        self.corr = corr / d[:, None] / d[None, :]
        self.chol = np.linalg.cholesky(self.corr)

    def sample(self, day: Days, m: int, rng: np.random.Generator) -> np.ndarray:
        scores = self.chol @ rng.standard_normal((24, m))
        return self.marginals.values(day.q[0], scores).T


class ConditionalFlow(Independent):
    """Neural spline flow over the day's 24-vector, conditioned on ``Days.ctx``.

    ``target: residual`` models ``Z90 * (y - point) / width``: the error in units that would be
    standard normal if the model's raw 90% range were exactly right. The flow then sets both the
    marginals and the dependence. ``target: score`` models the normal scores of the RNN's
    (recalibrated) marginals, so the flow mainly changes the dependence and the samples are
    mapped back through the RNN's quantiles.

    With ``inflate: true`` the scenarios are widened (or narrowed) around each hour's median by
    one factor, chosen after training so that the flow's central 90% ranges cover 90% of the
    hours of the held-out days (the most recent ``holdout`` share, not trained on; used only
    for early stopping). The dependence the flow learned is kept.
    """

    name = "flow"

    def __init__(self, taus: list[float], cfg: dict, device: str | None = None):
        import torch

        super().__init__(taus, cfg)
        self.cfg = cfg["flow"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.flows: list = []

    def targets(self, days: Days) -> np.ndarray:
        if self.cfg["target"] == "residual":
            return Z90 * (days.y - days.point) / days.width
        if self.cfg["target"] == "score":
            return self.marginals.scores(days.q, days.y)
        raise ValueError(f"unknown flow target: {self.cfg['target']}")

    def fit(self, train: Days) -> None:
        import torch
        import zuko

        cfg = self.cfg
        t = np.clip(self.targets(train), -6.0, 6.0)  # the splines live on [-5, 5]
        self.ctx_mean = train.ctx.mean(axis=0)
        self.ctx_std = np.maximum(train.ctx.std(axis=0), 1e-3)
        c = (train.ctx - self.ctx_mean) / self.ctx_std
        n_hold = max(int(len(t) * cfg["holdout"]), 1)
        to = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)  # noqa: E731
        t_fit, c_fit, t_hold, c_hold = (
            to(t[:-n_hold]),
            to(c[:-n_hold]),
            to(t[-n_hold:]),
            to(c[-n_hold:]),
        )
        self.flows = []
        for seed in range(cfg["seeds"]):
            torch.manual_seed(seed)
            flow = zuko.flows.NSF(
                features=24,
                context=c.shape[1],
                transforms=cfg["transforms"],
                hidden_features=cfg["hidden"],
                bins=cfg["bins"],
            ).to(self.device)
            opt = torch.optim.AdamW(flow.parameters(), cfg["lr"], weight_decay=cfg["weight_decay"])
            best, best_state, waited = np.inf, None, 0
            gen = torch.Generator(device="cpu").manual_seed(seed)
            for _ in range(cfg["max_epochs"]):
                flow.train()
                order = torch.randperm(len(t_fit), generator=gen).to(self.device)
                for start in range(0, len(order), cfg["batch_size"]):
                    rows = order[start : start + cfg["batch_size"]]
                    loss = -flow(c_fit[rows]).log_prob(t_fit[rows]).mean()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
                    opt.step()
                flow.eval()
                with torch.no_grad():
                    hold = -flow(c_hold).log_prob(t_hold).mean().item()
                if hold < best - 1e-4:
                    best, waited = hold, 0
                    best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
                else:
                    waited += 1
                    if waited >= cfg["patience"]:
                        break
            flow.load_state_dict(best_state)
            self.flows.append(flow.eval())
        self.holdout_nll = best
        self.inflation = 1.0
        if cfg.get("inflate", False):
            chunks = [c_hold[i : i + 16] for i in range(0, len(c_hold), 16)]  # GPU memory
            sims = np.concatenate([self._sample_targets(ch, 400) for ch in chunks], axis=1)
            med = np.median(sims, axis=0)
            lo, hi = np.quantile(sims, [0.05, 0.95], axis=0)
            truth = self.targets(train.take(np.arange(len(t) - n_hold, len(t))))
            self.inflation = float(np.quantile(np.abs(truth - med) / ((hi - lo) / 2), 0.9))

    def _sample_targets(self, c, m: int) -> np.ndarray:
        """(m, ...) samples in target units, equally from each seed's flow, for context ``c``."""
        import torch

        per_flow = -(-m // len(self.flows))
        with torch.no_grad():
            t = torch.cat([f(c).sample((per_flow,)) for f in self.flows])[:m]
        return t.cpu().numpy().astype(float)

    def sample(self, day: Days, m: int, rng: np.random.Generator) -> np.ndarray:
        import torch

        c = torch.tensor(
            (day.ctx - self.ctx_mean) / self.ctx_std, dtype=torch.float32, device=self.device
        )
        torch.manual_seed(int(rng.integers(2**31)))
        t = self._sample_targets(c[0], m)
        t = t[rng.permutation(len(t))]  # mixes the seeds
        if self.inflation != 1.0:
            med = np.median(t, axis=0)
            t = med + self.inflation * (t - med)
        if self.cfg["target"] == "residual":
            return day.point[0] + t * day.width[0] / Z90
        return self.marginals.values(day.q[0], t.T.astype(float)).T


def joint_models(taus: list[float], cfg: dict | None = None) -> list[Independent]:
    cfg = cfg or load_model_config()
    return [Independent(taus, cfg), GaussianCopula(taus, cfg), ConditionalFlow(taus, cfg)]
