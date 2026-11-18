"""The RNN of DESIGN 7.4 with a flow head trained with it: end-to-end scenarios (DESIGN 7.5).

The network is the GRU encoder-decoder of ``models/rnn.py`` with its quantile head, plus a
conditional neural spline flow (``zuko``) over the 24 hours of the target window,

    r = Z90 * (y - m) / w,

with ``m`` and ``w`` the quantile head's own median and 5-95% width (detached). The flow's
context is a learned projection of the decoder's outputs over those 24 hours, so its
likelihood also trains the encoder and the decoder. Loss: pinball + ``weight`` times the flow's
negative log-likelihood per hour.

As for the RNN every hour of the history is a training origin. The flow window is hours
``WINDOW_START`` to ``WINDOW_START + 23`` after the origin, which is the target day when the
origin is the 10:00 issue time.

Predictions are the RNN's (quantiles, recalibrated by the backtest). The scenarios are kept on
the model and written by ``save_scenarios``, as offsets from the raw point forecast; turning
them into scored scenarios (and calibrating them) is done in ``evaluation/run_joint.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from loadforecast.config import load_config
from loadforecast.models.quantiles import quantile_column
from loadforecast.models.rnn import (
    WEEK,
    Y_CENTRE,
    Y_SCALE,
    RNNForecaster,
    Seq2Quantiles,
    covariates,
    fill_gaps,
    pinball,
    scale_demand,
    training_arrays,
    valid_origins,
)

WINDOW_START = 14  # hours from a 10:00 origin to the start of the next day
Z90 = 3.2897072539029435  # width of a standard normal's central 90% interval
CLIP = 6.0  # the splines live on [-5, 5]; beyond they are the identity


def load_model_config() -> dict:
    return load_config("models/rnn_flow")


class Seq2Flow(Seq2Quantiles):
    def __init__(self, n_taus: int, cfg: dict):
        import zuko

        super().__init__(n_taus, cfg["hidden"], cfg["layers"], cfg["dropout"])
        f = cfg["flow"]
        self.project = nn.Sequential(nn.Linear(24 * cfg["hidden"], f["context"]), nn.Tanh())
        self.flow = zuko.flows.NSF(
            features=24,
            context=f["context"],
            transforms=f["transforms"],
            hidden_features=f["hidden"],
            bins=f["bins"],
        )

    def window(self, q: torch.Tensor, out: torch.Tensor, start: int):
        """Context, median and 5-95% width of the 24-hour window that begins at ``start``."""
        win = slice(start, start + 24)
        ctx = self.project(self.dropout(out[:, win]).flatten(1))
        mid = q[:, win, self.median].detach()
        width = (q[:, win, -1] - q[:, win, 0]).detach().clamp_min(1e-3)
        return ctx, mid, width


def train_network(
    y: np.ndarray, x: np.ndarray, taus: list[float], cfg: dict, seed: int, device: str
) -> Seq2Flow:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, dec = cfg["encoder_hours"], cfg["decoder_hours"]
    assert dec >= WINDOW_START + 24
    origins = valid_origins(y, enc, dec)
    y_t = torch.tensor(y, device=device)
    x_t = torch.tensor(x, device=device)
    tau_t = torch.tensor(taus, device=device)
    past_offsets = torch.arange(-enc, 0, device=device)
    future_offsets = torch.arange(0, dec, device=device)
    weight = cfg["flow"]["weight"]

    model = Seq2Flow(len(taus), cfg).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = int(np.ceil(len(origins) / cfg["batch_size"]))
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch
    )
    model.train()
    for _ in range(cfg["epochs"]):
        order = rng.permutation(origins)
        for start in range(0, len(order), cfg["batch_size"]):
            origin = torch.tensor(order[start : start + cfg["batch_size"]], device=device)
            past_rows = origin[:, None] + past_offsets
            future_rows = origin[:, None] + future_offsets
            past = torch.cat([x_t[past_rows], y_t[past_rows][..., None]], dim=-1)
            future = torch.cat([x_t[future_rows], y_t[future_rows - WEEK][..., None]], dim=-1)
            target = y_t[future_rows]
            out = model.decode(past, future)
            q = model.quantiles(out, future)
            loss = pinball(q, target, tau_t)
            ctx, mid, width = model.window(q, out, WINDOW_START)
            y_win = target[:, WINDOW_START : WINDOW_START + 24]
            known = ~torch.isnan(y_win).any(dim=1)
            if known.any():
                r = (Z90 * (y_win[known] - mid[known]) / width[known]).clamp(-CLIP, CLIP)
                nll = -model.flow(ctx[known]).log_prob(r).mean() / 24
                loss = loss + weight * nll
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_norm"])
            optimiser.step()
            schedule.step()
    return model.eval()


class RNNFlowForecaster(RNNForecaster):
    _fitted: dict[str, list[Seq2Flow]] = {}

    def __init__(self, weather: str, taus: list[float], cfg: dict | None = None):
        super().__init__(weather, taus, cfg or load_model_config())
        self.name = f"rnn_flow_{weather}"
        self.n_samples = self.cfg["flow"]["n_samples"]
        self.scenarios: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, history: pd.DataFrame) -> None:
        cutoff = history.index[-1]
        key = json.dumps([str(cutoff), self.taus, self.cfg], sort_keys=True, default=str)
        if key not in self._fitted:
            recent = history[history["local_date"] >= self.cfg["train_start"]]
            y, x = training_arrays(recent)
            self._fitted[key] = [
                train_network(y, x, self.taus, self.cfg, seed, self.device)
                for seed in range(self.cfg["seeds"])
            ]
        self.networks = self._fitted[key]

    @torch.no_grad()
    def predict(
        self, history: pd.DataFrame, future: pd.DataFrame, gap: pd.DataFrame
    ) -> pd.DataFrame:
        enc = self.cfg["encoder_hours"]
        tail = history.iloc[-(enc + WEEK) :]
        demand = scale_demand(fill_gaps(tail["demand_mw"]))
        past_x = covariates(tail, tail["temp_obs"].to_numpy(), tail["rad_obs"].to_numpy())
        past = np.column_stack([past_x, demand])[-enc:]
        ahead = pd.concat([gap, future])
        temp = np.concatenate([gap[self.gap_src["temp"]], future[self.target_src["temp"]]])
        rad = np.concatenate([gap[self.gap_src["rad"]], future[self.target_src["rad"]]])
        week_before = demand[len(demand) - WEEK : len(demand) - WEEK + len(ahead)]
        ahead_x = np.column_stack([covariates(ahead, temp, rad), week_before])
        past_t = torch.tensor(past[None], dtype=torch.float32, device=self.device)
        ahead_t = torch.tensor(ahead_x[None], dtype=torch.float32, device=self.device)

        quantiles, samples = [], []
        per_net = -(-self.n_samples // len(self.networks))
        start = len(gap)
        for net in self.networks:
            out = net.decode(past_t, ahead_t)
            q = net.quantiles(out, ahead_t)
            quantiles.append(q[0, -len(future) :])
            if len(future) == 24:  # clock-change days have no 24-hour scenarios
                ctx, mid, width = net.window(q, out, start)
                r = net.flow(ctx[0]).sample((per_net,))
                samples.append(mid[0] + r * width[0] / Z90)
        by_seed = torch.stack(quantiles)
        by_seed = (by_seed * Y_SCALE + Y_CENTRE).cpu().numpy()  # (seeds, hours, taus)
        mean = by_seed.mean(axis=0)
        point = mean[:, len(self.taus) // 2]
        pred = pd.DataFrame(index=future.index)
        pred["point"] = point
        for j, tau in enumerate(self.taus):
            pred[quantile_column(tau)] = mean[:, j]
        if samples:
            x = (torch.cat(samples)[: self.n_samples] * Y_SCALE + Y_CENTRE).cpu().numpy()
            order = np.argsort(future["local_hour"].to_numpy())
            date = pd.Timestamp(future["local_date"].iloc[0])
            # Offsets from the raw point forecast, compact enough to keep a year in memory.
            self.scenarios[date] = (
                point[order].astype(np.float32),
                (x[:, order] - point[order]).astype(np.float16),
            )
        return pred

    def save_scenarios(self, out_dir: Path) -> None:
        """Write every day's scenarios (dates, raw point (D, 24), offsets (D, M, 24))."""
        dates = sorted(self.scenarios)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"{self.name}.npz",
            dates=pd.DatetimeIndex(dates).to_numpy(),
            point=np.stack([self.scenarios[d][0] for d in dates]),
            offsets=np.stack([self.scenarios[d][1] for d in dates]),
        )


def rnn_flow_models(taus: list[float]) -> list[tuple[object, int]]:
    cfg = load_model_config()
    return [(RNNFlowForecaster("forecast", taus, cfg), cfg["refit_every_days"])]
