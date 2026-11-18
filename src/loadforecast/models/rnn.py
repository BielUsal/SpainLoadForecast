"""Sequence model (PyTorch): a GRU encoder-decoder with a quantile head. DESIGN section 7.4.

The encoder reads the last ``encoder_hours`` of demand, weather and calendar up to the issue
time. The decoder then runs over the hours that follow (the rest of the issue day and the
target day), fed with what is known about them in advance: the calendar, the weather, and the
demand one week earlier. For every one of those hours a linear head gives the quantiles
directly, trained with the pinball loss.

Training uses a forecast origin at every hour of the history, not only at 10:00, and scores
all ``decoder_hours`` that follow it: about 24 times more examples than one per day. Like
the other models it is trained on observed weather; ``weather`` only chooses what the decoder
is fed when it predicts.

The forecast is the average of ``seeds`` networks that differ only in their random seed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from loadforecast.config import load_config
from loadforecast.features.build import future_columns, weather_sources
from loadforecast.models.quantiles import quantile_column

Y_CENTRE, Y_SCALE = 27000.0, 5000.0  # MW
TEMP_CENTRE, TEMP_SCALE = 15.0, 8.0  # deg C
RAD_SCALE = 400.0  # W/m2
WEEK = 168
FLAGS = ["holiday_intensity", "bridge_intensity", "christmas", "august", "holy_week"]
N_COVARIATES = 4 + 7 + len(FLAGS) + 2


def load_model_config() -> dict:
    return load_config("models/rnn")


def covariates(frame: pd.DataFrame, temp: np.ndarray, rad: np.ndarray) -> np.ndarray:
    """Per-hour inputs known in advance: clock, calendar and weather. Shape (rows, C)."""
    hour = 2 * np.pi * frame["local_hour"].to_numpy() / 24
    doy = 2 * np.pi * frame["local_date"].dt.day_of_year.to_numpy() / 365.25
    weekday = np.eye(7)[frame["weekday"].to_numpy()]
    flags = frame[FLAGS].to_numpy(dtype=float)
    weather = np.column_stack([(temp - TEMP_CENTRE) / TEMP_SCALE, rad / RAD_SCALE])
    clock = np.column_stack([np.sin(hour), np.cos(hour), np.sin(doy), np.cos(doy)])
    return np.column_stack([clock, weekday, flags, weather]).astype(np.float32)


def scale_demand(mw: np.ndarray) -> np.ndarray:
    return ((mw - Y_CENTRE) / Y_SCALE).astype(np.float32)


class Seq2Quantiles(nn.Module):
    def __init__(self, n_taus: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.encoder = nn.GRU(N_COVARIATES + 1, hidden, layers, batch_first=True)
        self.decoder = nn.GRU(N_COVARIATES + 1, hidden, layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_taus))
        self.median = n_taus // 2

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """``past`` (B, L, C+1) ends with demand; ``future`` (B, F, C+1) ends with the demand
        one week earlier. Returns ordered quantiles (B, F, taus) in scaled units."""
        return self.quantiles(self.decode(past, future), future)

    def decode(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """Decoder outputs (B, F, hidden)."""
        _, state = self.encoder(past)
        out, _ = self.decoder(future, state)
        return out

    def quantiles(self, out: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.dropout(out))
        # The median is an offset from the same hour last week; the other quantiles are the
        # median minus or plus cumulated positive steps, so they cannot cross.
        mid = future[..., -1:] + raw[..., self.median : self.median + 1]
        steps = nn.functional.softplus(raw)
        below = mid - torch.cumsum(steps[..., : self.median].flip(-1), dim=-1).flip(-1)
        above = mid + torch.cumsum(steps[..., self.median + 1 :], dim=-1)
        return torch.cat([below, mid, above], dim=-1)


def pinball(pred: torch.Tensor, y: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss over the hours with a known ``y`` (NaN where unknown)."""
    known = ~torch.isnan(y)
    diff = torch.nan_to_num(y)[..., None] - pred
    loss = torch.maximum(taus * diff, (taus - 1) * diff).mean(dim=-1)
    return (loss * known).sum() / known.sum()


def training_arrays(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Scaled demand (rows,) and covariates (rows, C) of the history, with observed weather."""
    x = covariates(history, history["temp_obs"].to_numpy(), history["rad_obs"].to_numpy())
    return scale_demand(history["demand_mw"].to_numpy()), x


def valid_origins(y: np.ndarray, encoder_hours: int, decoder_hours: int) -> np.ndarray:
    """Row numbers that can be a forecast origin: a complete encoder window before them and
    ``decoder_hours`` rows after them."""
    missing = np.concatenate([[0], np.cumsum(np.isnan(y))])
    origins = np.arange(encoder_hours, len(y) - decoder_hours + 1)
    complete = missing[origins] - missing[origins - encoder_hours] == 0
    return origins[complete]


def train_network(
    y: np.ndarray, x: np.ndarray, taus: list[float], cfg: dict, seed: int, device: str
) -> Seq2Quantiles:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, dec = cfg["encoder_hours"], cfg["decoder_hours"]
    origins = valid_origins(y, enc, dec)
    y_t = torch.tensor(y, device=device)
    x_t = torch.tensor(x, device=device)
    tau_t = torch.tensor(taus, device=device)
    past_offsets = torch.arange(-enc, 0, device=device)
    future_offsets = torch.arange(0, dec, device=device)

    model = Seq2Quantiles(len(taus), cfg["hidden"], cfg["layers"], cfg["dropout"]).to(device)
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
            loss = pinball(model(past, future), y_t[future_rows], tau_t)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_norm"])
            optimiser.step()
            schedule.step()
    return model.eval()


def fill_gaps(demand: pd.Series) -> np.ndarray:
    """Demand with missing hours (masked days) replaced by the value one week earlier."""
    values = demand.to_numpy(dtype=float).copy()
    for i in np.flatnonzero(np.isnan(values)):
        if i >= WEEK:
            values[i] = values[i - WEEK]
    return values


class RNNForecaster:
    needs_gap = True
    native_quantiles = True
    # Trained networks are shared between the weather variants, which differ only in what the
    # decoder is fed at prediction time.
    _fitted: dict[str, list[Seq2Quantiles]] = {}

    def __init__(self, weather: str, taus: list[float], cfg: dict | None = None):
        self.cfg = cfg or load_model_config()
        self.weather = weather
        self.taus = taus
        self.name = f"rnn_{weather}"
        self.future_columns = future_columns(weather)
        self.target_src, self.gap_src = weather_sources(weather)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.networks: list[Seq2Quantiles] = []
        self.recalibrate = bool(self.cfg.get("recalibrate", False))
        self.point_from_median = bool(self.cfg.get("point_from_median", False))
        # Forecast radiation only exists from 2024-01-20.
        self.min_start = "2024-01-21" if weather == "forecast" else None

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
        by_seed = torch.stack([net(past_t, ahead_t)[0, -len(future) :] for net in self.networks])
        by_seed = (by_seed * Y_SCALE + Y_CENTRE).cpu().numpy()  # (seeds, hours, taus)
        mean = by_seed.mean(axis=0)
        pred = pd.DataFrame(index=future.index)
        pred["point"] = mean[:, len(self.taus) // 2]
        for j, tau in enumerate(self.taus):
            pred[quantile_column(tau)] = mean[:, j]
        for k in range(len(self.networks)):
            pred[f"point_seed{k}"] = by_seed[k, :, len(self.taus) // 2]
        return pred


def rnn_models(taus: list[float]) -> list[tuple[object, int]]:
    cfg = load_model_config()
    refit = cfg["refit_every_days"]
    return [
        (RNNForecaster("oracle", taus, cfg), refit),
        (RNNForecaster("forecast", taus, cfg), refit),
    ]
