"""LightGBM day-ahead model on issue-time features.

One model predicts every target hour directly (the hour is a feature). It is always trained
on observed weather; ``weather`` only chooses what it is fed when it predicts:
``"oracle"`` (observed weather, an upper bound) or ``"forecast"`` (archived forecasts).

The model gives a point forecast. Quantiles are added afterwards from its own past
out-of-sample errors (``loadforecast.models.quantiles``): direct quantile regression was
tried and was badly overconfident (DESIGN section 7.2).
"""

from __future__ import annotations

import json

import lightgbm as lgb
import pandas as pd

from loadforecast.config import DATA_DIR, load_config
from loadforecast.features.build import build_dataset, future_columns, make_features

FEATURES_DIR = DATA_DIR / "features"
NOT_FEATURES = ["local_date", "y"]


def load_model_config() -> dict:
    return load_config("models/lightgbm")


def training_dataset(
    table: pd.DataFrame, dataset_hash: str, cfg: dict, backtest: dict
) -> pd.DataFrame:
    """Oracle-weather features and targets for every day, cached on disk by dataset hash."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURES_DIR / f"oracle_{dataset_hash}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    end = table["local_date"].max()
    data = build_dataset(
        table,
        cfg["features_start"],
        end,
        issue_time_local=backtest["issue_time_local"],
        tz=backtest["timezone"],
        weather="oracle",
    )
    data.to_parquet(path)
    return data


class LightGBMForecaster:
    needs_gap = True
    # Fitted boosters are shared between instances: the oracle and forecast variants train on
    # exactly the same rows and differ only in what they are fed at prediction time.
    _fitted: dict[str, lgb.Booster] = {}

    def __init__(self, dataset: pd.DataFrame, weather: str, cfg: dict | None = None):
        self.cfg = cfg or load_model_config()
        self.weather = weather
        self.name = f"lgbm_{weather}"
        self.future_columns = future_columns(weather)
        self.dataset = dataset[dataset["y"].notna()]
        self.features = [c for c in dataset.columns if c not in NOT_FEATURES]
        self.booster: lgb.Booster | None = None
        # Forecast weather other than temperature only exists from 2024-01-20.
        self.min_start = "2024-01-21" if weather == "forecast" else None

    def fit(self, history: pd.DataFrame) -> None:
        # Days before the issue day are fully observed, so their targets are known.
        cutoff = history["local_date"].iloc[-1]
        key = json.dumps([str(cutoff), len(self.dataset), self.cfg], sort_keys=True, default=str)
        if key not in self._fitted:
            train = self.dataset[self.dataset["local_date"] < cutoff]
            age_years = (train["local_date"].max() - train["local_date"]).dt.days / 365.25
            weight = 0.5 ** (age_years / self.cfg["half_life_years"])
            data = lgb.Dataset(train[self.features], train["y"], weight=weight)
            self._fitted[key] = lgb.train(self.cfg["params"], data, self.cfg["num_boost_round"])
        self.booster = self._fitted[key]

    def features_for(self, history: pd.DataFrame, future: pd.DataFrame, gap: pd.DataFrame):
        return make_features(history, future, gap, self.weather)[self.features]

    def predict(
        self, history: pd.DataFrame, future: pd.DataFrame, gap: pd.DataFrame
    ) -> pd.DataFrame:
        x = self.features_for(history, future, gap)
        return pd.DataFrame({"point": self.booster.predict(x)}, index=future.index)

    def shap_values(self, x: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contributions to each prediction (TreeSHAP), plus the ``base`` value."""
        contrib = self.booster.predict(x[self.features], pred_contrib=True)
        return pd.DataFrame(contrib, index=x.index, columns=[*self.features, "base"])


def importance(booster: lgb.Booster) -> pd.Series:
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    return (gain / gain.sum()).sort_values(ascending=False)


def lightgbm_models(
    table: pd.DataFrame, dataset_hash: str, backtest: dict
) -> list[tuple[object, int]]:
    cfg = load_model_config()
    dataset = training_dataset(table, dataset_hash, cfg, backtest)
    refit = cfg["refit_every_days"]
    return [
        (LightGBMForecaster(dataset, "oracle", cfg), refit),
        (LightGBMForecaster(dataset, "forecast", cfg), refit),
    ]
