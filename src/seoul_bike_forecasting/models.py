from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES, TARGETS


def _deterministic_params(config: dict[str, Any], seed: int) -> dict[str, Any]:
    params = dict(config["params"])
    params.update(
        {
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "deterministic": True,
            "force_col_wise": True,
        }
    )
    return params


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target: str,
    features: list[str],
    model_config: dict[str, Any],
    seed: int,
) -> lgb.Booster:
    train_set = lgb.Dataset(
        train[features],
        label=train[target],
        categorical_feature=[
            feature for feature in CATEGORICAL_FEATURES if feature in features
        ],
        free_raw_data=True,
    )
    validation_set = lgb.Dataset(
        validation[features],
        label=validation[target],
        reference=train_set,
        categorical_feature=[
            feature for feature in CATEGORICAL_FEATURES if feature in features
        ],
        free_raw_data=True,
    )
    callbacks: list[Any] = [
        lgb.early_stopping(
            int(model_config["early_stopping_rounds"]),
            verbose=False,
        )
    ]
    log_period = int(model_config.get("log_period", 0))
    if log_period:
        callbacks.append(lgb.log_evaluation(log_period))
    return lgb.train(
        _deterministic_params(model_config, seed),
        train_set,
        num_boost_round=int(model_config["num_boost_round"]),
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=callbacks,
    )


def train_two_track(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    model_config: dict[str, Any],
    seed: int,
    model_dir: str | Path | None = None,
    filename_prefix: str = "",
) -> dict[str, lgb.Booster]:
    models: dict[str, lgb.Booster] = {}
    if model_dir:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
    for offset, target in enumerate(TARGETS):
        print(
            f"[train] target={target}, rows={len(train):,}, features={len(features)}",
            flush=True,
        )
        model = train_model(
            train,
            validation,
            target,
            features,
            model_config,
            seed + offset,
        )
        models[target] = model
        print(
            f"[train] target={target}, best_iteration={model.best_iteration}",
            flush=True,
        )
        if model_dir:
            model.save_model(
                str(Path(model_dir) / f"{filename_prefix}lgb_{target}.txt")
            )
    return models


def predict_two_track(
    models: dict[str, lgb.Booster],
    frame: pd.DataFrame,
    features: list[str],
) -> dict[str, np.ndarray]:
    return {
        target: np.clip(
            models[target].predict(
                frame[features],
                num_iteration=models[target].best_iteration,
            ),
            0,
            None,
        )
        for target in TARGETS
    }
