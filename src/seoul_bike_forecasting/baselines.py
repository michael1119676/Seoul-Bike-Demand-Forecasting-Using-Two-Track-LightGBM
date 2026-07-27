from __future__ import annotations

import numpy as np
import pandas as pd

from .features import TARGETS


def historical_mean_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    train_keys = train[["station_id_code", "hour"]]
    test_keys = test[["station_id_code", "hour"]]
    for target in TARGETS:
        fitting = train_keys.copy()
        fitting[target] = train[target].to_numpy()
        station_hour = fitting.groupby(
            ["station_id_code", "hour"],
            observed=True,
        )[target].mean()
        hour_mean = fitting.groupby("hour", observed=True)[target].mean()
        global_mean = float(fitting[target].mean())

        lookup = pd.MultiIndex.from_frame(test_keys)
        prediction = station_hour.reindex(lookup).to_numpy(dtype="float64")
        missing = np.isnan(prediction)
        if missing.any():
            prediction[missing] = (
                test_keys.loc[missing, "hour"].map(hour_mean).fillna(global_mean)
            )
        predictions[target] = np.clip(prediction, 0, None)
    return predictions


def lag_predictions(test: pd.DataFrame, lag: int) -> dict[str, np.ndarray]:
    return {
        target: np.clip(
            test[f"{target}_lag_{lag}h"].to_numpy(dtype="float64"),
            0,
            None,
        )
        for target in TARGETS
    }


def all_baseline_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        "historical_mean": historical_mean_predictions(train, test),
        "lag_24h": lag_predictions(test, 24),
        "lag_168h": lag_predictions(test, 168),
    }
