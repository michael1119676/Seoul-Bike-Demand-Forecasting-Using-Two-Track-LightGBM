from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import TARGETS


def regression_metrics(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def evaluate_prediction_set(
    model_name: str,
    actual_frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        rows.append(
            {
                "model": model_name,
                "target": target,
                **regression_metrics(actual_frame[target], predictions[target]),
                "rows": int(len(actual_frame)),
            }
        )
    actual_net = (
        actual_frame["return_count"].to_numpy()
        - actual_frame["rent_count"].to_numpy()
    )
    predicted_net = predictions["return_count"] - predictions["rent_count"]
    rows.append(
        {
            "model": model_name,
            "target": "net_flow",
            **regression_metrics(actual_net, predicted_net),
            "rows": int(len(actual_frame)),
        }
    )
    return rows


def compare_predictions(
    actual_frame: pd.DataFrame,
    prediction_sets: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = [
        row
        for model_name, predictions in prediction_sets.items()
        for row in evaluate_prediction_set(model_name, actual_frame, predictions)
    ]
    return pd.DataFrame(rows).sort_values(["target", "MAE", "model"]).reset_index(
        drop=True
    )


def baseline_improvements(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, target_metrics in metrics.groupby("target", sort=False):
        lightgbm = target_metrics.loc[target_metrics["model"] == "lightgbm"].iloc[0]
        baselines = target_metrics.loc[target_metrics["model"] != "lightgbm"]
        best = baselines.sort_values("MAE").iloc[0]
        improvement = float(best["MAE"] - lightgbm["MAE"])
        rows.append(
            {
                "target": target,
                "best_baseline": best["model"],
                "best_baseline_MAE": float(best["MAE"]),
                "lightgbm_MAE": float(lightgbm["MAE"]),
                "absolute_MAE_improvement": improvement,
                "percent_MAE_improvement": (
                    100 * improvement / float(best["MAE"])
                    if float(best["MAE"]) != 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def error_analysis(
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> dict[str, pd.DataFrame]:
    detail = test[
        ["dt", "station_id", "district", "hour", *TARGETS]
    ].copy()
    for target in TARGETS:
        short = target.removesuffix("_count")
        detail[f"abs_error_{short}"] = np.abs(
            detail[target].to_numpy() - predictions[target]
        )
    actual_net = detail["return_count"] - detail["rent_count"]
    predicted_net = predictions["return_count"] - predictions["rent_count"]
    detail["abs_error_net_flow"] = np.abs(actual_net.to_numpy() - predicted_net)

    def aggregate(group_columns: list[str]) -> pd.DataFrame:
        aggregated = (
            detail.groupby(group_columns, dropna=False, observed=True)
            .agg(
                rows=("hour", "size"),
                rent_MAE=("abs_error_rent", "mean"),
                return_MAE=("abs_error_return", "mean"),
                net_flow_MAE=("abs_error_net_flow", "mean"),
            )
            .reset_index()
        )
        return aggregated.sort_values(
            ["net_flow_MAE", "rows"],
            ascending=[False, False],
        ).reset_index(drop=True)

    return {
        "station": aggregate(["station_id", "district"]),
        "hour": aggregate(["hour"]).sort_values("hour").reset_index(drop=True),
        "district": aggregate(["district"]),
    }


def feature_importance_table(models: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for target, model in models.items():
        total_gain = model.feature_importance(importance_type="gain").sum()
        for feature, gain, split in zip(
            model.feature_name(),
            model.feature_importance(importance_type="gain"),
            model.feature_importance(importance_type="split"),
            strict=True,
        ):
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "importance_gain": float(gain),
                    "importance_gain_fraction": (
                        float(gain / total_gain) if total_gain else 0.0
                    ),
                    "importance_split": int(split),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["target", "importance_gain"],
        ascending=[True, False],
    )
