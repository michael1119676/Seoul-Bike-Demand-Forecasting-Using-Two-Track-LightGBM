from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TARGETS = ("rent_count", "return_count")
LAGS = (1, 24, 168)

FEATURE_GROUPS: dict[str, list[str]] = {
    "station": ["station_id_code", "district_code", "lat", "lon"],
    "time": [
        "year",
        "month",
        "hour",
        "dayofweek",
        "hour_sin",
        "hour_cos",
    ],
    "calendar": [
        "is_weekend",
        "is_holiday",
        "is_non_working_day",
        "is_commute_time",
    ],
    "weather": [
        "TA",
        "WS",
        "HM",
        "rain_amount",
        "dist",
        "weather_missing_flag",
    ],
    "lag": [
        f"{target}_lag_{lag}h"
        for target in TARGETS
        for lag in LAGS
    ],
    "interaction": [
        "rain_x_rent_lag_24h",
        "commute_x_rent_lag_1h",
        "non_working_x_rent_lag_168h",
        "freezing_x_rent_lag_24h",
        "rain_x_return_lag_24h",
        "commute_x_return_lag_1h",
        "non_working_x_return_lag_168h",
        "freezing_x_return_lag_24h",
    ],
}

MODEL_FEATURES = [
    feature
    for group_features in FEATURE_GROUPS.values()
    for feature in group_features
]
CATEGORICAL_FEATURES = ["station_id_code", "district_code"]


def add_strict_lag_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create station lags and invalidate any row without the exact time offset."""
    required = {"station_id", "dt", *TARGETS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Cannot create lags; missing columns: {missing}")

    result = frame.copy()
    result["dt"] = pd.to_datetime(result["dt"], errors="raise")
    result = result.sort_values(["station_id", "dt"], kind="mergesort").reset_index(drop=True)
    duplicate_count = int(result.duplicated(["station_id", "dt"]).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate station-hour rows")

    grouped = result.groupby("station_id", sort=False, observed=True)
    audit: dict[str, Any] = {
        "rows_checked": int(len(result)),
        "duplicate_station_hours": duplicate_count,
        "lag_columns": {},
    }
    for target in TARGETS:
        for lag in LAGS:
            feature = f"{target}_lag_{lag}h"
            lagged_value = grouped[target].shift(lag)
            lagged_dt = grouped["dt"].shift(lag)
            exact_offset = (result["dt"] - lagged_dt) == pd.Timedelta(hours=lag)
            candidate_count = int(lagged_value.notna().sum())
            invalid_offset_count = int((lagged_value.notna() & ~exact_offset).sum())
            result[feature] = lagged_value.where(exact_offset)
            audit["lag_columns"][feature] = {
                "candidate_rows": candidate_count,
                "invalid_time_offsets": invalid_offset_count,
                "missing_rows": int(result[feature].isna().sum()),
            }
    return result, audit


def add_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rain = pd.to_numeric(result["rain_amount"], errors="coerce").fillna(0)
    commute = pd.to_numeric(result["is_commute_time"], errors="coerce").fillna(0)
    non_working = pd.to_numeric(result["is_non_working_day"], errors="coerce").fillna(0)
    freezing = (pd.to_numeric(result["TA"], errors="coerce").fillna(1) <= 0).astype("int8")

    for target in TARGETS:
        prefix = target.removesuffix("_count")
        result[f"rain_x_{prefix}_lag_24h"] = rain * result[f"{target}_lag_24h"]
        result[f"commute_x_{prefix}_lag_1h"] = commute * result[f"{target}_lag_1h"]
        result[f"non_working_x_{prefix}_lag_168h"] = (
            non_working * result[f"{target}_lag_168h"]
        )
        result[f"freezing_x_{prefix}_lag_24h"] = freezing * result[f"{target}_lag_24h"]
    return result


def assert_no_target_leakage(feature_names: list[str]) -> dict[str, Any]:
    forbidden = set(TARGETS) & set(feature_names)
    suspicious = [
        name
        for name in feature_names
        if name.startswith("actual_") or name.startswith("pred_")
    ]
    if forbidden or suspicious:
        raise ValueError(
            f"Target leakage risk in model features: {sorted(forbidden | set(suspicious))}"
        )
    lag_names = [name for name in feature_names if "_lag_" in name]
    malformed_lags = [
        name
        for name in lag_names
        if not any(name.endswith(f"_lag_{lag}h") for lag in LAGS)
    ]
    if malformed_lags:
        raise ValueError(f"Unrecognized lag features: {malformed_lags}")
    return {
        "current_targets_excluded": True,
        "prediction_columns_excluded": True,
        "allowed_lags_hours": list(LAGS),
        "forecast_protocol": "rolling_1h",
        "protocol_assumption": "Targets through t-1 are observed before hour t is predicted.",
    }


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
