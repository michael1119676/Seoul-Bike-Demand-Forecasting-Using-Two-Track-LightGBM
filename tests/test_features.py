from __future__ import annotations

import pandas as pd

from seoul_bike_forecasting.features import (
    MODEL_FEATURES,
    add_strict_lag_features,
    assert_no_target_leakage,
)


def test_lags_use_only_exact_prior_station_hours() -> None:
    timestamps = pd.date_range("2026-01-01", periods=200, freq="h")
    frame = pd.DataFrame(
        {
            "station_id": ["ST-1"] * len(timestamps),
            "dt": timestamps,
            "rent_count": range(len(timestamps)),
            "return_count": range(1000, 1000 + len(timestamps)),
        }
    )
    lagged, audit = add_strict_lag_features(frame)
    row = lagged.iloc[168]
    assert row["rent_count_lag_1h"] == 167
    assert row["rent_count_lag_24h"] == 144
    assert row["rent_count_lag_168h"] == 0
    assert row["return_count_lag_168h"] == 1000
    assert audit["duplicate_station_hours"] == 0
    assert all(
        item["invalid_time_offsets"] == 0
        for item in audit["lag_columns"].values()
    )


def test_gap_is_not_mislabeled_as_a_lag() -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["ST-1", "ST-1"],
            "dt": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 02:00"]),
            "rent_count": [3, 99],
            "return_count": [4, 88],
        }
    )
    lagged, audit = add_strict_lag_features(frame)
    assert pd.isna(lagged.loc[1, "rent_count_lag_1h"])
    assert audit["lag_columns"]["rent_count_lag_1h"]["invalid_time_offsets"] == 1


def test_model_features_exclude_current_targets() -> None:
    audit = assert_no_target_leakage(MODEL_FEATURES)
    assert audit["current_targets_excluded"] is True
