from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seoul_bike_forecasting.prepare import prepare_feature_files


def make_source_month(month: int, stations: int = 3, hours: int = 240) -> pd.DataFrame:
    start = pd.to_datetime(str(month), format="%Y%m")
    timestamps = pd.date_range(start, periods=hours, freq="h")
    rows = []
    for station_number in range(1, stations + 1):
        for timestamp in timestamps:
            hour = timestamp.hour
            commute = int(hour in {7, 8, 9, 17, 18, 19})
            daily = 2.0 + 1.5 * np.sin(2 * np.pi * hour / 24)
            rent = max(
                0,
                round(daily + station_number * 0.4 + commute * 2 + timestamp.day % 3),
            )
            returns = max(
                0,
                round(daily + station_number * 0.3 + commute * 1.7 + (timestamp.day + 1) % 3),
            )
            rows.append(
                {
                    "station_id": f"ST-{station_number}",
                    "dt": timestamp,
                    "rent_count": float(rent),
                    "return_count": float(returns),
                    "district": "강남구" if station_number == 1 else pd.NA,
                    "lat": 37.50 + station_number / 100,
                    "lon": 127.00 + station_number / 100,
                    "year": timestamp.year,
                    "month": timestamp.month,
                    "hour": hour,
                    "dayofweek": timestamp.dayofweek,
                    "hour_sin": np.sin(2 * np.pi * hour / 24),
                    "hour_cos": np.cos(2 * np.pi * hour / 24),
                    "is_weekend": int(timestamp.dayofweek >= 5),
                    "is_holiday": 0,
                    "is_non_working_day": int(timestamp.dayofweek >= 5),
                    "is_commute_time": commute,
                    "TA": 12.0 + 8 * np.sin(2 * np.pi * hour / 24),
                    "WS": 2.0,
                    "HM": 55.0,
                    "RN_HR1": float(hour == 3),
                    "weather_distance_km": 1.2,
                    "weather_missing_flag": 0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def prepared_dir(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    source.mkdir()
    for month in (202301, 202302, 202303, 202304):
        make_source_month(month).to_csv(
            source / f"unscaled_{month}.csv.gz",
            index=False,
            compression="gzip",
        )
    station_metadata = tmp_path / "station_metadata.csv"
    pd.DataFrame(
        {
            "station_id": ["ST-1", "ST-2", "ST-3"],
            "district": ["강남구", "마포구", "종로구"],
        }
    ).to_csv(station_metadata, index=False)
    prepare_feature_files(
        source,
        output,
        station_metadata=station_metadata,
    )
    return output
