from __future__ import annotations

import gc
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .features import LAGS, MODEL_FEATURES, TARGETS, add_interaction_features, add_strict_lag_features


SOURCE_COLUMNS = [
    "station_id",
    "dt",
    "rent_count",
    "return_count",
    "district",
    "lat",
    "lon",
    "year",
    "month",
    "hour",
    "dayofweek",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_holiday",
    "is_non_working_day",
    "is_commute_time",
    "TA",
    "WS",
    "HM",
    "RN_HR1",
    "weather_distance_km",
    "weather_missing_flag",
]

OUTPUT_COLUMNS = [
    "dt",
    "station_id",
    "district",
    *TARGETS,
    *[
        feature
        for feature in MODEL_FEATURES
        if feature not in {"station_id_code", "district_code"}
    ],
]


def month_from_source_path(path: str | Path) -> int:
    match = re.search(r"(?:unscaled|model_features)_(\d{6})\.(?:csv\.gz|parquet)$", str(path))
    if not match:
        raise ValueError(f"Cannot parse YYYYMM from {path}")
    return int(match.group(1))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_station_district_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            master = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise ValueError(f"Could not decode station metadata {path}") from last_error

    if {"station_id", "district"}.issubset(master.columns):
        mapped = master[["station_id", "district"]].copy()
    elif {"대여소_ID", "주소1"}.issubset(master.columns):
        mapped = master[["대여소_ID", "주소1"]].rename(
            columns={"대여소_ID": "station_id", "주소1": "address"}
        )
        mapped["district"] = mapped["address"].astype(str).str.extract(
            r"서울(?:특별)?시\s+([가-힣]+구)",
            expand=False,
        )
        mapped = mapped[["station_id", "district"]]
    else:
        raise KeyError(
            "Station metadata must contain station_id/district or 대여소_ID/주소1"
        )
    mapped["station_id"] = mapped["station_id"].astype(str).str.strip()
    mapped["district"] = mapped["district"].astype("string").str.strip()
    mapped = mapped.dropna(subset=["district"]).drop_duplicates("station_id", keep="last")
    return dict(zip(mapped["station_id"], mapped["district"], strict=False))


def _read_source(path: Path) -> pd.DataFrame:
    if path.name.endswith(".csv.gz"):
        frame = pd.read_csv(path, usecols=SOURCE_COLUMNS, low_memory=False)
    elif path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=SOURCE_COLUMNS)
    else:
        raise ValueError(f"Unsupported source format: {path}")
    frame["station_id"] = frame["station_id"].astype(str).str.strip()
    frame["dt"] = pd.to_datetime(frame["dt"], errors="raise")
    frame["district"] = frame["district"].astype("string").str.strip()
    frame["district"] = frame["district"].replace({"": pd.NA, "nan": pd.NA})
    frame["rain_amount"] = pd.to_numeric(frame.pop("RN_HR1"), errors="coerce").fillna(0)
    frame["dist"] = pd.to_numeric(
        frame.pop("weather_distance_km"), errors="coerce"
    ).fillna(0)
    return frame


def prepare_feature_files(
    input_dir: str | Path,
    output_dir: str | Path,
    station_metadata: str | Path | None = None,
    source_pattern: str = "unscaled_*.csv.gz",
    overwrite: bool = False,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    files = sorted(input_path.glob(source_pattern), key=month_from_source_path)
    if not files:
        raise FileNotFoundError(f"No {source_pattern} files found in {input_path}")

    district_map = _read_station_district_map(
        Path(station_metadata) if station_metadata else None
    )
    history = pd.DataFrame(columns=["station_id", "dt", *TARGETS])
    monthly_audits: list[dict[str, Any]] = []

    for index, source_path in enumerate(files, start=1):
        month = month_from_source_path(source_path)
        destination = output_path / f"model_features_{month}.parquet"
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"{destination} already exists. Pass --overwrite to replace prepared data."
            )

        print(f"[prepare {index}/{len(files)}] reading {source_path.name}", flush=True)
        current = _read_source(source_path)
        district_missing_before = int(current["district"].isna().sum())
        if district_map:
            current["district"] = current["district"].fillna(
                current["station_id"].map(district_map)
            )
        district_missing_after = int(current["district"].isna().sum())
        current["district"] = current["district"].fillna("UNKNOWN")
        current["_current_month"] = True

        context = history.copy()
        context["_current_month"] = False
        combined = (
            current.copy()
            if context.empty
            else pd.concat([context, current], ignore_index=True, sort=False)
        )
        combined, lag_audit = add_strict_lag_features(combined)
        prepared = combined.loc[combined["_current_month"].fillna(False)].copy()
        required_lags = [
            f"{target}_lag_{lag}h"
            for target in TARGETS
            for lag in LAGS
        ]
        rows_before_lag_drop = len(prepared)
        prepared = prepared.dropna(subset=required_lags)
        prepared = add_interaction_features(prepared)

        for column in OUTPUT_COLUMNS:
            if column not in prepared:
                raise KeyError(f"Prepared frame is missing output column {column}")
        prepared = prepared[OUTPUT_COLUMNS].sort_values(
            ["dt", "station_id"], kind="mergesort"
        )
        for column in TARGETS:
            prepared[column] = pd.to_numeric(prepared[column], errors="raise").astype(
                "float32"
            )
        numeric_columns = [
            column
            for column in prepared.columns
            if column not in {"dt", "station_id", "district", *TARGETS}
        ]
        for column in numeric_columns:
            prepared[column] = pd.to_numeric(
                prepared[column], errors="coerce"
            ).fillna(0)
            if prepared[column].dtype.kind == "f":
                prepared[column] = prepared[column].astype("float32")

        prepared.to_parquet(
            destination,
            index=False,
            compression="zstd",
            engine="pyarrow",
        )
        monthly_audits.append(
            {
                "month": month,
                "source_file": str(source_path.resolve()),
                "source_size_bytes": source_path.stat().st_size,
                "source_sha256": sha256_file(source_path),
                "rows_read": int(len(current)),
                "rows_written": int(len(prepared)),
                "rows_dropped_missing_lag": int(rows_before_lag_drop - len(prepared)),
                "district_missing_before": district_missing_before,
                "district_missing_after": district_missing_after,
                "lag_audit": lag_audit,
                "output_file": str(destination.resolve()),
                "output_size_bytes": destination.stat().st_size,
                "output_sha256": sha256_file(destination),
            }
        )
        print(
            f"[prepare {index}/{len(files)}] wrote {destination.name}: "
            f"{len(prepared):,} rows",
            flush=True,
        )
        history = combined[["station_id", "dt", *TARGETS]].groupby(
            "station_id",
            sort=False,
            observed=True,
        ).tail(max(LAGS))
        del context, current, combined, prepared
        gc.collect()

    manifest = {
        "status": "complete",
        "source_pattern": source_pattern,
        "station_metadata": str(Path(station_metadata).resolve())
        if station_metadata
        else None,
        "forecast_protocol": "rolling_1h",
        "lag_generation": (
            "Station-sorted shift with an exact timestamp-offset check. Rows without "
            "all 1h/24h/168h lags are dropped."
        ),
        "district_backfill_entries": len(district_map),
        "months": monthly_audits,
    }
    (output_path / "preparation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
