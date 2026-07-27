from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .features import MODEL_FEATURES, TARGETS


DISTRICTS = [
    "강남구",
    "강동구",
    "강북구",
    "강서구",
    "관악구",
    "광진구",
    "구로구",
    "금천구",
    "노원구",
    "도봉구",
    "동대문구",
    "동작구",
    "마포구",
    "서대문구",
    "서초구",
    "성동구",
    "성북구",
    "송파구",
    "양천구",
    "영등포구",
    "용산구",
    "은평구",
    "종로구",
    "중구",
    "중랑구",
]
DISTRICT_CODES = {name: index + 1 for index, name in enumerate(DISTRICTS)}
RAW_MODEL_COLUMNS = [
    "station_id",
    "district",
    *[
        feature
        for feature in MODEL_FEATURES
        if feature not in {"station_id_code", "district_code"}
    ],
]


@dataclass(frozen=True)
class FileSplit:
    train: list[Path]
    validation: list[Path]
    test: list[Path]


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    split: FileSplit
    audit: dict[str, Any]


def month_from_path(path: str | Path) -> int:
    match = re.search(r"model_features_(\d{6})\.parquet$", str(path))
    if not match:
        raise ValueError(f"Cannot parse YYYYMM from {path}")
    return int(match.group(1))


def list_feature_files(feature_dir: str | Path, pattern: str) -> list[Path]:
    files = sorted(Path(feature_dir).glob(pattern), key=month_from_path)
    if not files:
        raise FileNotFoundError(f"No {pattern} files found in {feature_dir}")
    return files


def make_file_split(files: list[Path], data_config: dict[str, Any]) -> FileSplit:
    train_start = int(data_config["train_start"])
    train_end = int(data_config["train_end"])
    validation_month = int(data_config["validation_month"])
    test_month = int(data_config["test_month"])
    split = FileSplit(
        train=[
            path
            for path in files
            if train_start <= month_from_path(path) <= train_end
        ],
        validation=[
            path for path in files if month_from_path(path) == validation_month
        ],
        test=[path for path in files if month_from_path(path) == test_month],
    )
    if not split.train or len(split.validation) != 1 or len(split.test) != 1:
        raise ValueError(
            "Incomplete time split: "
            f"train={len(split.train)}, validation={len(split.validation)}, "
            f"test={len(split.test)}"
        )
    train_months = {month_from_path(path) for path in split.train}
    validation_months = {month_from_path(path) for path in split.validation}
    test_months = {month_from_path(path) for path in split.test}
    if (train_months & validation_months) or (train_months & test_months):
        raise ValueError("Time split overlap detected")
    if max(train_months) >= min(validation_months) or max(validation_months) >= min(
        test_months
    ):
        raise ValueError("Time split is not strictly chronological")
    return split


def validate_schema(path: Path) -> None:
    columns = set(pq.ParquetFile(path).schema_arrow.names)
    required = set(RAW_MODEL_COLUMNS + list(TARGETS) + ["dt"])
    missing = sorted(required - columns)
    if missing:
        raise KeyError(f"{path.name} is missing required columns: {missing}")


def verify_test_lags(
    validation_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    lag_columns = [
        f"{target}_lag_{lag}h"
        for target in TARGETS
        for lag in (1, 24, 168)
    ]
    history_columns = ["dt", "station_id", *TARGETS]
    test_columns = [*history_columns, *lag_columns]
    history = pd.read_parquet(validation_path, columns=history_columns)
    test = pd.read_parquet(test_path, columns=test_columns)
    history = history.loc[
        history["station_id"].astype(str).str.fullmatch(r"ST-\d+")
    ].copy()
    test = test.loc[
        test["station_id"].astype(str).str.fullmatch(r"ST-\d+")
    ].copy()
    history["_is_test"] = False
    test["_is_test"] = True
    panel = pd.concat([history, test], ignore_index=True, sort=False)
    panel["dt"] = pd.to_datetime(panel["dt"], errors="raise")
    panel = panel.sort_values(["station_id", "dt"], kind="mergesort")
    grouped = panel.groupby("station_id", sort=False, observed=True)
    test_rows = panel["_is_test"]
    checks: dict[str, Any] = {}
    for target in TARGETS:
        for lag in (1, 24, 168):
            feature = f"{target}_lag_{lag}h"
            expected = grouped[target].shift(lag)
            expected_dt = grouped["dt"].shift(lag)
            exact_offset = (panel["dt"] - expected_dt) == pd.Timedelta(hours=lag)
            comparable = test_rows & exact_offset & expected.notna()
            mismatches = (
                panel.loc[comparable, feature].to_numpy()
                != expected.loc[comparable].to_numpy()
            )
            checks[feature] = {
                "test_rows": int(test_rows.sum()),
                "comparable_rows": int(comparable.sum()),
                "value_mismatches": int(mismatches.sum()),
                "final_missing_values": int(
                    panel.loc[test_rows, feature].isna().sum()
                ),
            }
    del history, test, panel
    return {
        "context_month": month_from_path(validation_path),
        "test_month": month_from_path(test_path),
        "method": "Recompute station shifts across validation-test context and require exact timestamp offsets.",
        "checks": checks,
    }


def preparation_summary(feature_dir: Path) -> dict[str, Any] | None:
    manifest_path = feature_dir / "preparation_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    months = manifest.get("months", [])
    return {
        "status": manifest.get("status"),
        "source_pattern": manifest.get("source_pattern"),
        "month_count": len(months),
        "first_month": months[0]["month"] if months else None,
        "last_month": months[-1]["month"] if months else None,
        "rows_read": sum(item["rows_read"] for item in months),
        "rows_written": sum(item["rows_written"] for item in months),
        "rows_dropped_missing_lag": sum(
            item["rows_dropped_missing_lag"] for item in months
        ),
        "district_missing_before": sum(
            item["district_missing_before"] for item in months
        ),
        "district_missing_after": sum(
            item["district_missing_after"] for item in months
        ),
        "district_backfill_entries": manifest.get("district_backfill_entries"),
        "lag_generation": manifest.get("lag_generation"),
    }


def _encode_station_id(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(
            series.astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )
        .fillna(0)
        .astype("int32")
    )


def _encode_district(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .map(DISTRICT_CODES)
        .fillna(0)
        .astype("int8")
    )


def _sample_frame(
    frame: pd.DataFrame,
    fraction: float,
    seed: int,
    month: int,
) -> pd.DataFrame:
    if fraction >= 1:
        return frame
    return frame.sample(frac=fraction, random_state=seed + month).sort_index()


def _load_files(
    files: list[Path],
    sample_fraction: float,
    seed: int,
    keep_metadata: bool,
    split_name: str,
) -> tuple[pd.DataFrame, int]:
    load_columns = list(dict.fromkeys(["dt", *RAW_MODEL_COLUMNS, *TARGETS]))
    chunks: list[pd.DataFrame] = []
    invalid_station_rows = 0
    for index, path in enumerate(files, start=1):
        print(
            f"[load {split_name} {index}/{len(files)}] {path.name}",
            flush=True,
        )
        frame = pd.read_parquet(path, columns=load_columns, engine="pyarrow")
        canonical_station = frame["station_id"].astype(str).str.fullmatch(r"ST-\d+")
        invalid_in_file = int((~canonical_station).sum())
        if invalid_in_file:
            print(
                f"[load {split_name}] excluded {invalid_in_file:,} rows with "
                "non-canonical station_id",
                flush=True,
            )
            invalid_station_rows += invalid_in_file
            frame = frame.loc[canonical_station].copy()
        frame = _sample_frame(frame, sample_fraction, seed, month_from_path(path))
        frame["station_id_code"] = _encode_station_id(frame["station_id"])
        frame["district_code"] = _encode_district(frame["district"])

        for column in MODEL_FEATURES:
            if column in {"station_id_code", "district_code"}:
                continue
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
            if frame[column].dtype.kind == "f":
                frame[column] = frame[column].astype("float32")
            elif frame[column].dtype.itemsize > 2:
                frame[column] = frame[column].astype("int16")
        for target in TARGETS:
            frame[target] = pd.to_numeric(frame[target], errors="raise").astype("float32")

        selected = [*MODEL_FEATURES, *TARGETS]
        if keep_metadata:
            frame["dt"] = pd.to_datetime(frame["dt"], errors="raise")
            selected = ["dt", "station_id", "district", *selected]
        chunks.append(frame[selected])
    return pd.concat(chunks, ignore_index=True), invalid_station_rows


def load_datasets(config: dict[str, Any]) -> DatasetBundle:
    data_config = config["data"]
    feature_dir = Path(data_config["feature_dir"])
    files = list_feature_files(feature_dir, data_config["file_pattern"])
    split = make_file_split(files, data_config)
    for path in [split.train[0], split.validation[0], split.test[0]]:
        validate_schema(path)
    lag_verification = verify_test_lags(split.validation[0], split.test[0])

    seed = int(config["seed"])
    train, train_invalid_station_rows = _load_files(
        split.train,
        float(data_config["train_sample_frac"]),
        seed,
        keep_metadata=False,
        split_name="train",
    )
    validation, validation_invalid_station_rows = _load_files(
        split.validation,
        float(data_config["validation_sample_frac"]),
        seed,
        keep_metadata=False,
        split_name="validation",
    )
    test, test_invalid_station_rows = _load_files(
        split.test,
        float(data_config["test_sample_frac"]),
        seed,
        keep_metadata=True,
        split_name="test",
    )
    audit = {
        "feature_dir": str(feature_dir),
        "available_months": [month_from_path(path) for path in files],
        "train_months": [month_from_path(path) for path in split.train],
        "validation_months": [month_from_path(path) for path in split.validation],
        "test_months": [month_from_path(path) for path in split.test],
        "split_is_strictly_chronological": True,
        "rows_loaded": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "sample_fractions": {
            "train": float(data_config["train_sample_frac"]),
            "validation": float(data_config["validation_sample_frac"]),
            "test": float(data_config["test_sample_frac"]),
        },
        "excluded_noncanonical_station_rows_before_sampling": {
            "train": train_invalid_station_rows,
            "validation": validation_invalid_station_rows,
            "test": test_invalid_station_rows,
        },
        "station_id_rule": r"^ST-\d+$",
        "test_time_range": [
            test["dt"].min().isoformat(),
            test["dt"].max().isoformat(),
        ],
        "test_unknown_district_rate": float((test["district_code"] == 0).mean()),
        "preparation_summary": preparation_summary(feature_dir),
        "test_lag_verification": lag_verification,
    }
    return DatasetBundle(
        train=train,
        validation=validation,
        test=test,
        split=split,
        audit=audit,
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_file_manifest(split: FileSplit) -> list[dict[str, Any]]:
    records = []
    split_by_path = {
        path: name
        for name, paths in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
        for path in paths
    }
    for path, split_name in split_by_path.items():
        metadata = pq.ParquetFile(path).metadata
        records.append(
            {
                "split": split_name,
                "month": month_from_path(path),
                "path": path.name,
                "rows": metadata.num_rows,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records
