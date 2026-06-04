from __future__ import annotations

import argparse
import gc
import json
import re
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


DEFAULT_BASE_DIR = Path("/content/drive/MyDrive/ML1_데이터/전처리 데이터 최종본")
DEFAULT_FEATURE_DIR = DEFAULT_BASE_DIR / "reprocessed_rent_return_v2_strict"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "trained_models_v2"

TRAIN_START_YM = 202212
TRAIN_END_YM = 202512
VAL_YM = 202603
TEST_YM = 202604
TRAIN_SAMPLE_FRAC = 0.30
RANDOM_STATE = 42

SELECTED_FEATURES = [
    "year", "month", "hour", "dayofweek", "hour_sin", "hour_cos",
    "is_weekend", "is_holiday", "is_non_working_day", "is_commute_time",
    "rent_count_lag_1h", "rent_count_lag_24h", "rent_count_lag_168h",
    "return_count_lag_1h", "return_count_lag_24h", "return_count_lag_168h",
    "TA", "WS", "HM", "rain_amount", "dist", "weather_missing_flag",
    "rain_x_rent_lag_24h", "commute_x_rent_lag_1h",
    "non_working_x_rent_lag_168h", "freezing_x_rent_lag_24h",
    "rain_x_return_lag_24h", "commute_x_return_lag_1h",
    "non_working_x_return_lag_168h", "freezing_x_return_lag_24h",
]

CAT_FEATURES = ["station_id", "district"]

DISTRICT_MAP = {
    "강남구": 1, "강동구": 2, "강북구": 3, "강서구": 4, "관악구": 5,
    "광진구": 6, "구로구": 7, "금천구": 8, "노원구": 9, "도봉구": 10,
    "동대문구": 11, "동작구": 12, "마포구": 13, "서대문구": 14, "서초구": 15,
    "성동구": 16, "성북구": 17, "송파구": 18, "양천구": 19, "영등포구": 20,
    "용산구": 21, "은평구": 22, "종로구": 23, "중구": 24, "중랑구": 25,
}

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}


def month_from_path(path: str | Path) -> int:
    match = re.search(r"model_features_(\d{6})\.parquet$", str(path))
    if not match:
        raise ValueError(f"Cannot parse YYYYMM from {path}")
    return int(match.group(1))


def list_feature_files(feature_dir: Path) -> list[Path]:
    files = sorted(feature_dir.glob("model_features_*.parquet"), key=month_from_path)
    if not files:
        raise FileNotFoundError(f"No model_features_*.parquet files found in {feature_dir}")
    return files


def split_files(files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    train_files = [p for p in files if TRAIN_START_YM <= month_from_path(p) <= TRAIN_END_YM]
    val_files = [p for p in files if month_from_path(p) == VAL_YM]
    test_files = [p for p in files if month_from_path(p) == TEST_YM]
    if not train_files or not val_files or not test_files:
        raise ValueError(
            "Invalid split. "
            f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        )
    return train_files, val_files, test_files


def validate_columns(file_path: Path) -> None:
    columns = set(pd.read_parquet(file_path, engine="pyarrow").columns)
    required = set(SELECTED_FEATURES + CAT_FEATURES + ["rent_count", "return_count"])
    missing = sorted(required - columns)
    if missing:
        raise KeyError(f"{file_path.name} is missing required columns: {missing}")


def encode_station_id(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series.astype(str).str.replace(r"\D", "", regex=True), errors="coerce")
        .fillna(0)
        .astype("int32")
    )


def encode_district(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    return cleaned.map(DISTRICT_MAP).fillna(0).astype("int32")


def load_model_matrix(
    files: list[Path],
    target_col: str,
    sample_frac: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    chunks: list[pd.DataFrame] = []
    load_cols = SELECTED_FEATURES + CAT_FEATURES + [target_col]

    for file_path in files:
        df = pd.read_parquet(file_path, columns=load_cols)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=RANDOM_STATE)

        df["station_id"] = encode_station_id(df["station_id"])
        df["district"] = encode_district(df["district"])

        for col in SELECTED_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if col in {"year", "month", "hour", "dayofweek"} or "flag" in col or col.startswith("is_"):
                df[col] = df[col].astype("int16")
            else:
                df[col] = df[col].astype("float32")

        for col in CAT_FEATURES:
            df[col] = df[col].astype("category")

        chunks.append(df)

    combined = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    x = combined[SELECTED_FEATURES + CAT_FEATURES]
    y = pd.to_numeric(combined[target_col], errors="coerce").fillna(0).astype("float32")
    return x, y


def train_model(
    train_files: list[Path],
    val_files: list[Path],
    target_col: str,
    output_dir: Path,
) -> lgb.Booster:
    x_train, y_train = load_model_matrix(train_files, target_col, sample_frac=TRAIN_SAMPLE_FRAC)
    x_val, y_val = load_model_matrix(val_files, target_col, sample_frac=1.0)

    train_dataset = lgb.Dataset(x_train, label=y_train, categorical_feature=CAT_FEATURES)
    val_dataset = lgb.Dataset(x_val, label=y_val, reference=train_dataset, categorical_feature=CAT_FEATURES)
    del x_train, y_train, x_val, y_val
    gc.collect()

    model = lgb.train(
        LGBM_PARAMS,
        train_dataset,
        num_boost_round=1000,
        valid_sets=[train_dataset, val_dataset],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )
    model_path = output_dir / f"lgb_{target_col}.txt"
    model.save_model(str(model_path))
    return model


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_feature_importance(model: lgb.Booster, target: str, output_dir: Path) -> None:
    importance = (
        pd.DataFrame(
            {
                "target": target,
                "feature": model.feature_name(),
                "importance_gain": model.feature_importance(importance_type="gain"),
                "importance_split": model.feature_importance(importance_type="split"),
            }
        )
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(output_dir / f"feature_importance_{target}.csv", index=False)


def main(feature_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list_feature_files(feature_dir)
    validate_columns(files[0])
    train_files, val_files, test_files = split_files(files)

    print("Feature directory:", feature_dir)
    print("Output directory :", output_dir)
    print("Train months     :", month_from_path(train_files[0]), "-", month_from_path(train_files[-1]))
    print("Validation month :", VAL_YM)
    print("Test month       :", TEST_YM)

    rent_model = train_model(train_files, val_files, "rent_count", output_dir)
    return_model = train_model(train_files, val_files, "return_count", output_dir)

    x_test, y_rent = load_model_matrix(test_files, "rent_count", sample_frac=1.0)
    _, y_return = load_model_matrix(test_files, "return_count", sample_frac=1.0)

    pred_rent = np.clip(rent_model.predict(x_test), 0, None)
    pred_return = np.clip(return_model.predict(x_test), 0, None)

    metrics = [
        {"target": "rent_count", **evaluate_predictions(y_rent, pred_rent)},
        {"target": "return_count", **evaluate_predictions(y_return, pred_return)},
    ]
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_dir / "model_comparison.csv", index=False)
    print(metrics_df)

    prediction_df = x_test[["station_id", "district", "hour"]].copy()
    prediction_df["actual_rent"] = y_rent.to_numpy()
    prediction_df["actual_return"] = y_return.to_numpy()
    prediction_df["pred_rent"] = pred_rent
    prediction_df["pred_return"] = pred_return
    prediction_df["pred_net_flow"] = prediction_df["pred_return"] - prediction_df["pred_rent"]
    prediction_df["actual_net_flow"] = prediction_df["actual_return"] - prediction_df["actual_rent"]
    prediction_df.to_parquet(output_dir / "predictions_202604.parquet", index=False)

    station_flow = (
        prediction_df.groupby("station_id", as_index=False)["pred_net_flow"]
        .sum()
        .sort_values("pred_net_flow")
    )
    station_flow.head(20).to_csv(output_dir / "top_shortage_stations.csv", index=False)
    station_flow.tail(20).sort_values("pred_net_flow", ascending=False).to_csv(
        output_dir / "top_surplus_stations.csv", index=False
    )

    save_feature_importance(rent_model, "rent_count", output_dir)
    save_feature_importance(return_model, "return_count", output_dir)

    manifest = {
        "feature_dir": str(feature_dir),
        "output_dir": str(output_dir),
        "train_months": [month_from_path(p) for p in train_files],
        "validation_month": VAL_YM,
        "test_month": TEST_YM,
        "train_sample_frac": TRAIN_SAMPLE_FRAC,
        "selected_features": SELECTED_FEATURES,
        "categorical_features": CAT_FEATURES,
        "lgbm_params": LGBM_PARAMS,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    main(args.feature_dir, args.output_dir)
