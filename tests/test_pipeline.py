from __future__ import annotations

from pathlib import Path

import pandas as pd

from seoul_bike_forecasting.config import load_config
from seoul_bike_forecasting.data import list_feature_files, make_file_split
from seoul_bike_forecasting.pipeline import run_experiment


def test_split_is_chronological(prepared_dir: Path) -> None:
    files = list_feature_files(prepared_dir, "model_features_*.parquet")
    split = make_file_split(
        files,
        {
            "train_start": 202301,
            "train_end": 202302,
            "validation_month": 202303,
            "test_month": 202304,
        },
    )
    assert len(split.train) == 2
    assert split.validation[0].name.endswith("202303.parquet")
    assert split.test[0].name.endswith("202304.parquet")


def test_synthetic_fixture_runs_end_to_end(
    prepared_dir: Path,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "test_config.yaml"
    result_dir = tmp_path / "results"
    artifact_dir = tmp_path / "artifacts"
    config_path.write_text(
        f"""
seed: 7
data:
  feature_dir: {prepared_dir}
  file_pattern: model_features_*.parquet
  train_start: 202301
  train_end: 202302
  validation_month: 202303
  test_month: 202304
  train_sample_frac: 1.0
  validation_sample_frac: 1.0
  test_sample_frac: 1.0
  forecast_protocol: rolling_1h
model:
  num_boost_round: 30
  early_stopping_rounds: 5
  log_period: 0
  params:
    objective: regression_l1
    metric: mae
    learning_rate: 0.1
    num_leaves: 15
    max_depth: 5
    min_data_in_leaf: 5
    feature_fraction: 1.0
    bagging_fraction: 1.0
    bagging_freq: 0
    num_threads: 1
    verbosity: -1
ablation:
  enabled: false
  train_fraction_of_loaded: 1.0
  validation_fraction: 1.0
  test_fraction: 1.0
  num_boost_round: 10
  early_stopping_rounds: 3
output:
  result_dir: {result_dir}
  artifact_dir: {artifact_dir}
  save_test_predictions: false
""",
        encoding="utf-8",
    )
    manifest = run_experiment(load_config(config_path))
    metrics = pd.read_csv(result_dir / "metrics.csv")
    assert manifest["status"] == "complete"
    assert len(metrics) == 12
    assert set(metrics["model"]) == {
        "historical_mean",
        "lag_24h",
        "lag_168h",
        "lightgbm",
    }
    assert set(metrics["target"]) == {"rent_count", "return_count", "net_flow"}
    assert (result_dir / "figures" / "model_comparison.png").exists()
    assert (result_dir / "run_manifest.json").exists()
