from __future__ import annotations

import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from .baselines import all_baseline_predictions
from .data import DatasetBundle, input_file_manifest, load_datasets, sha256_file
from .evaluation import (
    baseline_improvements,
    compare_predictions,
    error_analysis,
    feature_importance_table,
)
from .features import (
    FEATURE_GROUPS,
    MODEL_FEATURES,
    assert_no_target_leakage,
    seed_everything,
)
from .models import predict_two_track, train_two_track
from .reporting import (
    plot_ablation,
    plot_feature_importance,
    plot_hourly_errors,
    plot_model_comparison,
    write_json,
    write_summary,
)


def _sample(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 1:
        return frame
    return frame.sample(frac=fraction, random_state=seed).sort_index()


def _run_ablation(
    bundle: DatasetBundle,
    config: dict[str, Any],
) -> pd.DataFrame:
    ablation_config = config["ablation"]
    seed = int(config["seed"])
    train = _sample(
        bundle.train,
        float(ablation_config["train_fraction_of_loaded"]),
        seed + 101,
    )
    validation = _sample(
        bundle.validation,
        float(ablation_config["validation_fraction"]),
        seed + 102,
    )
    test = _sample(
        bundle.test,
        float(ablation_config["test_fraction"]),
        seed + 103,
    )
    model_config = deepcopy(config["model"])
    model_config["num_boost_round"] = int(ablation_config["num_boost_round"])
    model_config["early_stopping_rounds"] = int(
        ablation_config["early_stopping_rounds"]
    )
    model_config["log_period"] = 0

    rows = []
    for dropped_group in ["none", *FEATURE_GROUPS]:
        print(f"[ablation] dropped_group={dropped_group}", flush=True)
        dropped = set(FEATURE_GROUPS.get(dropped_group, []))
        features = [feature for feature in MODEL_FEATURES if feature not in dropped]
        models = train_two_track(
            train,
            validation,
            features,
            model_config,
            seed + 200,
        )
        predictions = predict_two_track(models, test, features)
        result = compare_predictions(test, {"lightgbm": predictions})
        for row in result.to_dict(orient="records"):
            row["dropped_group"] = dropped_group
            row["feature_count"] = len(features)
            row["train_rows"] = len(train)
            row["validation_rows"] = len(validation)
            row["test_rows"] = len(test)
            rows.append(row)
    ablation = pd.DataFrame(rows)
    references = (
        ablation.loc[ablation["dropped_group"] == "none"]
        .set_index("target")["MAE"]
        .to_dict()
    )
    ablation["delta_MAE_vs_all_features"] = ablation.apply(
        lambda row: row["MAE"] - references[row["target"]],
        axis=1,
    )
    return ablation.sort_values(["target", "delta_MAE_vs_all_features"])


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _environment() -> dict[str, Any]:
    packages = [
        "pandas",
        "numpy",
        "pyarrow",
        "lightgbm",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "PyYAML",
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {package: version(package) for package in packages},
    }


def _output_manifest(result_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(result_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(result_dir.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    ]


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    seed = int(config["seed"])
    seed_everything(seed)
    leakage_audit = assert_no_target_leakage(MODEL_FEATURES)

    result_dir = Path(config["output"]["result_dir"])
    artifact_dir = Path(config["output"]["artifact_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("[pipeline] loading chronological splits", flush=True)
    bundle = load_datasets(config)
    print(
        "[pipeline] loaded "
        f"train={len(bundle.train):,}, validation={len(bundle.validation):,}, "
        f"test={len(bundle.test):,}",
        flush=True,
    )
    print("[pipeline] training main Two-Track LightGBM", flush=True)
    models = train_two_track(
        bundle.train,
        bundle.validation,
        MODEL_FEATURES,
        config["model"],
        seed,
        model_dir=artifact_dir / "models",
    )
    lightgbm_predictions = predict_two_track(models, bundle.test, MODEL_FEATURES)
    print("[pipeline] evaluating baselines and LightGBM", flush=True)
    prediction_sets = all_baseline_predictions(bundle.train, bundle.test)
    prediction_sets["lightgbm"] = lightgbm_predictions

    metrics = compare_predictions(bundle.test, prediction_sets)
    improvements = baseline_improvements(metrics)
    errors = error_analysis(bundle.test, lightgbm_predictions)
    importance = feature_importance_table(models)
    ablation = (
        _run_ablation(bundle, config)
        if bool(config["ablation"]["enabled"])
        else pd.DataFrame()
    )

    print("[pipeline] writing compact result artifacts", flush=True)
    metrics.to_csv(result_dir / "metrics.csv", index=False)
    improvements.to_csv(result_dir / "baseline_improvement.csv", index=False)
    importance.to_csv(result_dir / "feature_importance.csv", index=False)
    for level, table in errors.items():
        table.to_csv(result_dir / f"error_by_{level}.csv", index=False)
    if not ablation.empty:
        ablation.to_csv(result_dir / "ablation.csv", index=False)

    data_audit = {
        **bundle.audit,
        "leakage_audit": leakage_audit,
        "input_files": input_file_manifest(bundle.split),
    }
    write_json(data_audit, result_dir / "data_audit.json")
    plot_model_comparison(metrics, result_dir / "figures" / "model_comparison.png")
    plot_feature_importance(
        importance,
        result_dir / "figures" / "feature_importance.png",
    )
    plot_hourly_errors(
        errors["hour"],
        result_dir / "figures" / "error_by_hour.png",
    )
    if not ablation.empty:
        plot_ablation(ablation, result_dir / "figures" / "ablation.png")
    write_summary(
        metrics,
        improvements,
        bundle.audit,
        result_dir / "summary.md",
    )

    if bool(config["output"].get("save_test_predictions", False)):
        predictions = bundle.test[
            ["dt", "station_id", "district", "rent_count", "return_count"]
        ].copy()
        for model_name, model_predictions in prediction_sets.items():
            predictions[f"{model_name}_rent"] = model_predictions["rent_count"]
            predictions[f"{model_name}_return"] = model_predictions["return_count"]
        predictions.to_parquet(
            artifact_dir / "test_predictions.parquet",
            index=False,
        )

    manifest = {
        "status": "complete",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config": {
            key: value for key, value in config.items() if not key.startswith("_")
        },
        "config_path": config.get("_config_path"),
        "git": _git_state(),
        "environment": _environment(),
        "data_audit": data_audit,
        "best_iterations": {
            target: model.best_iteration for target, model in models.items()
        },
        "ablation_protocol": {
            "same_temporal_split": True,
            "same_rows_across_variants": True,
            "same_seed_across_variants": True,
            "seed": seed + 200,
        },
        "outputs": _output_manifest(result_dir),
    }
    write_json(manifest, result_dir / "run_manifest.json")
    return manifest
