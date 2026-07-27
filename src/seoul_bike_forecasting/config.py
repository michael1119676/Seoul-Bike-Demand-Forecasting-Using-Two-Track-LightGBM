from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if overrides:
        config = _deep_update(config, overrides)
    validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {"data", "model", "ablation", "output"}
    missing = sorted(required_sections - set(config))
    if missing:
        raise ConfigError(f"Missing config sections: {missing}")

    data = config["data"]
    ordered_months = [
        int(data["train_start"]),
        int(data["train_end"]),
        int(data["validation_month"]),
        int(data["test_month"]),
    ]
    if ordered_months != sorted(ordered_months) or len(set(ordered_months)) != 4:
        raise ConfigError(
            "Expected train_start < train_end < validation_month < test_month; "
            f"received {ordered_months}"
        )
    if data.get("forecast_protocol") != "rolling_1h":
        raise ConfigError(
            "Only forecast_protocol=rolling_1h is implemented. This protocol assumes "
            "targets through t-1 are observed before predicting hour t."
        )
    for name in (
        "train_sample_frac",
        "validation_sample_frac",
        "test_sample_frac",
    ):
        value = float(data[name])
        if not 0 < value <= 1:
            raise ConfigError(f"{name} must be in (0, 1], got {value}")

    if int(config["seed"]) < 0:
        raise ConfigError("seed must be non-negative")
