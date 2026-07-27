from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


COLORS = {
    "historical_mean": "#94A3B8",
    "lag_24h": "#F59E0B",
    "lag_168h": "#8B5CF6",
    "lightgbm": "#0F766E",
}


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_model_comparison(metrics: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    order = ["rent_count", "return_count", "net_flow"]
    model_order = ["historical_mean", "lag_24h", "lag_168h", "lightgbm"]
    figure, axis = plt.subplots(figsize=(10, 5.4))
    sns.barplot(
        data=metrics,
        x="target",
        y="MAE",
        hue="model",
        order=order,
        hue_order=model_order,
        palette=COLORS,
        ax=axis,
    )
    axis.set_title("Same-split MAE: baselines vs Two-Track LightGBM")
    axis.set_xlabel("")
    axis.set_ylabel("MAE (bikes per station-hour)")
    axis.legend(title="Model", frameon=True, ncol=2)
    _save_figure(figure, path)


def plot_feature_importance(importance: pd.DataFrame, path: Path) -> None:
    targets = ["rent_count", "return_count"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=False)
    for axis, target in zip(axes, targets, strict=True):
        top = (
            importance.loc[importance["target"] == target]
            .nlargest(12, "importance_gain_fraction")
            .sort_values("importance_gain_fraction")
        )
        axis.barh(top["feature"], top["importance_gain_fraction"], color="#0F766E")
        axis.set_title(target)
        axis.set_xlabel("Fraction of total gain")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("LightGBM feature importance", fontsize=14)
    figure.tight_layout()
    _save_figure(figure, path)


def plot_hourly_errors(hourly: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.2))
    for column, label, color in (
        ("rent_MAE", "Rent", "#2563EB"),
        ("return_MAE", "Return", "#DC2626"),
        ("net_flow_MAE", "Net flow", "#0F766E"),
    ):
        axis.plot(hourly["hour"], hourly[column], marker="o", label=label, color=color)
    axis.set_xticks(range(24))
    axis.set_title("LightGBM error by hour")
    axis.set_xlabel("Hour")
    axis.set_ylabel("MAE (bikes per station-hour)")
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, path)


def plot_ablation(ablation: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=False)
    for axis, target in zip(
        axes,
        ["rent_count", "return_count", "net_flow"],
        strict=True,
    ):
        subset = (
            ablation.loc[
                (ablation["target"] == target)
                & (ablation["dropped_group"] != "none")
            ]
            .sort_values("delta_MAE_vs_all_features")
        )
        colors = [
            "#B91C1C" if value > 0 else "#64748B"
            for value in subset["delta_MAE_vs_all_features"]
        ]
        axis.barh(
            subset["dropped_group"],
            subset["delta_MAE_vs_all_features"],
            color=colors,
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(target)
        axis.set_xlabel("MAE change after dropping group")
    figure.suptitle("Feature-group ablation (positive means the group helps)", fontsize=14)
    figure.tight_layout()
    _save_figure(figure, path)


def write_summary(
    metrics: pd.DataFrame,
    improvements: pd.DataFrame,
    audit: dict[str, Any],
    path: Path,
) -> None:
    display_metrics = metrics.copy()
    for column in ("MAE", "RMSE", "R2"):
        display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.4f}")
    table_columns = ["model", "target", "MAE", "RMSE", "R2", "rows"]
    table_lines = [
        "| " + " | ".join(table_columns) + " |",
        "|" + "|".join(["---"] * len(table_columns)) + "|",
    ]
    for row in display_metrics[table_columns].itertuples(index=False, name=None):
        table_lines.append("| " + " | ".join(map(str, row)) + " |")
    improvement_lines = []
    for row in improvements.itertuples(index=False):
        improvement_lines.append(
            f"- `{row.target}`: LightGBM MAE {row.lightgbm_MAE:.4f}; "
            f"{row.percent_MAE_improvement:.2f}% vs best baseline `{row.best_baseline}`."
        )
    content = "\n".join(
        [
            "# Reproducible experiment summary",
            "",
            "This file is generated from the same chronological split for every model.",
            "",
            *table_lines,
            "",
            "## Improvement over baselines",
            "",
            *improvement_lines,
            "",
            "## Evaluation protocol",
            "",
            "- Forecast: rolling one-hour-ahead.",
            "- At hour `t`, observed targets through `t-1` are available.",
            "- Lag-24 and lag-168 baselines and LightGBM use the same test rows.",
            f"- Train rows loaded: {audit['rows_loaded']['train']:,}.",
            f"- Validation rows loaded: {audit['rows_loaded']['validation']:,}.",
            f"- Test rows loaded: {audit['rows_loaded']['test']:,}.",
            "",
            "These metrics are not valid for a frozen multi-hour batch forecast, because "
            "that protocol would require recursive or forecasted lag values.",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
