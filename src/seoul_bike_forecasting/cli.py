from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .config import load_config
from .pipeline import run_experiment
from .prepare import prepare_feature_files


def _run_overrides(args: argparse.Namespace) -> dict[str, Any]:
    data: dict[str, Any] = {}
    output: dict[str, Any] = {}
    ablation: dict[str, Any] = {}
    if args.feature_dir:
        data["feature_dir"] = str(args.feature_dir)
    if args.train_sample_frac is not None:
        data["train_sample_frac"] = args.train_sample_frac
    if args.result_dir:
        output["result_dir"] = str(args.result_dir)
    if args.artifact_dir:
        output["artifact_dir"] = str(args.artifact_dir)
    if args.disable_ablation:
        ablation["enabled"] = False
    return {
        key: value
        for key, value in (
            ("data", data),
            ("output", output),
            ("ablation", ablation),
        )
        if value
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seoul-bike",
        description="Reproducible Seoul Bike two-track forecasting",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Create leak-audited Two-Track parquet features",
    )
    prepare_parser.add_argument("--input-dir", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--station-metadata", type=Path)
    prepare_parser.add_argument(
        "--source-pattern",
        default="unscaled_*.csv.gz",
    )
    prepare_parser.add_argument("--overwrite", action="store_true")

    run_parser = subparsers.add_parser(
        "run",
        help="Train, evaluate, ablate, and generate reports",
    )
    run_parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    run_parser.add_argument("--feature-dir", type=Path)
    run_parser.add_argument("--result-dir", type=Path)
    run_parser.add_argument("--artifact-dir", type=Path)
    run_parser.add_argument("--train-sample-frac", type=float)
    run_parser.add_argument("--disable-ablation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_feature_files(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            station_metadata=args.station_metadata,
            source_pattern=args.source_pattern,
            overwrite=args.overwrite,
        )
        print(
            f"Prepared {len(manifest['months'])} months in "
            f"{Path(args.output_dir).resolve()}"
        )
        return

    config = load_config(args.config, _run_overrides(args))
    manifest = run_experiment(config)
    print(
        f"Experiment complete: {manifest['config']['output']['result_dir']}"
    )


if __name__ == "__main__":
    main()
