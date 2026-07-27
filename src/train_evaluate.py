"""Backward-compatible script entry point for the canonical package CLI."""

from __future__ import annotations

import sys

from seoul_bike_forecasting.cli import main


if __name__ == "__main__":
    main(["run", *sys.argv[1:]])
