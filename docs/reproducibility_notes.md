# Reproducibility notes

## What was available

The workspace contained 41 monthly, station-hour source files from 2022-12
through 2026-04 with separate `rent_count` and `return_count` targets. It did not
contain the exact `reprocessed_rent_return_v2_strict` parquet directory named by
the original code. The new preparation command therefore regenerated Two-Track
features from the available real source files.

The source data and generated parquet files are intentionally not committed. The
committed audit records input filenames, row counts, sizes, and SHA-256 hashes.

## Independent rerun versus the attached report

The attached final report contains results, but the repository previously did
not contain the artifacts or executable lineage needed to verify them. The new
run is an independent rerun, not a transcription of report values.

| Target | Metric | Attached report | Independent rerun |
|---|---:|---:|---:|
| Rent | MAE | 0.9914 | 1.0074 |
| Rent | RMSE | 2.1759 | 1.8225 |
| Rent | R² | 0.8067 | 0.7130 |
| Return | MAE | 0.9606 | 0.9660 |
| Return | RMSE | 2.1045 | 1.7281 |
| Return | R² | 0.8247 | 0.7520 |
| Net flow | MAE | 1.3400 | 1.2589 |

The exact cause of the differences cannot be isolated because the report's
processed parquet snapshot and run manifest were absent. Plausible lineage
differences include station scope, district enrichment, and the exact lag
generation snapshot. The committed results should be treated as the verified
repository result.

## Leakage controls

- Files are split by month before training: train 2022-12 through 2025-12,
  validation 2026-03, test 2026-04.
- Current-hour targets are never model features.
- Lags are generated after sorting by station and timestamp.
- A lag is accepted only when the timestamp difference is exactly 1, 24, or 168
  hours; otherwise it is set missing and the affected row is excluded.
- April lags were independently checked against the available prepared
  March-April history. Depending on lag, 1,998,144-2,000,983 of 2,001,000 test
  rows were directly comparable; all six lag columns had zero value mismatches,
  and the final test features had zero missing lag values.
- The protocol is rolling one-hour-ahead. Actual outcomes through `t-1` are
  assumed observed before predicting hour `t`.

That last assumption matters. These metrics do not describe a frozen, multi-hour
batch forecast. Such a deployment needs recursive or separately forecast lag
values and a new evaluation.

## Data quality decisions

- Address-based station metadata reduced the April unknown-district rate from
  47.7% to 0.40%.
- A non-physical `station_id="X"` placeholder appeared once per hour. The loader
  excludes rows that do not match `^ST-\d+$` and records the excluded counts:
  26,712 train-period rows, 744 validation rows, and 720 test rows before
  sampling.
- The first available month lacks a prior local 168-hour context. Rows without
  all exact lags are excluded rather than imputed.

## Completed and still open

Completed:

- actual-data LightGBM run with fixed seed and environment;
- historical mean, lag-24, and lag-168 baselines on the same test rows;
- rent, return, and direct net-flow MAE/RMSE/R²;
- station, hour, and district error analysis;
- feature importance and feature-group ablation;
- identical rows and seed across every ablation variant;
- compact tables, figures, data audit, and run manifest;
- synthetic fixture tests and GitHub Actions.

Still open:

- raw public data redistribution and license documentation;
- exact reproduction of the missing report parquet snapshot;
- rolling-origin evaluation over several test months;
- multi-step forecasting without observed within-horizon lags;
- event, transit disruption, and real dispatch/rebalancing outcome data.
