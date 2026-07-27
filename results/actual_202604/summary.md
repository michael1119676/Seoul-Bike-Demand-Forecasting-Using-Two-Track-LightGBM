# Reproducible experiment summary

This file is generated from the same chronological split for every model.

| model | target | MAE | RMSE | R2 | rows |
|---|---|---|---|---|---|
| lightgbm | net_flow | 1.2589 | 2.1365 | 0.4139 | 2001000 |
| historical_mean | net_flow | 1.3424 | 2.2914 | 0.3258 | 2001000 |
| lag_168h | net_flow | 1.6264 | 2.6763 | 0.0803 | 2001000 |
| lag_24h | net_flow | 1.7089 | 2.8688 | -0.0568 | 2001000 |
| lightgbm | rent_count | 1.0074 | 1.8225 | 0.7130 | 2001000 |
| historical_mean | rent_count | 1.2539 | 2.3304 | 0.5307 | 2001000 |
| lag_168h | rent_count | 1.3379 | 2.4454 | 0.4833 | 2001000 |
| lag_24h | rent_count | 1.4113 | 2.6608 | 0.3883 | 2001000 |
| lightgbm | return_count | 0.9660 | 1.7281 | 0.7520 | 2001000 |
| historical_mean | return_count | 1.2038 | 2.2939 | 0.5631 | 2001000 |
| lag_168h | return_count | 1.2702 | 2.3386 | 0.5459 | 2001000 |
| lag_24h | return_count | 1.3398 | 2.5802 | 0.4472 | 2001000 |

## Improvement over baselines

- `net_flow`: LightGBM MAE 1.2589; 6.22% vs best baseline `historical_mean`.
- `rent_count`: LightGBM MAE 1.0074; 19.66% vs best baseline `historical_mean`.
- `return_count`: LightGBM MAE 0.9660; 19.75% vs best baseline `historical_mean`.

## Evaluation protocol

- Forecast: rolling one-hour-ahead.
- At hour `t`, observed targets through `t-1` are available.
- Lag-24 and lag-168 baselines and LightGBM use the same test rows.
- Train rows loaded: 21,946,693.
- Validation rows loaded: 2,066,880.
- Test rows loaded: 2,001,000.

These metrics are not valid for a frozen multi-hour batch forecast, because that protocol would require recursive or forecasted lag values.
