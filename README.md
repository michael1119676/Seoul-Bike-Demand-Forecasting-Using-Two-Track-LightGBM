# Seoul Bike Demand Forecasting

This repository contains the training and evaluation code for a Seoul Bike
station-hour demand forecasting project.

The model predicts two target variables independently:

- `rent_count`: hourly bike rentals at each station
- `return_count`: hourly bike returns at each station

Predicted net flow is then calculated as:

```text
pred_net_flow = pred_return - pred_rent
```

A negative value indicates a potential bike shortage, while a positive value
indicates a potential bike surplus.

## Repository Structure

```text
notebooks/
  01_train_evaluate_two_track_lightgbm.ipynb
src/
  train_evaluate.py
docs/
  report_repository_note.md
requirements.txt
.gitignore
```

## Data

Raw and processed data files are not included because of file size limits.
The code expects monthly parquet feature files with names like:

```text
model_features_202604.parquet
```

Default Colab feature directory:

```text
/content/drive/MyDrive/ML1_데이터/전처리 데이터 최종본/reprocessed_rent_return_v2_strict
```

## Train, Validation, and Test Split

| Dataset | Period | Sampling |
|---|---:|---:|
| Train | 2022-12 to 2025-12 | 30% |
| Validation | 2026-03 | 100% |
| Test | 2026-04 | 100% |

The train sample is applied after monthly feature engineering and lag creation.
Validation and test data are evaluated without sampling.

## How to Run

### Colab Notebook

Open:

```text
notebooks/01_train_evaluate_two_track_lightgbm.ipynb
```

Then run all cells.

### Python Script

```bash
pip install -r requirements.txt
python src/train_evaluate.py \
  --feature-dir "/content/drive/MyDrive/ML1_데이터/전처리 데이터 최종본/reprocessed_rent_return_v2_strict" \
  --output-dir "/content/drive/MyDrive/ML1_데이터/전처리 데이터 최종본/trained_models_v2"
```

## Main Outputs

```text
lgb_rent_model.txt
lgb_return_model.txt
model_comparison.csv
predictions_202604.parquet
feature_importance_rent_count.csv
feature_importance_return_count.csv
top_shortage_stations.csv
top_surplus_stations.csv
run_manifest.json
```

## Notes

- The repository stores code only.
- Large parquet files, model binaries, and raw Seoul Bike data should be stored
  externally, such as in Google Drive.
- The LightGBM models are trained in a two-track structure so rental and return
  patterns can be learned separately.
