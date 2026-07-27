# Notebook policy

The original unexecuted Colab notebook duplicated the training script and depended
on a private Google Drive path. It was removed to avoid two drifting sources of
truth.

The canonical workflow is now script-first:

```bash
PYTHONPATH=src python -m seoul_bike_forecasting.cli prepare --help
PYTHONPATH=src python -m seoul_bike_forecasting.cli run --help
```

Compact executed outputs are committed under `results/actual_202604/`. A future
notebook should read those artifacts for exploration rather than reimplement the
training pipeline.
