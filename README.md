# Energy Consumption Forecasting (RNN vs LSTM)

Forecasting hourly power consumption (megawatts) for the Dominion (DOM) region
using deep learning (SimpleRNN and LSTM), based on the
[PJM Hourly Energy Consumption dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).

## Problem

Given the past 24 hours of consumption + time features, predict the next
hour's power consumption (`DOM_MW`). This is a univariate time-series
regression problem tackled with two recurrent architectures so their
performance can be compared.

## Pipeline

```
Raw CSV → datetime index → time features (hour/dayofweek/month)
→ cyclical encoding (sin/cos) → normalize target (MinMaxScaler)
→ rolling mean (24h) → sliding-window sequences (seq_len=24)
→ train/test split (chronological, 80/20)
→ train SimpleRNN (3 layers, 40 units) → train LSTM (3 layers, 40 units)
→ evaluate (R², RMSE) → permutation feature importance
```

## Project Structure

```
energy-consumption-forecasting/
├── data/
│   ├── raw/                  # place DOM_hourly.csv here
│   └── processed/            # transformed_data.csv + scaler.pkl generated here
├── notebooks/
│   └── dm-practicle.ipynb    # original exploration notebook
├── config/
│   └── config.yaml           # all paths + hyperparameters
├── src/
│   ├── config_reader.py      # loads config.yaml
│   ├── data_loader.py        # read_raw_data(), load_data() (sequence builder)
│   ├── preprocessing.py      # datetime index, cyclical features, normalize, rolling mean
│   ├── models.py              # build_rnn_model(), build_lstm_model()
│   ├── train.py                # standalone training script
│   ├── evaluate.py             # compute_metrics(), plot_predictions(), save_scores()
│   └── feature_importance.py   # permutation_importance(), plot_feature_importance()
├── models/                   # saved trained models (.keras)
├── outputs/
│   ├── figures/              # prediction + feature importance plots
│   └── metrics.json          # R²/RMSE for both models
├── main.py                   # runs the entire pipeline end to end
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the dataset at `data/raw/DOM_hourly.csv` (download from the Kaggle link above).

## Run

Run the full pipeline (ingestion → preprocessing → training → evaluation → feature importance):

```bash
python main.py
```

Or just train the models without evaluation/plots:

```bash
python -m src.train
```

## Results

| Model | R² (test) | Notes |
|-------|-----------|-------|
| SimpleRNN | ~0.95 | 3 stacked layers, 40 units, dropout 0.15 |
| LSTM | ~0.98 | 3 stacked layers, 40 units, dropout 0.15 |

LSTM outperforms SimpleRNN due to its gating mechanism handling longer-term
dependencies better. Scores are logged to `outputs/metrics.json` after each run.

Note: R²/RMSE are computed on the **normalized** (0–1 scaled) target. Use
`src/preprocessing.py::inverse_transform()` with the saved `scaler.pkl` to
convert predictions back to real megawatts if you need interpretable error values.

## Possible Improvements

- Drop redundant raw `hour`/`dayofweek`/`month` columns once cyclically encoded (kept in notebook, currently unused decision).
- Add early stopping / learning-rate scheduling during training.
- Try GRU or a Transformer-based model for comparison.
- Hyperparameter tuning (units, dropout, seq_len) via config sweeps.
