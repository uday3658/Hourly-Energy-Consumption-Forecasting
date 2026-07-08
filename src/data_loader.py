"""
Data ingestion + sequence (windowing) creation for time series modeling.

Mirrors the logic in dm-practicle.ipynb:
- read_raw_data(): loads DOM_hourly.csv
- load_data(): turns the dataframe into (X, y) sliding-window sequences
  and splits them into train/test sets (chronological split, no shuffling).
"""

import numpy as np
import pandas as pd


def read_raw_data(raw_data_path: str) -> pd.DataFrame:
    """Read the raw hourly energy consumption CSV."""
    df = pd.read_csv(raw_data_path)
    return df


def load_data(df: pd.DataFrame, seq_len: int, target_col: str, split_ratio: float = 0.8):
    """
    Build sliding-window sequences from a dataframe for RNN/LSTM training.

    For each timestep i, X contains the previous `seq_len` rows (all features)
    and y contains the target value at timestep i.

    Args:
        df: dataframe with datetime index and all engineered features.
        seq_len: number of past timesteps to use as input window.
        target_col: name of the column to predict.
        split_ratio: fraction of samples used for training (chronological split).

    Returns:
        X_train, y_train, X_test, y_test as numpy arrays.
    """
    X = []
    y = []

    data = df.values
    target_index = df.columns.get_loc(target_col)

    for i in range(seq_len, len(df)):
        X.append(data[i - seq_len:i])       # all features
        y.append(data[i, target_index])     # only target

    X = np.array(X)
    y = np.array(y)

    split = int(split_ratio * len(X))

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    return X_train, y_train, X_test, y_test
