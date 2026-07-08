"""
Feature engineering / preprocessing, mirroring dm-practicle.ipynb:
- set_datetime_index(): parse Datetime column into index + extract hour/dayofweek/month
- add_cyclical_features(): sin/cos encode hour, dayofweek, month
- normalize_data(): MinMaxScaler on the target column (and saves the scaler for inverse-transform)
- add_rolling_mean(): 24h rolling mean of the target column
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def set_datetime_index(df: pd.DataFrame, datetime_col: str = "Datetime") -> pd.DataFrame:
    """Set the datetime column as index and extract hour/dayofweek/month."""
    df = df.set_index(datetime_col)
    df.index = pd.to_datetime(df.index)

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encodings for hour, dayofweek, month so cyclical closeness is preserved."""
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def normalize_data(df: pd.DataFrame, target_col: str = "DOM_MW", scaler_path: str = None) -> pd.DataFrame:
    """
    Scale the target column into 0-1 range with MinMaxScaler.
    Optionally saves the fitted scaler to disk so predictions can be
    inverse-transformed back to real MW values later.
    """
    scaler = MinMaxScaler()
    df[target_col] = scaler.fit_transform(df[target_col].values.reshape(-1, 1))

    if scaler_path:
        joblib.dump(scaler, scaler_path)

    return df


def add_rolling_mean(df: pd.DataFrame, target_col: str = "DOM_MW", window: int = 24) -> pd.DataFrame:
    """Add a rolling mean of the target column, back-filling initial NaNs."""
    df["rolling"] = df[target_col].rolling(window).mean()
    df["rolling"] = df["rolling"].bfill()
    return df


def inverse_transform(values: np.ndarray, scaler_path: str) -> np.ndarray:
    """Load a saved scaler and inverse-transform normalized predictions back to real MW."""
    scaler = joblib.load(scaler_path)
    values = np.array(values).reshape(-1, 1)
    return scaler.inverse_transform(values).flatten()


def run_preprocessing_pipeline(df: pd.DataFrame, target_col: str = "DOM_MW", scaler_path: str = None) -> pd.DataFrame:
    """Run the full feature-engineering pipeline used in the notebook, in order."""
    df = set_datetime_index(df)
    df = add_cyclical_features(df)
    df = normalize_data(df, target_col=target_col, scaler_path=scaler_path)
    df = add_rolling_mean(df, target_col=target_col)
    return df
