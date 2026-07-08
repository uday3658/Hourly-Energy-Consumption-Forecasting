"""
Evaluation utilities, mirroring dm-practicle.ipynb:
- compute_metrics(): R2 and RMSE for a fitted model
- plot_predictions(): actual vs predicted line chart, optionally saved to disk
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    """Return R2 and RMSE for a set of predictions."""
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2": float(r2), "rmse": rmse}


def plot_predictions(test, predicted, title: str, save_path: str = None):
    """Plot actual vs predicted values. Saves to save_path if given, else shows the plot."""
    plt.figure(figsize=(16, 4))
    plt.plot(test, color="blue", label="Actual power consumption data")
    plt.plot(predicted, alpha=0.7, color="orange", label="Predicted power consumption data")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Normalized power consumption scale")
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_scores(scores: dict, scores_file_path: str):
    """Save a dict of metrics (e.g. {'rnn': {...}, 'lstm': {...}}) to a JSON file."""
    with open(scores_file_path, "w") as f:
        json.dump(scores, f, indent=4)
