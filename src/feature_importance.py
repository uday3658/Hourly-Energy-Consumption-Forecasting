"""
Permutation feature importance, mirroring dm-practicle.ipynb.

For each feature, shuffle its values across the test set (keeping other
features intact), re-score the model, and treat the R2 drop as that
feature's importance.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


def permutation_importance(model, X_test, y_test, feature_names: list) -> dict:
    """Compute permutation importance (R2 drop) for every feature."""
    baseline = r2_score(y_test, model.predict(X_test))
    importances = {}

    for i, feat in enumerate(feature_names):
        X_perm = X_test.copy()
        np.random.shuffle(X_perm[:, :, i])  # shuffle feature i across all timesteps
        score = r2_score(y_test, model.predict(X_perm))
        importances[feat] = baseline - score  # drop = importance

    return importances


def plot_feature_importance(rnn_imp: dict, lstm_imp: dict, save_path: str = None):
    """Plot RNN vs LSTM feature importance side-by-side as horizontal bar charts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, imp, title in zip(axes, [rnn_imp, lstm_imp], ["RNN", "LSTM"]):
        sorted_imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
        ax.barh(list(sorted_imp.keys()), list(sorted_imp.values()), color="steelblue")
        ax.set_title(f"{title} - Feature Importance (R\u00b2 drop)")
        ax.set_xlabel("Importance (R\u00b2 drop when shuffled)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
