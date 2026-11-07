"""Full evaluation suite (blueprint Section 8)."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              f1_score, precision_score, recall_score)


def binary_metrics(y_true, y_prob, mask=None) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    if mask is not None:
        m = np.asarray(mask).ravel().astype(bool)
        y_true, y_prob = y_true[m], y_prob[m]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "log_loss": float(-np.mean(
            y_true * np.log(np.clip(y_prob, 1e-12, 1))
            + (1 - y_true) * np.log(np.clip(1 - y_prob, 1e-12, 1)))),
    }


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = y_prob[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / len(y_prob)) * abs(acc - conf)
    return float(ece)