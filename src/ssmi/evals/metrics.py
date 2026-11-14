import math
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix


def itr_bits_per_min(acc: float, n_classes: int, trial_sec: float) -> float:
    """Wolpaw information transfer rate."""
    p, n = min(max(acc, 1e-9), 1 - 1e-9), n_classes
    if p <= 1 / n:
        return 0.0
    bits = math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))
    return bits * 60.0 / trial_sec


def latency_ms(model, X_one: np.ndarray, reps: int = 20) -> float:
    t0 = time.perf_counter()
    for _ in range(reps):
        model.predict(X_one)
    return (time.perf_counter() - t0) / reps * 1000


def summarize(y_true, y_pred, n_classes, trial_sec) -> dict:
    acc = balanced_accuracy_score(y_true, y_pred)
    return {
        "balanced_accuracy": float(acc),
        "chance": 1.0 / n_classes,
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "itr_bits_min": itr_bits_per_min(acc, n_classes, trial_sec),
        "confusion": confusion_matrix(y_true, y_pred, labels=range(n_classes)).tolist(),
    }
