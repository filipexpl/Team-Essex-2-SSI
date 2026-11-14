"""Model registry. Every model is a sklearn-compatible estimator taking X (trials, ch, samples).

Add a model: create a module here, then register it in REGISTRY with a factory(fs) -> estimator.
"""

from collections.abc import Callable

from sklearn.base import BaseEstimator

from ssmi.models.csp import csp_lda
from ssmi.models.emg_features import td_features_lda

REGISTRY: dict[str, Callable[[float], BaseEstimator]] = {
    "csp_lda": csp_lda,
    "td_features_lda": td_features_lda,
}


def get_model(name: str, fs: float) -> BaseEstimator:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](fs)
