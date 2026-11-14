"""Euclidean Alignment (He & Wu, IEEE TBME 2020).

Whitens each subject/session by the inverse square root of its mean spatial covariance, so
that every group's mean covariance becomes the identity. Needs NO labels from the new user,
only some unlabeled trials (e.g. the rest baseline or the calibration trials themselves).

Works for EEG (subject differences) and facial sEMG (electrode re-attachment, gain drift).
"""

import numpy as np
from scipy.linalg import fractional_matrix_power


def mean_cov(X: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    Xc = X - X.mean(axis=-1, keepdims=True)
    C = np.einsum("nct,ndt->cd", Xc, Xc) / (X.shape[0] * X.shape[-1])
    return C + reg * np.trace(C) / len(C) * np.eye(len(C))


class EuclideanAligner:
    """Fit on unlabeled trials of ONE group, then transform that group's trials."""

    def fit(self, X: np.ndarray) -> "EuclideanAligner":
        self.R_inv_sqrt_ = np.real(fractional_matrix_power(mean_cov(X), -0.5))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.einsum("cd,ndt->nct", self.R_inv_sqrt_, X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def align_by_group(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Align every group separately (used for pooled source data)."""
    out = np.empty_like(X)
    for g in np.unique(groups):
        m = groups == g
        out[m] = EuclideanAligner().fit_transform(X[m])
    return out
