"""Motor imagery baseline: 8-30 Hz bandpass -> one-vs-rest CSP log-variance -> LDA.

Pure numpy/scipy so it runs in CI without MNE.
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ssmi.preprocessing import bandpass


class Bandpass(BaseEstimator, TransformerMixin):
    def __init__(self, fs=250.0, lo=8.0, hi=30.0):
        self.fs, self.lo, self.hi = fs, lo, hi

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return bandpass(X, self.fs, self.lo, self.hi)


def _cov(X):
    C = np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
    return C / np.trace(C, axis1=1, axis2=2)[:, None, None]


class OvRCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_filters=4, reg=1e-3):
        self.n_filters, self.reg = n_filters, reg

    def fit(self, X, y):
        covs = _cov(X)
        self.classes_ = np.unique(y)
        filters = []
        for c in self.classes_:
            Ca, Cb = covs[y == c].mean(0), covs[y != c].mean(0)
            Cb = Cb + self.reg * np.trace(Cb) * np.eye(len(Cb))
            w, V = eigh(Ca, Ca + Cb)
            idx = np.argsort(w)
            half = self.n_filters // 2
            pick = np.r_[idx[:half], idx[-half:]]
            filters.append(V[:, pick].T)
        self.W_ = np.concatenate(filters)
        return self

    def transform(self, X):
        Z = np.einsum("fc,nct->nft", self.W_, X)
        return np.log(Z.var(axis=-1) + 1e-12)


def csp_lda(fs: float):
    return make_pipeline(
        Bandpass(fs=fs),
        OvRCSP(n_filters=4),
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
