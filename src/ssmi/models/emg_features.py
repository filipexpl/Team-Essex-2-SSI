"""Silent speech baseline: classic time-domain EMG features on sub-windows -> LDA.

Features per channel per sub-window: RMS, MAV, waveform length, zero crossings,
slope sign changes. Sub-windows keep coarse temporal order of articulation.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ssmi.preprocessing import bandpass, notch


class EMGPreproc(BaseEstimator, TransformerMixin):
    def __init__(self, fs=250.0, lo=20.0, hi=450.0, mains=50.0):
        self.fs, self.lo, self.hi, self.mains = fs, lo, hi, mains

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return bandpass(notch(X, self.fs, self.mains), self.fs, self.lo, self.hi)


class TDFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, n_windows=5, thr=0.01):
        self.n_windows, self.thr = n_windows, thr

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for seg in np.array_split(X, self.n_windows, axis=-1):
            d = np.diff(seg, axis=-1)
            rms = np.sqrt((seg**2).mean(-1))
            mav = np.abs(seg).mean(-1)
            wl = np.abs(d).sum(-1)
            zc = (np.diff(np.sign(seg), axis=-1) != 0).mean(-1)
            ssc = (np.diff(np.sign(d), axis=-1) != 0).mean(-1)
            feats += [np.log(rms + 1e-9), np.log(mav + 1e-9), np.log(wl + 1e-9), zc, ssc]
        return np.concatenate(feats, axis=1)


def td_features_lda(fs: float):
    return make_pipeline(
        EMGPreproc(fs=fs),
        TDFeatures(),
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
