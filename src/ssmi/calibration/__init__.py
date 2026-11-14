"""Per-user calibration.

A new person records a short calibration block (rest + K cued trials per class). We then:
1. fit a EuclideanAligner on that person's trials (unsupervised),
2. train a model on  [aligned source data from other people]  +  [aligned calibration trials]
   (mode="pooled"), or on calibration trials only (mode="subject"),
3. save everything needed for live use in a UserProfile.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ssmi.acquisition import Recording
from ssmi.calibration.alignment import EuclideanAligner, align_by_group
from ssmi.models import get_model

PROFILES = Path(__file__).resolve().parents[3] / "profiles"


@dataclass
class UserProfile:
    user_id: str
    task: str
    fs: float
    ch_names: list[str]
    class_names: list[str]
    aligner: EuclideanAligner | None
    model: object
    model_name: str
    mode: str
    n_calib_per_class: int
    calib_cv_accuracy: float | None
    created: float = field(default_factory=time.time)
    onset_offset_s: float | None = None  # silent speech: articulation onset in window

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:  # single trial (channels, samples)
            X = X[None]
        if self.aligner is not None:
            X = self.aligner.transform(X)
        return self.model.predict(X)

    def predict_label(self, x: np.ndarray) -> str:
        return self.class_names[int(self.predict(x)[0])]

    def save(self, root: Path = PROFILES) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.user_id}_{self.task}.pkl"
        path.write_bytes(pickle.dumps(self))
        return path

    @staticmethod
    def load(user_id: str, task: str, root: Path = PROFILES) -> UserProfile:
        return pickle.loads((root / f"{user_id}_{task}.pkl").read_bytes())


def fit_calibrated(
    calib: Recording,
    model_name: str,
    source: Recording | None = None,
    mode: str = "pooled",
    align: bool = True,
    calib_weight: int = 3,
):
    """Return (fitted model, aligner) for one new user/session.

    calib_weight: calibration trials are repeated this many times in the pooled training set
    so a few personal trials are not drowned by thousands of source trials.
    """
    if source is not None and mode == "pooled" and list(source.ch_names) != list(calib.ch_names):
        raise ValueError(
            f"Source channels {source.ch_names} != calibration channels {calib.ch_names}. "
            "Pick matching channels in the source loader."
        )
    aligner = EuclideanAligner().fit(calib.X) if align else None
    Xc = aligner.transform(calib.X) if align else calib.X
    if mode == "subject" or source is None:
        X, y = Xc, calib.y
    elif mode in ("pooled", "source_only"):
        Xs = align_by_group(source.X, source.groups) if align else source.X
        if mode == "source_only":
            X, y = Xs, source.y
        else:
            X = np.concatenate([Xs] + [Xc] * calib_weight)
            y = np.concatenate([source.y] + [calib.y] * calib_weight)
    else:
        raise ValueError(f"unknown mode {mode}")
    model = get_model(model_name, calib.fs).fit(X, y)
    return model, aligner


def calibrate_user(
    user_id: str,
    task: str,
    calib: Recording,
    model_name: str,
    source: Recording | None = None,
    mode: str = "pooled",
    align: bool = True,
    seed: int = 0,
) -> UserProfile:
    """Fit + estimate accuracy on the calibration block (k-fold, within-session => optimistic)."""
    n_per_class = int(np.bincount(calib.y).min())
    cv_acc = None
    if n_per_class >= 3 and (mode == "subject" or source is None):
        k = min(5, n_per_class)
        Xc = EuclideanAligner().fit_transform(calib.X) if align else calib.X
        cv = StratifiedKFold(k, shuffle=True, random_state=seed)
        cv_acc = float(cross_val_score(get_model(model_name, calib.fs), Xc, calib.y, cv=cv).mean())
    model, aligner = fit_calibrated(calib, model_name, source, mode, align)
    onset = None
    if task == "ss":
        from ssmi.realtime.vad import trial_onset_offset

        onset = trial_onset_offset(calib.X, calib.fs)
    return UserProfile(
        user_id,
        task,
        calib.fs,
        list(calib.ch_names),
        list(calib.class_names),
        aligner,
        model,
        model_name,
        mode,
        n_per_class,
        cv_acc,
        onset_offset_s=onset,
    )
