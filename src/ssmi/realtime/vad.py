"""Articulation onset detection on facial EMG (energy-based voice-activity detection).

The envelope is the mean over channels of the squared, band-passed EMG, smoothed with a moving
RMS window. A threshold is learned from rest (median + k * MAD of the envelope). The same
functions are used at calibration (to learn where articulation starts inside the training
windows) and live (to find articulation in the stream), so both are cut the same way.
"""

from __future__ import annotations

import numpy as np

from ssmi.preprocessing import bandpass, notch


def envelope(x: np.ndarray, fs: float, smooth_s: float = 0.05, lo: float = 20.0) -> np.ndarray:
    """x: (channels, samples) -> (samples,) RMS envelope of the EMG band."""
    y = bandpass(notch(x, fs), fs, lo, 0.45 * fs)
    p = (y**2).mean(axis=0)
    k = max(1, int(smooth_s * fs))
    return np.sqrt(np.convolve(p, np.ones(k) / k, mode="same"))


def threshold_from_rest(rest_env: np.ndarray, k: float = 6.0) -> float:
    med = float(np.median(rest_env))
    mad = float(np.median(np.abs(rest_env - med))) + 1e-12
    return med + k * 1.4826 * mad


def first_onset(
    env: np.ndarray, thr: float, fs: float, min_on_s: float = 0.05, start: int = 0
) -> int | None:
    """Index of the first sample >= start where env stays above thr for min_on_s."""
    above = env > thr
    need = max(1, int(min_on_s * fs))
    run = 0
    for i in range(start, len(above)):
        run = run + 1 if above[i] else 0
        if run >= need:
            return i - need + 1
    return None


def trial_onset_offset(X: np.ndarray, fs: float, k: float = 6.0) -> float:
    """Median onset (seconds from window start) of articulation in calibration trials.

    The rest level is taken from the quietest 20 % of each trial's envelope."""
    offs = []
    for x in X:
        env = envelope(x, fs)
        quiet = np.sort(env)[: max(5, len(env) // 5)]
        o = first_onset(env, threshold_from_rest(quiet, k), fs)
        if o is not None:
            offs.append(o / fs)
    return float(np.median(offs)) if offs else 0.3
