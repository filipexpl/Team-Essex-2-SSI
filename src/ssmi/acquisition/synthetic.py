"""Synthetic data so that tests and CI run without hardware or downloads.

MI: class-specific mu/beta ERD on C3/Cz/C4-like channels.
SS: class-specific bursts of band-limited 'EMG' on different facial channels.
"""

import numpy as np

from ssmi import MI_CLASSES
from ssmi.acquisition import Recording

MINI_CHANNELS = ["F3", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]
FACE_CHANNELS = [f"EMG{i}" for i in range(1, 9)]


def _pink_noise(rng, shape):
    white = rng.standard_normal(shape)
    f = np.fft.rfftfreq(shape[-1])
    f[0] = f[1]
    return np.fft.irfft(np.fft.rfft(white) / np.sqrt(f), n=shape[-1])


def make_mi(n_per_class=40, n_subjects=3, fs=250.0, dur=3.0, seed=0) -> Recording:
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    t = np.arange(n) / fs
    # ERD pattern: amplitude multiplier of mu rhythm per channel (C3, Cz, C4 indices 2,3,4)
    patterns = {
        0: {4: 0.25},  # left hand -> ERD over right hemisphere (C4)
        1: {2: 0.25},  # right hand -> ERD C3
        2: {2: 0.4, 4: 0.4},  # both hands -> bilateral
        3: {3: 0.2},  # feet -> Cz
    }
    X, y, g = [], [], []
    for s in range(n_subjects):
        mu_f = 10 + rng.uniform(-1.5, 1.5)
        for c in range(len(MI_CLASSES)):
            for _ in range(n_per_class):
                x = _pink_noise(rng, (8, n)) * 2.0
                phase = rng.uniform(0, 2 * np.pi, 8)
                amp = np.ones(8) * 3.0
                for ch, mult in patterns[c].items():
                    amp[ch] *= mult
                x += amp[:, None] * np.sin(2 * np.pi * mu_f * t + phase[:, None])
                X.append(x)
                y.append(c)
                g.append(s)
    return Recording(
        np.asarray(X),
        np.asarray(y),
        fs,
        MINI_CHANNELS,
        MI_CLASSES,
        np.asarray(g),
        meta={"source": "synthetic"},
    )


def make_ss(
    words=("tak", "nie", "start", "stop", "lewo", "prawo"),
    n_per_class=30,
    n_sessions=3,
    fs=250.0,
    dur=1.5,
    seed=0,
) -> Recording:
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    t = np.arange(n) / fs
    X, y, g = [], [], []
    # each word = a sequence of (channel, onset, duration) muscle activations
    proto = {
        w: [(int(rng.integers(8)), rng.uniform(0.1, 0.9), rng.uniform(0.15, 0.4)) for _ in range(3)]
        for w in words
    }
    for sess in range(n_sessions):
        gain = rng.uniform(0.7, 1.3, 8)  # electrode re-attachment between sessions
        for c, w in enumerate(words):
            for _ in range(n_per_class):
                x = rng.standard_normal((8, n)) * 0.5
                jitter = rng.uniform(-0.08, 0.08)
                for ch, on, d in proto[w]:
                    env = np.exp(-0.5 * ((t - on - jitter) / (d / 2)) ** 2)
                    x[ch] += 4 * env * rng.standard_normal(n)
                X.append(x * gain[:, None])
                y.append(c)
                g.append(sess)
    return Recording(
        np.asarray(X),
        np.asarray(y),
        fs,
        FACE_CHANNELS,
        list(words),
        np.asarray(g),
        meta={"source": "synthetic"},
    )
