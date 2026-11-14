"""Signal sources with one interface for scripts:  fs, n_channels, ch_names, read(seconds, cue).

`cue` is only used by SyntheticSource (to fake a class-specific response for dry runs);
real devices ignore it.
"""

from __future__ import annotations

import time

import numpy as np

from ssmi.acquisition import synthetic as syn


class SyntheticSource:
    """Dry-run source: returns trials of a simulated person (no hardware needed)."""

    def __init__(
        self,
        task: str,
        class_names: list[str],
        fs: float = 250.0,
        seed: int = 123,
        realtime: bool = False,
    ):
        self.fs, self.task, self.realtime = fs, task, realtime
        self.class_names = list(class_names)
        self.n_channels = 8
        self.ch_names = syn.MINI_CHANNELS if task == "mi" else syn.FACE_CHANNELS
        self._rng = np.random.default_rng(seed)
        # pre-generate a pool of trials for one new "person"
        if task == "mi":
            rec = syn.make_mi(n_per_class=60, n_subjects=1, fs=fs, dur=6.0, seed=seed, mix=0.8)
            rec.class_names = list(class_names)
        else:
            rec = syn.make_ss(
                words=tuple(class_names),
                n_per_class=60,
                n_sessions=1,
                fs=fs,
                dur=4.0,
                seed=seed,
                mix=0.8,
            )
        self._pool = rec

    def read(self, seconds: float, cue: str | None = None) -> np.ndarray:
        if self.realtime:
            time.sleep(seconds)
        n = int(seconds * self.fs)
        if cue is None or cue not in self.class_names:  # rest
            return self._rng.standard_normal((self.n_channels, n))
        c = self.class_names.index(cue)
        if c >= len(np.unique(self._pool.y)):
            return self._rng.standard_normal((self.n_channels, n))
        idx = self._rng.choice(np.flatnonzero(self._pool.y == c))
        x = self._pool.X[idx]
        if x.shape[1] < n:
            x = np.pad(x, ((0, 0), (0, n - x.shape[1])))
        return x[:, :n]


def open_source(device: str, task: str, class_names: list[str], realtime: bool = False):
    if device == "synthetic":
        return SyntheticSource(task, class_names, realtime=realtime)
    if device == "lsl":
        from ssmi.acquisition.brainaccess_stream import LSLSource

        src = LSLSource(stream_type="EEG")
        src.ch_names = [f"ch{i + 1}" for i in range(src.n_channels)]
        _read = src.read
        src.read = lambda seconds, cue=None: _read(seconds)
        return src
    raise ValueError(f"unknown device {device}")
