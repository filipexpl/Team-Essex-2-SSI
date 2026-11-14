"""Simulated continuous stream (no hardware) for the real-time loop and its tests.

Silent speech: rest, then words from the simulated person's trial pool at random gaps.
Motor imagery: consecutive imagery blocks of `block_s` seconds per class.
Time is virtual (`advance`) or follows the wall clock (`realtime=True`).
"""

from __future__ import annotations

import time

import numpy as np

from ssmi.acquisition.sources import SyntheticSource


class SimulatedStream:
    def __init__(
        self,
        task: str,
        class_names: list[str],
        fs: float = 500.0,
        seconds: float = 60,
        seed: int = 123,
        realtime: bool = False,
        block_s: float = 6.0,
    ):
        self.fs, self.task, self.realtime = fs, task, realtime
        src = SyntheticSource(task, class_names, fs=fs, seed=seed)
        self.ch_names, self.n_channels = src.ch_names, src.n_channels
        pool, rng = src._pool, np.random.default_rng(seed + 1)
        self.truth: list[tuple[float, str]] = []  # (time of segment start, label)
        parts, t = [], 0.0

        def add(x, label=None):
            nonlocal t
            if label:
                self.truth.append((t, label))
            parts.append(x)
            t += x.shape[1] / fs

        def rest(sec):  # quiet tail of pool trials = same noise and mixing as the trials
            n, out = int(sec * fs), []
            while sum(o.shape[1] for o in out) < n:
                out.append(pool.X[rng.integers(len(pool.y))][:, int(2.6 * fs) :])
            return np.concatenate(out, axis=1)[:, :n]

        if task == "ss":
            add(rest(6.0))
            while t < seconds:
                c = int(rng.integers(len(class_names)))
                trial = pool.X[rng.choice(np.flatnonzero(pool.y == c))]
                add(trial[:, : int(1.5 * fs)], class_names[c])
                add(rest(rng.uniform(1.5, 3.0)))
        else:  # motor imagery: back-to-back imagery blocks (the loop's rest phase eats the start)
            while t < seconds:
                c = int(rng.integers(len(class_names)))
                trial = pool.X[rng.choice(np.flatnonzero(pool.y == c))]
                add(trial[:, : int(block_s * fs)], class_names[c])
        self.data = np.concatenate(parts, axis=1)
        self.duration = self.data.shape[1] / fs
        self._t0 = time.time()
        self.now = 0.0

    def clock(self) -> float:
        return time.time() - self._t0 if self.realtime else self.now

    def sleep(self, s: float) -> None:
        if self.realtime:
            time.sleep(s)
        else:
            self.now += s

    def latest(self, seconds: float) -> np.ndarray:
        end = min(int(self.clock() * self.fs), self.data.shape[1])
        start = max(0, end - int(seconds * self.fs))
        return self.data[:, start:end]
