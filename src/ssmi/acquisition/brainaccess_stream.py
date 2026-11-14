"""Live acquisition from BrainAccess devices.

Two paths, both producing a rolling buffer of shape (n_channels, n_samples):
1. LSL  — BrainAccess Board streams to LSL; we read with pylsl. Most robust for a hackathon.
2. SDK  — direct `brainaccess` Python package. API names MUST be verified against the
          installed SDK version (see docs/hardware.md); this module keeps the SDK call
          surface in one place so only this file changes.
"""

from __future__ import annotations

import time

import numpy as np


class LSLSource:
    def __init__(self, stream_type: str = "EEG", name: str | None = None, timeout: float = 10.0):
        from pylsl import StreamInlet, resolve_byprop  # optional dependency

        prop, val = ("name", name) if name else ("type", stream_type)
        streams = resolve_byprop(prop, val, timeout=timeout)
        if not streams:
            raise RuntimeError(f"No LSL stream with {prop}={val}. Is BrainAccess Board streaming?")
        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        self.fs = info.nominal_srate()
        self.n_channels = info.channel_count()

    def read(self, seconds: float) -> np.ndarray:
        n_target = int(self.fs * seconds)
        chunks, got = [], 0
        t0 = time.time()
        while got < n_target and time.time() - t0 < seconds + 5:
            data, _ = self.inlet.pull_chunk(timeout=0.2)
            if data:
                arr = np.asarray(data).T
                chunks.append(arr)
                got += arr.shape[1]
        return np.concatenate(chunks, axis=1)[:, :n_target]


class BrainAccessSDKSource:
    """TODO(hardware-day): fill in after checking `brainaccess` SDK examples
    ("Continuous data acquisition demo"). Keep the same interface as LSLSource:
    attributes `fs`, `n_channels`, method `read(seconds) -> (n_channels, n_samples)`.
    """

    def __init__(self, port: str | None = None):
        raise NotImplementedError(
            "Verify brainaccess SDK API on hardware day; use LSLSource meanwhile."
        )
