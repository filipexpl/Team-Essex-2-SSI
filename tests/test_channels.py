"""Channel selection on the real EXG8 channel list reported by BrainAccess Board (2026-11-14)."""

import numpy as np
import pytest

from ssmi.acquisition.channels import select_channels
from ssmi.datasets import load_own, load_session

EXG8 = (
    [f"F{i}" for i in range(1, 9)]
    + [f"contact_F{i}" for i in range(1, 9)]
    + [
        "Accel_x",
        "Accel_y",
        "Accel_z",
        "Gyro_x",
        "Gyro_y",
        "Gyro_z",
        "Sample",
        "Streaming",
        "Battery",
    ]
)


def _session(seed, n=12, fs=1000.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(EXG8), 300)) * 100 + 5000  # EMG-like with DC offset
    X[:, 8:16] = 1e6  # contact flags, constant
    X[:, 22] = np.arange(300) * 1e9  # sample counter
    X[:, 23] = 1e6  # streaming flag
    return X, np.repeat(np.arange(6), n // 6)


def test_auto_keeps_only_electrodes():
    X, _ = _session(0)
    idx = select_channels(EXG8, None, X)
    assert [EXG8[i] for i in idx] == [f"F{i}" for i in range(1, 9)]


def test_explicit_and_regex_and_missing():
    assert select_channels(EXG8, "F1,F3") == [0, 2]
    assert len(select_channels(EXG8, "re:^F[1-8]$")) == 8
    with pytest.raises(KeyError):
        select_channels(EXG8, "F9")


def test_constant_electrode_is_dropped():
    X, _ = _session(1)
    X[:, 3] = 7000.0  # disconnected electrode: flat
    names = [EXG8[i] for i in select_channels(EXG8, None, X)]
    assert "F4" not in names and len(names) == 7


def test_old_25_channel_sessions_load_as_8_and_pool(tmp_path):
    for k in range(2):
        X, y = _session(k)
        np.savez(
            tmp_path / f"P01_s{k}.npz",
            X=X,
            y=y,
            fs=1000.0,
            ch_names=np.array(EXG8),
            classes=np.array(["yes", "no", "up", "down", "left", "right"]),
            groups=np.zeros(len(y), int),
            participant="P01",
        )
    s = load_session(tmp_path / "P01_s0.npz")
    assert s["X"].shape[1] == 8 and s["ch_names"][0] == "F1"
    rec = load_own(root=tmp_path)
    assert rec.X.shape[1:] == (8, 300) and len(np.unique(rec.groups)) == 2
    rec2 = load_own(files=[tmp_path / "P01_s1.npz"])
    assert len(rec2.y) == 12
