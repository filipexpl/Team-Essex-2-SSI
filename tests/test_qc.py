"""scripts/qc.py: channel names auto-detected, verdict flags no-contact recordings."""

import importlib.util
from pathlib import Path

import numpy as np
from test_board import fake_bb

spec = importlib.util.spec_from_file_location(
    "qc_script", Path(__file__).resolve().parents[1] / "scripts" / "qc.py"
)
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)


def test_emg_levels_removes_mains_and_keeps_emg():
    fs, rng = 1000.0, np.random.default_rng(0)
    t = np.arange(int(4 * fs)) / fs
    mains = 500 * np.sin(2 * np.pi * 50 * t)
    x = np.vstack([mains + 10 * rng.standard_normal(t.size), mains + 0.0 * t + 3000])
    emg, net = qc.emg_levels(x, fs)
    assert 5 < emg[0] < 15  # broadband EMG survives, 50 Hz removed
    assert emg[1] < 5 and net[1] > 300  # pure mains + DC offset -> no EMG


def test_verdict():
    names = ["F1", "F2"]
    assert qc.verdict(np.array([10, 12]), np.array([60, 15]), names) == []
    msgs = qc.verdict(np.array([700, 0.5]), np.array([720, 0.5]), names)
    assert any("F1: noisy" in m for m in msgs) and any("F2: flat" in m for m in msgs)
    assert any("REF/BIAS" in m for m in msgs)


def test_main_runs_with_any_channel_names(monkeypatch):
    monkeypatch.setattr(qc.time, "sleep", lambda s: None)
    rc = qc.main(["--seconds", "2"], bb_module=fake_bb([]), prompt=lambda msg: "")
    assert rc in (0, 1)
