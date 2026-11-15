"""Facial EMG signal check before calibration (macOS, Windows, Linux).

Reads the live BrainAccess Board stream, picks the electrode channels automatically (works with
any channel names: F1..F8 on one computer, F3..O2 on another) and prints, per channel, the EMG
level (50 Hz + harmonics removed, 20-450 Hz) for a relaxed face and a jaw clench.

  python scripts/qc.py                 # Board running, EXG8 connected (or: make qc)
  python scripts/qc.py --seconds 8 --device EXG8

Good signal: relaxed 5-30 uV on every channel, clench several times higher on jaw channels.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):  # Windows: cp1250 console cannot print µ
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ssmi.preprocessing import bandpass, notch  # noqa: E402

REST_MAX_UV = 30.0  # relaxed face above this: poor contact or mains pickup
FLAT_UV = 1.0  # below this: flat / disconnected channel
CLENCH_RATIO = 2.0  # clench / rest below this on every channel: no EMG is reaching the amplifier


def emg_levels(x_uv: np.ndarray, fs: float, edge_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(channels, samples) in uV -> (EMG RMS after notch + 20-450 Hz, mains 45-55 Hz RMS)."""
    x = x_uv - x_uv.mean(axis=1, keepdims=True)
    e = int(edge_s * fs)
    emg = bandpass(notch(x, fs, 50.0), fs, 20.0, 450.0)[:, e:-e]
    mains = bandpass(x, fs, 45.0, 55.0)[:, e:-e]
    return emg.std(axis=1), mains.std(axis=1)


def verdict(rest: np.ndarray, clench: np.ndarray, names: list[str]) -> list[str]:
    msgs = []
    for n, r in zip(names, rest, strict=True):
        if r < FLAT_UV:
            msgs.append(f"{n}: flat ({r:.1f} uV) - lead not connected?")
        elif r > REST_MAX_UV:
            msgs.append(f"{n}: noisy at rest ({r:.0f} uV) - re-seat electrode, add gel")
    if np.max(clench / np.maximum(rest, 1e-9)) < CLENCH_RATIO:
        msgs.append(
            "no channel rises on jaw clench: check REF/BIAS contact and sockets first, "
            "then that the green contact dots in Board are on"
        )
    return msgs


def measure(src, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    time.sleep(seconds)
    return emg_levels(src.latest(seconds), src.fs)


def main(argv=None, bb_module=None, prompt=input) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--device", default=None, help="device id or part of its name, e.g. EXG8")
    ap.add_argument("--channels", default=None, help="default: auto (electrode channels only)")
    a = ap.parse_args(argv)

    from ssmi.acquisition.board import BoardSource

    src = BoardSource(
        device=a.device, channels=a.channels, markers=False, lag_s=0.0, bb_module=bb_module
    )
    names = src.ch_names
    print(f"device {src.device_id}: fs={src.fs} Hz, channels={names}")
    levels = {}
    for key, label in [("rest", "RELAX face"), ("clench", "CLENCH jaw and hold")]:
        prompt(f"{label}, then press Enter (measuring {a.seconds:.0f} s)...")
        emg, mains = measure(src, a.seconds)
        levels[key] = emg
        print(
            f"  {label:<20} EMG  : "
            + " ".join(f"{n}={v:.0f}" for n, v in zip(names, emg, strict=True))
        )
        print(
            f"  {'':<20} 50 Hz: "
            + " ".join(f"{n}={v:.0f}" for n, v in zip(names, mains, strict=True))
        )
    msgs = verdict(levels["rest"], levels["clench"], names)
    print("\nOK: signal looks usable, go on to calibration." if not msgs else "\nFix first:")
    for m in msgs:
        print("  -", m)
    return 1 if msgs else 0


if __name__ == "__main__":
    sys.exit(main())
