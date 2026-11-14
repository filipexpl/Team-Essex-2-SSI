"""Live decoding with a calibrated user profile.

python scripts/live.py --task ss --user P07 --device lsl          # Enter = decode one window
python scripts/live.py --task ss --user P07 --device synthetic --n 20   # automatic dry run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssmi.acquisition.sources import open_source  # noqa: E402
from ssmi.calibration import UserProfile  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mi", "ss"], required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--device", choices=["synthetic", "board", "lsl"], default="synthetic")
    ap.add_argument("--n", type=int, default=0, help="non-interactive: N random cued trials")
    a = ap.parse_args(argv)

    prof = UserProfile.load(a.user, a.task)
    proto = yaml.safe_load((ROOT / "configs" / "vocab.yaml").read_text())["protocol"]
    proto = proto["motor_imagery" if a.task == "mi" else "silent_speech"]
    src = open_source(
        a.device,
        a.task,
        prof.class_names,
        realtime=a.device in ("lsl", "board"),
        channels=prof.ch_names if a.device == "board" else None,
    )
    if src.fs != prof.fs:
        raise SystemExit(f"Device fs {src.fs} != profile fs {prof.fs}. Re-calibrate.")
    print(f"Profil {a.user}/{a.task}: klasy {prof.class_names}, model {prof.model_name}")

    def decode(cue=None):
        x = src.read(proto["window"], cue=cue)
        if a.task == "mi":
            x = x[:, int(proto["epoch_start"] * src.fs) : int(proto["epoch_end"] * src.fs)]
        t0 = time.perf_counter()
        label = prof.predict_label(x)
        return label, (time.perf_counter() - t0) * 1000

    if a.n:  # dry run with known cues -> accuracy
        rng = np.random.default_rng(1)
        hits = 0
        for _ in range(a.n):
            cue = str(rng.choice(prof.class_names))
            label, ms = decode(cue)
            hits += label == cue
            print(f"  cue={cue:<12} → {label:<12} {'✓' if label == cue else '✗'}  {ms:.1f} ms")
        print(f"trafność: {hits / a.n:.2f}")
        return
    while True:
        if input("\nEnter = dekoduj, q = wyjście > ").strip().lower() == "q":
            break
        print(" ● TERAZ", flush=True)
        label, ms = decode()
        print(f"  → {label.upper()}   ({ms:.1f} ms)")


if __name__ == "__main__":
    main()
