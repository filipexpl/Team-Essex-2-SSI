"""First thing to run on the hardware: checks the BrainAccess Board connection and prints what
the protocol needs to know (fs, channels, units, database lag, marker support).

  python scripts/board_check.py                 # live: Board running, amplifier connected
  python scripts/board_check.py --db file.db    # inspect a recorded Board database

Paste the output into docs/hardware.md ("Do potwierdzenia").
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssmi.acquisition.board import BoardControl, BoardSource, load_board_db  # noqa: E402


def describe(raw, label: str) -> None:
    x = raw.get_data()
    types = raw.get_channel_types() if hasattr(raw, "get_channel_types") else ["?"] * len(x)
    print(
        f"[{label}] fs={raw.info['sfreq']} Hz, {len(raw.ch_names)} channels, "
        f"{x.shape[1]} samples ({x.shape[1] / raw.info['sfreq']:.1f} s)"
    )
    for name, t, row in zip(raw.ch_names, types, x, strict=True):
        print(
            f"   {name:<12} type={t:<5} RMS={np.sqrt(np.mean(row**2)):.3g} (raw units)"
            f"  p2p={np.ptp(row):.3g}"
        )
    ann = getattr(raw, "annotations", None)
    print(
        f"   annotations: {len(ann) if ann is not None else 'n/a'}"
        + (f" e.g. {list(ann.description[:5])}" if ann is not None and len(ann) else "")
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args(argv)

    if a.db:
        for dev, raw in load_board_db(a.db).items():
            describe(raw, f"{a.db} / {dev}")
        return

    ctl = BoardControl()
    print("test:", ctl.send("test"))
    print("status:", ctl.send("status"))
    print("devices:", ctl.send("get_devices"))

    src = BoardSource(markers=True)
    print(
        f"\nselected device {src.device_id} of {src.devices}; fs={src.fs} Hz; "
        f"signal channels={src.ch_names}"
    )
    print(f"markers attached to Board: {bool(src.markers and src.markers.attached)}")
    describe(src.db.get_mne(duration=a.seconds)[src.device_id], "live")

    # database lag: how much new data appears per second of wall-clock time
    n0 = src.db.get_mne()[src.device_id].n_times
    t0 = time.time()
    time.sleep(3)
    n1 = src.db.get_mne()[src.device_id].n_times
    rate = (n1 - n0) / (time.time() - t0)
    print(
        f"\nsamples arriving: {rate:.1f}/s vs nominal {src.fs} Hz "
        f"({100 * rate / src.fs:.1f} %; < 98 % means packet loss)"
    )
    print(
        "Units: MNE stores volts; BoardSource multiplies by 1e6. If RMS above is ~1e-5..1e-4, "
        "the data are in volts (expected); if ~10..100, they are already in microvolts: "
        "use BoardSource(to_microvolts=False)."
    )


if __name__ == "__main__":
    main()
