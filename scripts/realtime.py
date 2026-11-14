"""Continuous real-time decoding with a calibrated profile (no key presses).

  # real device (BrainAccess Board running, amplifier connected, profile from calibrate.py)
  python scripts/realtime.py --task ss --user P07 --device board
  python scripts/realtime.py --task mi --user P07 --device board --lsl-out
  # without hardware: a simulated person streaming words / imagery
  python scripts/realtime.py --task ss --user SIM --device synthetic --duration 60

Silent speech: articulate one word at a time with a short pause; each detected word prints once.
Motor imagery: the current class prints when it changes and has been stable for a moment.
Decisions are appended as JSON lines to results/realtime_<user>_<task>.jsonl and, with
--lsl-out, published on the LSL stream 'SSMI_Decoded' for a demo UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssmi.calibration import UserProfile  # noqa: E402
from ssmi.realtime import LSLOutput, make_decoder, run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mi", "ss"], required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--device", choices=["synthetic", "board", "lsl"], default="synthetic")
    ap.add_argument("--board-device", default=None)
    ap.add_argument("--duration", type=float, default=None, help="seconds (default: until Ctrl+C)")
    ap.add_argument("--hop", type=float, default=None, help="poll interval (s)")
    ap.add_argument("--min-conf", type=float, default=None, help="confidence to report a class")
    ap.add_argument("--k", type=float, default=6.0, help="SS onset threshold (rest MAD units)")
    ap.add_argument("--lsl-out", action="store_true")
    a = ap.parse_args(argv)

    prof = UserProfile.load(a.user, a.task)
    if a.device == "synthetic":
        from ssmi.acquisition.stream_sim import SimulatedStream

        src = SimulatedStream(
            a.task, prof.class_names, fs=prof.fs, realtime=True, seconds=(a.duration or 120) + 10
        )
    elif a.device == "board":
        from ssmi.acquisition.board import BoardSource

        src = BoardSource(device=a.board_device, markers=False)
    else:
        from ssmi.acquisition.brainaccess_stream import LSLSource

        src = LSLSource()
    if abs(src.fs - prof.fs) > 1e-6:
        raise SystemExit(f"Device fs {src.fs} != profile fs {prof.fs}: re-calibrate.")

    kw = {"min_conf": a.min_conf} if a.min_conf is not None else {}
    if a.task == "ss":
        kw["k"] = a.k
    dec = make_decoder(prof, src.fs, **kw)
    hop = a.hop or (0.1 if a.task == "ss" else 0.25)
    out = LSLOutput() if a.lsl_out else None

    def show(ev):
        print(
            f"  {ev.t:8.1f}s  {ev.label.upper():<14} conf={ev.confidence:.2f}  "
            f"compute={ev.latency_ms:.1f} ms",
            flush=True,
        )
        if out:
            out(ev)

    print(
        f"Profile {a.user}/{a.task}: {prof.class_names} | device={a.device} fs={src.fs} Hz "
        f"| hop={hop}s" + (f" | onset offset={prof.onset_offset_s:.2f}s" if a.task == "ss" else "")
    )
    (ROOT / "results").mkdir(exist_ok=True)
    events = run(
        src,
        dec,
        hop_s=hop,
        duration_s=a.duration,
        on_event=show,
        log_path=ROOT / "results" / f"realtime_{a.user}_{a.task}.jsonl",
        clock=getattr(src, "clock", __import__("time").time),
        sleep=getattr(src, "sleep", __import__("time").sleep),
    )
    print(f"{len(events)} decisions.")


if __name__ == "__main__":
    main()
