"""Calibrate the decoder for a NEW person (~5-8 minutes).

  # dry run without hardware
  python scripts/calibrate.py --task ss --user P07 --device synthetic --trials 10
  # real device (BrainAccess Board streaming to LSL)
  python scripts/calibrate.py --task ss --user P07 --device lsl --vocab core6 --trials 15
  # re-calibrate from an already recorded session
  python scripts/calibrate.py --task mi --user P07 --from-npz data/processed/own_mi/P07_s1.npz

Steps: rest baseline -> K cued trials per class (random order) -> save session npz ->
fit Euclidean alignment + model (pooled with other people's data if available) -> save profile.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssmi.acquisition import Recording  # noqa: E402
from ssmi.acquisition.sources import open_source  # noqa: E402
from ssmi.calibration import calibrate_user  # noqa: E402
from ssmi.datasets import load_own  # noqa: E402

DEFAULT_MODEL = {"mi": "csp_lda", "ss": "td_features_lda"}
PROTOCOL_VERSION = "0.2"


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def classes_for(task: str, vocab: str | None, cfg: dict) -> list[str]:
    if task == "ss":
        return list(cfg["silent_speech"][vocab or "core6"])
    return list(cfg["motor_imagery"][vocab or "four_class"])


def record(task, src, classes, trials, proto, cues, realtime, seed=0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.repeat(np.arange(len(classes)), trials))
    print(f"\n[1/3] Spoczynek {proto['rest_baseline']} s — rozluźnij twarz/ręce, patrz w punkt.")
    rest = src.read(proto["rest_baseline"])
    print(f"[2/3] {len(order)} prób ({trials} na klasę). Ctrl+C przerywa.\n")
    X, y = [], []
    for i, c in enumerate(order):
        name = classes[c]
        print(
            f"  ({i + 1:>3}/{len(order)})  {cues.get(name, name.upper()):<50}", end="", flush=True
        )
        if hasattr(src, "mark"):
            src.mark(f"cue:{name}")
        if realtime:
            time.sleep(proto["prepare"])
        print(" ● TERAZ", flush=True)
        x = src.read(proto["window"], cue=name)
        if task == "mi":  # keep the stable part of imagery
            a, b = int(proto["epoch_start"] * src.fs), int(proto["epoch_end"] * src.fs)
            x = x[:, a:b]
        X.append(x)
        y.append(c)
        if realtime:
            time.sleep(proto["rest"] * rng.uniform(0.8, 1.2))
    return np.asarray(X), np.asarray(y), rest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mi", "ss"], required=True)
    ap.add_argument("--user", required=True, help="anonymous participant id, e.g. P07")
    ap.add_argument("--session", default=None, help="session id (default: timestamp)")
    ap.add_argument("--device", choices=["synthetic", "board", "lsl"], default="synthetic")
    ap.add_argument("--board-device", default=None, help="Board device id (substring) if several")
    ap.add_argument("--lag", type=float, default=0.25, help="Board database lag in seconds")
    ap.add_argument("--study", default="generic", help="replicated protocol, see protocol doc")
    ap.add_argument("--record-dir", default=None, help="also start a Board recording (.db) here")
    ap.add_argument("--vocab", default=None, help="key from configs/vocab.yaml")
    ap.add_argument("--trials", type=int, default=15, help="trials per class")
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", choices=["pooled", "subject"], default="pooled")
    ap.add_argument("--from-npz", default=None)
    ap.add_argument("--realtime", action="store_true", help="real timing for synthetic device")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load((ROOT / "configs" / "vocab.yaml").read_text())
    proto = cfg["protocol"]["motor_imagery" if a.task == "mi" else "silent_speech"]
    cues = cfg["motor_imagery"]["cues"] if a.task == "mi" else {}
    session = a.session or time.strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "data" / "processed" / f"own_{a.task}"
    model = a.model or DEFAULT_MODEL[a.task]

    if a.from_npz:
        d = np.load(a.from_npz, allow_pickle=True)
        calib = Recording(d["X"], d["y"], float(d["fs"]), list(d["ch_names"]), list(d["classes"]))
    else:
        classes = classes_for(a.task, a.vocab, cfg)
        realtime = a.device in ("lsl", "board") or a.realtime
        src = open_source(
            a.device, a.task, classes, realtime=realtime, board_device=a.board_device, lag_s=a.lag
        )
        print(f"      device={a.device} fs={src.fs} Hz channels={list(src.ch_names)}")
        board_rec = a.record_dir and getattr(src, "control", None)
        if board_rec:
            print(
                "      Board recording:",
                src.control.start_recording(
                    a.record_dir, a.user, session, 1, f"{a.task}-{a.study}"
                ),
            )
        try:
            X, y, rest = record(a.task, src, classes, a.trials, proto, cues, realtime)
        finally:
            if board_rec:
                print("      Board recording:", src.control.stop_recording())
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{a.user}_{session}.npz"
        np.savez(
            path,
            X=X,
            y=y,
            fs=src.fs,
            ch_names=np.array(src.ch_names),
            classes=np.array(classes),
            groups=np.zeros(len(y), int),
            rest=rest,
            participant=a.user,
            session=session,
            device=a.device,
            vocab=a.vocab or ("core6" if a.task == "ss" else "four_class"),
            study=a.study,
            protocol_version=PROTOCOL_VERSION,
            git_commit=git_commit(),
            timing=json.dumps(proto),
            markers=np.array(getattr(src, "marker_log", []), dtype=object),
        )
        print(f"      zapisano sesję → {path.relative_to(ROOT)}")
        calib = Recording(X, y, src.fs, list(src.ch_names), classes)

    # other people's sessions (same task, same channels, same classes) as source data
    source = None
    if a.mode == "pooled":
        try:
            allrec = load_own(task=a.task, exclude_participant=a.user)
            if allrec.ch_names == calib.ch_names and allrec.class_names == calib.class_names:
                source = allrec
        except FileNotFoundError:
            pass
    print(
        f"[3/3] Kalibracja: model={model}, tryb={'pooled' if source else 'subject'}, "
        f"dane innych osób: {0 if source is None else len(source.y)} prób"
    )
    prof = calibrate_user(
        a.user, a.task, calib, model, source, mode="pooled" if source else "subject"
    )
    p = prof.save()
    if prof.calib_cv_accuracy is not None:
        chance = 1 / len(calib.class_names)
        print(
            f"      trafność CV na kalibracji: {prof.calib_cv_accuracy:.2f} "
            f"(szansa {chance:.2f}; w obrębie sesji => optymistycznie)"
        )
    print(
        f"      profil → {p.relative_to(ROOT)}\n      dalej: python scripts/live.py "
        f"--task {a.task} --user {a.user} --device {a.device}"
    )


if __name__ == "__main__":
    main()
