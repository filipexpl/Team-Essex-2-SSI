"""How many calibration trials does a NEW person need?

For each held-out group (subject or session) of a dataset:
  - calibration set = first K trials per class of that group (K in --ks),
  - test set        = all remaining trials of that group,
  - source          = all other groups.
Compares strategies:
  source_only_raw   : model from other people, no alignment         (0 personal trials)
  source_only_ea    : same + Euclidean alignment on the K trials      (labels unused)
  subject_ea        : trained only on K personal trials (+ EA)
  pooled_ea         : other people + K personal trials, all aligned

python -m ssmi.evals.calibration_curve --task ss --dataset synthetic --model td_features_lda
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from ssmi.acquisition import Recording
from ssmi.calibration import fit_calibrated
from ssmi.datasets import load

ROOT = Path(__file__).resolve().parents[3]
STRATEGIES = {
    "source_only_raw": dict(mode="source_only", align=False),
    "source_only_ea": dict(mode="source_only", align=True),
    "subject_ea": dict(mode="subject", align=True),
    "pooled_ea": dict(mode="pooled", align=True),
}


def _sub(rec: Recording, mask) -> Recording:
    return Recording(
        rec.X[mask], rec.y[mask], rec.fs, rec.ch_names, rec.class_names, rec.groups[mask], rec.meta
    )


def split_calib(rec: Recording, k: int, seed: int):
    rng = np.random.default_rng(seed)
    calib = np.zeros(len(rec.y), bool)
    for c in np.unique(rec.y):
        idx = np.flatnonzero(rec.y == c)
        calib[rng.permutation(idx)[:k]] = True
    return calib


def run(task, dataset, model, ks=(2, 5, 10, 20), seed=0):
    rec = load(task, dataset)
    groups = np.unique(rec.groups)
    if len(groups) < 2:
        raise SystemExit("Need >= 2 subjects/sessions for a calibration curve.")
    res = {s: {k: [] for k in ks} for s in STRATEGIES}
    for g in groups:
        tgt, src = _sub(rec, rec.groups == g), _sub(rec, rec.groups != g)
        for k in ks:
            if np.bincount(tgt.y).min() <= k:
                continue
            cm = split_calib(tgt, k, seed)
            calib, test = _sub(tgt, cm), _sub(tgt, ~cm)
            for name, kw in STRATEGIES.items():
                m, al = fit_calibrated(calib, model, src, **kw)
                Xt = al.transform(test.X) if al is not None else test.X
                res[name][k].append(balanced_accuracy_score(test.y, m.predict(Xt)))
    summary = {
        s: {int(k): (float(np.mean(v)) if v else None) for k, v in d.items()}
        for s, d in res.items()
    }
    return {
        "task": task,
        "dataset": dataset,
        "model": model,
        "ks": list(ks),
        "chance": 1 / len(rec.class_names),
        "n_groups": int(len(groups)),
        "balanced_accuracy": summary,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mi", "ss"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ks", default="2,5,10,20")
    a = ap.parse_args(argv)
    out = run(a.task, a.dataset, a.model, tuple(int(k) for k in a.ks.split(",")))
    (ROOT / "results").mkdir(exist_ok=True)
    path = ROOT / "results" / f"calib_{a.task}_{a.dataset}_{a.model}.json"
    path.write_text(json.dumps(out, indent=2))
    ks = out["ks"]
    print(
        f"calibration curve {a.task}/{a.dataset}/{a.model} (chance {out['chance']:.2f}, "
        f"{out['n_groups']} groups) — balanced accuracy vs trials/class"
    )
    print(f"{'strategy':<18}" + "".join(f"{'K=' + str(k):>8}" for k in ks))
    for s, d in out["balanced_accuracy"].items():
        cells = "".join(f"{d[k]:8.2f}" if d[k] is not None else f"{'-':>8}" for k in ks)
        print(f"{s:<18}{cells}")


if __name__ == "__main__":
    main()
