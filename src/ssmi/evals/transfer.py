"""Train on one dataset, test on another (same classes / channels / fs).

  python -m ssmi.evals.transfer --task ss --train iitk --test iitk_sent --model td_features_lda
  python -m ssmi.evals.transfer --task ss --train iitk --test iitk_sent --align

Typical uses: isolated words -> words in sentences (IITK), team recordings -> new person.
Classes absent from the test set are still trained on; balanced accuracy is over present ones.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from ssmi.calibration.alignment import EuclideanAligner
from ssmi.datasets import load
from ssmi.models import get_model

ROOT = Path(__file__).resolve().parents[3]


def run(task, train, test, model, align=False):
    a, b = load(task, train), load(task, test)
    if (a.fs, list(a.ch_names), list(a.class_names)) != (
        b.fs,
        list(b.ch_names),
        list(b.class_names),
    ):
        raise SystemExit("train/test differ in fs, channels or classes")
    Xa, Xb = a.X, b.X
    if align:  # each dataset whitened by its own covariance (no test labels used)
        Xa, Xb = EuclideanAligner().fit_transform(Xa), EuclideanAligner().fit_transform(Xb)
    m = get_model(model, a.fs).fit(Xa, a.y)
    p = m.predict(Xb)
    present = np.unique(b.y)
    return {
        "task": task,
        "train": train,
        "test": test,
        "model": model,
        "align": align,
        "n_train": int(len(a.y)),
        "n_test": int(len(b.y)),
        "balanced_accuracy": float(balanced_accuracy_score(b.y, p)),
        "chance": 1 / len(present),
        "classes_in_test": [a.class_names[i] for i in present],
        "per_class_recall": {a.class_names[c]: float((p[b.y == c] == c).mean()) for c in present},
        "confusion": confusion_matrix(b.y, p, labels=present).tolist(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mi", "ss"], required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--model", default="td_features_lda")
    ap.add_argument("--align", action="store_true")
    a = ap.parse_args(argv)
    out = run(a.task, a.train, a.test, a.model, a.align)
    (ROOT / "results").mkdir(exist_ok=True)
    tag = "_ea" if a.align else ""
    path = ROOT / "results" / f"transfer_{a.task}_{a.train}_to_{a.test}_{a.model}{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(
        f"[{a.train} -> {a.test}, {a.model}{', EA' if a.align else ''}] "
        f"bal_acc={out['balanced_accuracy']:.3f} (chance {out['chance']:.3f}, "
        f"{out['n_test']} test trials, {len(out['classes_in_test'])} classes) -> {path.name}"
    )


if __name__ == "__main__":
    main()
