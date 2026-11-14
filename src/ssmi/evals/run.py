"""Eval harness.

python -m ssmi.evals.run --task mi --dataset synthetic --model csp_lda --check-thresholds
Writes results/<task>_<dataset>_<model>.json; exits 1 if thresholds are not met.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from ssmi.datasets import load
from ssmi.evals.metrics import latency_ms, summarize
from ssmi.evals.splits import get_splits
from ssmi.models import get_model

ROOT = Path(__file__).resolve().parents[3]


def run(task, dataset, model_name, scheme="leave_one_group_out", seed=0):
    rec = load(task, dataset)
    splits = get_splits(rec, scheme, seed)
    y_true, y_pred, folds = [], [], []
    for tr, te in splits:
        m = get_model(model_name, rec.fs).fit(rec.X[tr], rec.y[tr])
        p = m.predict(rec.X[te])
        y_true.append(rec.y[te])
        y_pred.append(p)
        folds.append(float((p == rec.y[te]).mean()))
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    trial_sec = rec.X.shape[-1] / rec.fs
    out = summarize(y_true, y_pred, len(rec.class_names), trial_sec)
    out.update(
        task=task,
        dataset=dataset,
        model=model_name,
        scheme=scheme,
        groups_independent=len(set(rec.groups)) > 1,
        fold_acc=folds,
        fs=rec.fs,
        n_trials=len(rec.y),
        classes=rec.class_names,
        latency_ms=latency_ms(m, rec.X[:1]),
    )
    return out


def check(out) -> list[str]:
    th = yaml.safe_load((ROOT / "evals" / "thresholds.yaml").read_text())
    key = f"{out['task']}/{out['dataset']}"
    rules = th.get(key, {})
    fails = []
    if (
        "min_balanced_accuracy" in rules
        and out["balanced_accuracy"] < rules["min_balanced_accuracy"]
    ):
        fails.append(
            f"balanced_accuracy {out['balanced_accuracy']:.3f} < {rules['min_balanced_accuracy']}"
        )
    if "max_latency_ms" in rules and out["latency_ms"] > rules["max_latency_ms"]:
        fails.append(f"latency {out['latency_ms']:.1f}ms > {rules['max_latency_ms']}ms")
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mi", "ss"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scheme", default="leave_one_group_out")
    ap.add_argument("--check-thresholds", action="store_true")
    a = ap.parse_args(argv)
    out = run(a.task, a.dataset, a.model, a.scheme)
    (ROOT / "results").mkdir(exist_ok=True)
    path = ROOT / "results" / f"{a.task}_{a.dataset}_{a.model}.json"
    path.write_text(json.dumps(out, indent=2))
    print(
        f"[{a.task}/{a.dataset}/{a.model}] bal_acc={out['balanced_accuracy']:.3f} "
        f"(chance {out['chance']:.2f}) kappa={out['kappa']:.2f} "
        f"ITR={out['itr_bits_min']:.1f} bit/min latency={out['latency_ms']:.1f}ms -> {path.name}"
    )
    if a.check_thresholds and (fails := check(out)):
        print("THRESHOLD FAIL:", *fails, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
