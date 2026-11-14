---
name: run-eval
description: Run the eval harness for a model/dataset, compare against baseline and report. Use after any change to preprocessing or models, or when asked "how good is X".
---

# run-eval

1. `make eval-smoke` — must pass first (pipeline sanity).
2. `make eval TASK=<mi|ss> DATASET=<synthetic|physionet|own> MODEL=<name>`.
3. Run the baseline on the same dataset (`csp_lda` for mi, `td_features_lda` for ss) unless
   `results/<task>_<dataset>_<baseline>.json` already exists from the current commit.
4. Report as a table: balanced_accuracy (vs chance), kappa, ITR bit/min, latency_ms,
   per-fold accuracy spread (min–max), and the confusion matrix with class names.
5. If `groups_independent` is false, say clearly that the result is optimistic (no
   cross-session split).
6. Never edit `evals/thresholds.yaml`. If a gate fails, say so and propose next experiments.
