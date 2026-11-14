# Decision log (append-only, newest on top)

- 2026-11-14 — Calibration for new users. Euclidean Alignment (He & Wu 2020) per person/session +
  pooled training (others' aligned data + K personal trials, weighted x3). Scripts
  `calibrate.py`/`live.py` (synthetic + LSL), `configs/vocab.yaml` (PL vocabularies), eval
  `calibration_curve`. On `synthetic_shift` (5 simulated people, spatial mixing): MI 4-class
  K=10 pooled_ea 0.53 vs 0.30 without alignment; SS 6 words source_only_ea 0.93 vs 0.86 raw.
  Synthetic only — real numbers after first recordings.

- 2026-11-14 — Scaffold. Baselines: MI = 8–30 Hz + OvR-CSP + shrinkage LDA; SS = notch 50 Hz +
  20–(Nyquist) Hz bandpass + time-domain features on 5 sub-windows + LDA. Eval = leave-one-group-out
  (subject/session), balanced accuracy, kappa, ITR, latency. Synthetic smoke: MI 0.74, SS 1.00.
