# Decision log (append-only, newest on top)

- 2026-11-14 — Hardware supports 250 and 500 Hz. Silent speech is recorded at **500 Hz**
  (Nyquist 250 Hz, most of the sEMG band); 250 Hz versions are made offline for IITK comparisons.
  Gaddy notebook simulates both 500 and 250 Hz.

- 2026-11-14 — IITK loader (words + words-from-sentences) and `evals.transfer`. TD-features+LDA:
  isolated words 0.71 (LOBO), but pre-onset rest alone 0.24 (chance 0.045) => per-file confound;
  words -> sentences 0.05 (chance). Keep randomized word order in our protocol; judge SS models on
  transfer, not on IITK isolated words.

- 2026-11-14 — Silent speech is **English only** (team decision). Vocabularies switched to EN:
  `core6` (yes/no/up/down/left/right), `control10`, `iitk22` (= IITK public set, same 8 ch @ 250 Hz),
  `nato26` (= Gowda et al.), `care12`, `digits10`. MI unchanged.

- 2026-11-14 — Calibration for new users. Euclidean Alignment (He & Wu 2020) per person/session +
  pooled training (others' aligned data + K personal trials, weighted x3). Scripts
  `calibrate.py`/`live.py` (synthetic + LSL), `configs/vocab.yaml` (vocabularies), eval
  `calibration_curve`. On `synthetic_shift` (5 simulated people, spatial mixing): MI 4-class
  K=10 pooled_ea 0.53 vs 0.30 without alignment; SS 6 words source_only_ea 0.93 vs 0.86 raw.
  Synthetic only — real numbers after first recordings.

- 2026-11-14 — Scaffold. Baselines: MI = 8–30 Hz + OvR-CSP + shrinkage LDA; SS = notch 50 Hz +
  20–(Nyquist) Hz bandpass + time-domain features on 5 sub-windows + LDA. Eval = leave-one-group-out
  (subject/session), balanced accuracy, kappa, ITR, latency. Synthetic smoke: MI 0.74, SS 1.00.
