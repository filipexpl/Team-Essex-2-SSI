# Decision log (append-only, newest on top)

- 2026-11-14 — Scaffold. Baselines: MI = 8–30 Hz + OvR-CSP + shrinkage LDA; SS = notch 50 Hz +
  20–(Nyquist) Hz bandpass + time-domain features on 5 sub-windows + LDA. Eval = leave-one-group-out
  (subject/session), balanced accuracy, kappa, ITR, latency. Synthetic smoke: MI 0.74, SS 1.00.
