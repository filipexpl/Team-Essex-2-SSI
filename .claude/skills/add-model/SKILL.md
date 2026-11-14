---
name: add-model
description: Add a new decoding model (MI or silent speech) to the registry with tests and an eval. Use when asked to try a new method (Riemann, EEGNet, filter-bank CSP, CNN on EMG, etc.).
---

# add-model

1. Create `src/ssmi/models/<name>.py` exposing a factory `<name>(fs: float) -> estimator`
   with sklearn API (`fit(X, y)`, `predict(X)`, ideally `predict_proba`). Input X is
   `(trials, channels, samples)` at `fs` — never assume fs.
2. Heavy deps (torch, pyriemann, braindecode) → import inside the factory, add to the
   matching optional extra in `pyproject.toml`. Tests must skip cleanly if missing
   (`pytest.importorskip`).
3. Register in `src/ssmi/models/__init__.py::REGISTRY`.
4. The parametrized test in `tests/test_pipeline.py` picks it up automatically; make it pass.
5. Run the `run-eval` skill on synthetic + the real dataset; compare against baseline.
6. Add 1–3 lines to `docs/decisions.md`: method, why, result vs baseline.
7. Deep models: must still run 1 epoch on CPU in < 60 s on synthetic data (CI budget).
   Full training goes to `.github/workflows/train-gpu.yml` or Colab/Kaggle.
