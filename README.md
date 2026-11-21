# Team-Essex-2-SSI

**Silent Speech + Motor Imagery Decoder**

Hackathon project: decoding **silently spoken words** from facial sEMG (8 electrodes, EXG8) and
**motor intentions** (left hand / right hand / both hands / feet) from EEG (BrainAccess MINI).

```bash
make setup        # install
make test         # unit tests (no hardware needed)
make eval-smoke   # end-to-end eval on synthetic data
make eval TASK=mi DATASET=physionet MODEL=csp_lda   # needs `pip install -e ".[eeg]"` + internet
```

- `docs/quickstart.md` — **running on the device (macOS / Windows)**: install, signal check, calibration, real time
- `CLAUDE.md` — rules for Claude (also read it, humans)
- `docs/hardware.md` — devices, montage, electrode placement, open questions
- `docs/datasets.md` — public datasets and how we use them
- `docs/decisions.md` — experiment log
- `evals/thresholds.yaml` — quality gates (human-only edits)

## Working with Claude
- Locally: `claude` in the repo root (reads `CLAUDE.md`, `.claude/skills`, `.claude/agents`).
- On GitHub: open an issue and write `@claude ...` — the Action opens a PR; CI must be green.
- Skills: `run-eval`, `add-model`, `signal-qc`, `record-session`. Agent: `eval-reviewer`.
