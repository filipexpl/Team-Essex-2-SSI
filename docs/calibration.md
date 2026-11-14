# Calibration: a new person using the device

Signals differ a lot between people and between sessions (electrode position, skin, anatomy,
mu-rhythm frequency). We handle it in two layers:

1. **Euclidean Alignment (EA)** — whitens each person/session by its mean spatial covariance.
   Needs **no labels**, only that person's trials. Implemented in `ssmi.calibration.alignment`.
2. **Short calibration block** — K cued trials per class; the model is trained on
   *other people's aligned data + this person's aligned trials* (`mode=pooled`), or on this
   person only (`mode=subject`) when no other data exist yet.

## Procedure (≈ 5–8 min)

```bash
# 0. dry run without hardware
make calibrate TASK=ss PID=P07 DEVICE=synthetic
make live TASK=ss PID=P07 DEVICE=synthetic

# 1. real device: start BrainAccess Board, connect the amplifier, then
python scripts/board_check.py
python scripts/calibrate.py --task ss --user P07 --device board --vocab core6 --trials 15 \
       --study generic --record-dir recordings/
python scripts/live.py      --task ss --user P07 --device board
```

| Step | Silent speech (EXG8) | Motor imagery (MINI) |
|---|---|---|
| Electrodes | gel Ag/AgCl, layout from `docs/hardware.md`, photo of placement | cap on motor strip, check contact in BrainAccess Board |
| Rest baseline | 20 s, relaxed face | 30 s, eyes open, relaxed |
| Trials | 6 words × 15 = 90 trials (~5 min) | 4 classes × 15 = 60 trials (~8 min) |
| Window | 1.5 s after "TERAZ" | 4 s imagery, epoch 0.5–3.5 s |
| Output | `data/processed/own_ss/P07_<session>.npz`, `profiles/P07_ss.pkl` | same with `mi` |

The script prints a cross-validated accuracy on the calibration block. It is **within-session,
so optimistic**; the honest number comes from the calibration curve below.

Recalibrate when electrodes are re-attached (EA alone can be refreshed from the rest baseline +
a few trials; the model only if accuracy drops).

## How many trials does a new person need?
```bash
make calib-curve TASK=ss DATASET=own MODEL=td_features_lda   # needs >= 2 recorded sessions
```
Leave-one-person/session-out; for K = 2, 5, 10, 20 trials per class compares:
`source_only_raw` (no calibration), `source_only_ea` (alignment only, zero labels),
`subject_ea` (only the K personal trials), `pooled_ea` (others + K personal trials).

Synthetic results with strong between-person differences (`synthetic_shift`, 5 people) — these
only show the machinery works, they say nothing about real data:

| strategy | SS K=2 | SS K=5 | SS K=10 | MI K=2 | MI K=5 | MI K=10 | MI K=20 |
|---|---|---|---|---|---|---|---|
| source_only_raw | 0.86 | 0.86 | 0.86 | 0.30 | 0.30 | 0.30 | 0.31 |
| source_only_ea  | 0.93 | 0.93 | 0.93 | 0.46 | 0.46 | 0.49 | 0.47 |
| subject_ea      | 0.11 | 1.00 | 1.00 | 0.30 | 0.37 | 0.41 | 0.53 |
| pooled_ea       | 1.00 | 1.00 | 1.00 | 0.47 | 0.49 | 0.53 | 0.53 |

Plan for real data: as soon as 2–3 team members have recorded, run the curve and pick K for the
demo (smallest K where `pooled_ea` is within ~5 pp of K=20).

## Protocol rule learned from IITK
Always interleave all words in random order within one session (calibrate.py does this). Recording
each word in its own block lets the model learn recording drift instead of articulation
(see `docs/datasets.md`, IITK section).

## Limits
- Pooled mode requires identical channel names, class lists and fs; otherwise it falls back to
  subject-only (printed by the script).
- Silent speech: public English data uses other electrode layouts (except IITK: 8 ch @ 250 Hz,
  `iitk22` vocab), so the main "source" is our own team's recordings — the more people record,
  the shorter calibration gets.
- Live decoding is push-to-talk (Enter → 1.5 s window). Continuous detection of word onsets is a
  next step (energy-based VAD on EMG envelope).
