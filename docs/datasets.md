# Datasets

What exists, what vocabulary each covers, and what we can realistically use it for.
Silent speech is **English only**. All English sEMG sets below share at least part of their
vocabulary with ours (`configs/vocab.yaml`), but only IITK matches our hardware (8 ch @ 250 Hz).
Public data is for pretraining, method validation and benchmarking; the demo model is calibrated
per person on our own recordings (`docs/calibration.md`).

## Silent speech (facial sEMG)

| Dataset | Vocabulary | Unit | Subjects | Ch / fs | Access | Use for us |
|---|---|---|---|---|---|---|
| **Gaddy & Klein 2020** — Digital Voicing of Silent Speech | (a) closed: **67 words**, date/time expressions ("<weekday> <month> <year>"), 500 utterances, 1 session; (b) open: book sentences (Project Gutenberg), ~18.6 h, many sessions | sentences, silent + vocalized + audio | 1 | **8 @ 1000 Hz** | open, Zenodo 4064409 (3.9 GB); code github.com/dgaddy/silent_speech | self-supervised EMG encoder pretraining (resample to device fs); cross-session drift studies. Not word-segmented → no direct word classifier |
| **Gowda et al. 2024** — geometry of orofacial sEMG | **26 NATO alphabet** words (alfa…zulu) spelled-out sentences + orofacial gestures/phonemes | words / sentences | 16 | 22 @ 5000 Hz | open, OSF DOI 10.17605/OSF.IO/YM5JD; code github.com/HarshavardhanaTG/geometryOfOrofacialNeuromuscularSystem | **closest task design to ours** (multi-subject word classification, SPD/Riemann methods = our alignment approach). Need to pick ~8 of 22 channels |
| Gowda et al. 2025 — emg2speech | ~6 800 words (9 h, healthy) + ~300 words (1 h, ALS) + orofacial gestures from 12 people | sentences | 1 + 1 + 12 | 31 @ 5000 Hz | open, OSF 65vbx; github.com/MyoVerse/emg2speech | large-vocab pretraining; gesture subset for cross-subject |
| Gowda et al. 2025 (ICML) | **67-word** limited English vocab | sentences | — | 22 @ 5000 Hz | open (same group) | benchmark for small vocab |
| **IITK silent speech** (Jain et al.) | **22 words**: all, anyone, anything, are, back, can, come, could, did, doing, find, help, here, in, now, out, please, right, stop, that, tomorrow, you | 4–5-word sentences | 1 | **8 @ 250 Hz, OpenBCI Cyton** | github.com/DarshanJain295/Silent-Speech-recognition-dataset (no license stated) | **same channel count and fs as our likely setup** → best sanity check for the pipeline at 250 Hz |
| NATO-alphabet KDE-SSI (arXiv 2308.06533) | 26 NATO words, 3 900 samples | isolated words | 5 | commercial device | check paper | isolated-word classification benchmark |
| AVE Speech (Zhou et al. 2025) | **100 Mandarin daily-need sentences** (3–5 words, e.g. "I'm hungry", "turn right") + no-command class | sentences, ×10 reps | 100 | 6 EMG | Hugging Face MML-Group/AVE-Speech (gated, accept terms) | largest cross-subject EMG set → test of cross-person transfer; inspiration for `care12` vocab |
| Mandarin overt/silent/imagined (Sci Data 2025) | Mandarin words/sentences | — | 30 | EEG + sEMG | Scientific Data (s41597-025-06213-z) | hybrid EEG+EMG ideas |
| EMG-UKA Trial Corpus (Wand 2014) | English read sentences (~108-word test vocab in session-independent work) | sentences | 4 (trial subset of 8) | 6 @ 600 Hz | trial subset free (1:52 h), CSL Bremen | session-independent baseline |
| CSL-EMG_Array (Diener 2020) | sentences | sentences | 8, 12 sessions | electrode array | open access | — |

## Motor imagery (EEG)

| Dataset | "Vocabulary" (classes) | Subjects | Ch / fs | Access | Use |
|---|---|---|---|---|---|
| **PhysioNet EEGMMIDB** | left fist, right fist, **both fists, both feet**, rest — imagined and executed (exactly our 4 classes) | 109 | 64 @ 160 Hz | open: `mne.datasets.eegbci`, `physionet download eegmmidb` | **source data for pooled calibration** (8 motor channels, resampled to 250 Hz) |
| BCI IV 2a (BNCI2014_001) | left, right, feet, tongue | 9 | 22 @ 250 Hz | MOABB | 3 of 4 classes overlap; benchmark |
| Lee2019 / OpenBMI | left, right | 54 | 62 @ 1000 Hz | MOABB | 2-class fallback |

EEGMMIDB caveat: T1/T2 = left/right fist in runs 3,4,7,8,11,12 but both-fists/both-feet in runs
5,6,9,10,13,14 — handled in `datasets.load_physionet_mi`.

## Our vocabularies (`configs/vocab.yaml`)
English only. Start with `core6` (yes, no, up, down, left, right); grow only when cross-session
accuracy on our own data holds (see `docs/calibration.md`).

| vocab | words | overlap with public data |
|---|---|---|
| `core6` | yes, no, up, down, left, right | "right" ∈ IITK-22 |
| `control10` | core6 + stop, go, help, back | right, stop, help, back ∈ IITK-22 |
| `iitk22` | exactly IITK-22 | **100 %** IITK (same 8 ch @ 250 Hz) → benchmark / pretraining |
| `nato26` | NATO alphabet | **100 %** Gowda et al. (16 subjects) → spelling any word |
| `care12` | water, food, pain, cold, hot, toilet, help, yes, no, doctor, sleep, thanks | help ∈ IITK-22 |
| `digits10` | zero … nine | number words in Gaddy closed vocab (dates/times) |
