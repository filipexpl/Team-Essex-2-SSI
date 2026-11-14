# Datasets

## Motor imagery (EEG)
| Dataset | Classes | Ch / fs | Subjects | Use | Access |
|---|---|---|---|---|---|
| **PhysioNet EEGMMIDB** (Schalk 2009) | L fist, R fist, **both fists, both feet**, rest — exactly our 4 classes (imagined + executed) | 64 @ 160 Hz | 109 | **primary pretraining + cross-subject eval** (pick 8 motor ch) | open, `mne.datasets.eegbci`, `physionet download eegmmidb`, S3 `s3://physionet-open/eegmmidb/1.0.0/` |
| BCI Competition IV 2a (BNCI2014_001) | L, R, feet, tongue | 22 @ 250 Hz | 9 | secondary benchmark (same fs as our device) | open, MOABB |
| OpenBMI Lee2019 (MOABB `Lee2019_MI`) | L, R | 62 @ 1000 Hz | 54 | 2-class fallback pretraining | open, MOABB |

EEGMMIDB caveat: T1/T2 mean L/R fist in runs 3,4,7,8,11,12 but both-fists/both-feet in runs
5,6,9,10,13,14 — mapping is implemented in `datasets.load_physionet_mi`.

## Silent speech (facial sEMG)
| Dataset | Content | Ch / fs | Use | Access |
|---|---|---|---|---|
| **Gaddy & Klein 2020** "Digital Voicing of Silent Speech" | ~20 h, 1 speaker, silent + vocalized, sentences (EN) | **8 @ 1000 Hz** (same channel count as ours) | self-supervised pretraining of EMG encoder; resample → device fs | open, Zenodo 4064409 (3.9 GB), code github.com/dgaddy/silent_speech |
| Gowda et al. 2024/25 (Neural Speech Interface) | 16 subjects, NATO alphabet + 67-word vocab | 22 @ 5000 Hz | multi-subject pretraining, classification setup closest to ours | open (see arXiv 2411.02591 / 2502.05762) |
| EMG-UKA Trial Corpus (Wand 2014) | ~2 h, 4 speakers | 6 @ 600 Hz | extra benchmark | on request (CSL Bremen) |
| Silent-Speech-recognition-dataset (IITK) | 22-word vocab | 8 @ 250 Hz, OpenBCI Cyton | **closest to our hardware (8ch @250 Hz)** | github.com/DarshanJain295/Silent-Speech-recognition-dataset |
| **Our own recordings** | 6–10 PL words | EXG8 | **final eval + demo** | `data/processed/own_ss` |

Reality check: no public dataset has Polish words with our electrode layout. Public data serves
for pretraining and pipeline validation; the demo model is trained/calibrated on our own sessions.
