"""Dataset loaders -> Recording. Heavy deps (mne) are imported lazily."""

from pathlib import Path

import numpy as np

from ssmi import MI_CLASSES
from ssmi.acquisition import Recording
from ssmi.acquisition import synthetic as syn

DATA = Path(__file__).resolve().parents[3] / "data" / "raw"

# 8 channels closest to the BrainAccess MINI cap layout we plan to use (see docs/hardware.md)
MI_8CH = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4"]


def load(task: str, name: str, **kw) -> Recording:
    if name == "synthetic":
        return syn.make_mi(**kw) if task == "mi" else syn.make_ss(**kw)
    if task == "mi" and name == "physionet":
        return load_physionet_mi(**kw)
    if task == "ss" and name == "gaddy":
        return load_gaddy(**kw)
    if name == "own":
        return load_own(task=task, **kw)
    raise KeyError(f"No loader for task={task} dataset={name}")


def load_physionet_mi(
    subjects=range(1, 11), channels=MI_8CH, tmin=0.5, tmax=3.5, imagery=True, resample_to=250.0
) -> Recording:
    """EEGMMIDB: runs 4/8/12 = imagined L/R fist, 6/10/14 = imagined both fists / both feet.
    (executed: 3/7/11 and 5/9/13). T1/T2 meaning depends on run type!
    """
    import mne
    from mne.datasets import eegbci

    lr_runs, bf_runs = ([4, 8, 12], [6, 10, 14]) if imagery else ([3, 7, 11], [5, 9, 13])
    X, y, g = [], [], []
    for s in subjects:
        for runs, mapping in ((lr_runs, {"T1": 0, "T2": 1}), (bf_runs, {"T1": 2, "T2": 3})):
            files = eegbci.load_data(s, runs, path=str(DATA / "physionet"), update_path=False)
            raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in files])
            eegbci.standardize(raw)
            raw.pick(channels).resample(resample_to)
            ev, ev_id = mne.events_from_annotations(
                raw, event_id={k: v + 1 for k, v in mapping.items()}
            )
            ep = mne.Epochs(raw, ev, ev_id, tmin, tmax, baseline=None, preload=True, verbose=False)
            X.append(ep.get_data())
            y.append(ep.events[:, 2] - 1)
            g.append(np.full(len(ep), s))
    return Recording(
        np.concatenate(X),
        np.concatenate(y),
        resample_to,
        list(channels),
        MI_CLASSES,
        np.concatenate(g),
        meta={"source": "physionet_eegmmidb"},
    )


def load_gaddy(root=None, **kw) -> Recording:
    """Gaddy & Klein 2020 (Zenodo 4064409): {i}_emg.npy (T, 8) @ 1000 Hz, sentence-level.
    Not word-segmented -> used for self-supervised pretraining / sequence models, not the
    word-classification eval. See docs/datasets.md.
    """
    raise NotImplementedError("Gaddy loader: implement in sequence-model track (see docs).")


def load_own(task="ss", root=None, **kw) -> Recording:
    """Own recordings (scripts/record_session.py): npz with X, y, fs, ch_names, classes, groups."""
    root = Path(root or Path(__file__).resolve().parents[3] / "data" / "processed" / f"own_{task}")
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz in {root}")
    parts = [np.load(f, allow_pickle=True) for f in files]
    return Recording(
        np.concatenate([p["X"] for p in parts]),
        np.concatenate([p["y"] for p in parts]),
        float(parts[0]["fs"]),
        list(parts[0]["ch_names"]),
        list(parts[0]["classes"]),
        np.concatenate([p["groups"] for p in parts]),
        meta={"source": "own"},
    )
