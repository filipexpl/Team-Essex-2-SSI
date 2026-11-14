import numpy as np

from ssmi.acquisition import synthetic
from ssmi.calibration import UserProfile, calibrate_user, fit_calibrated
from ssmi.calibration.alignment import EuclideanAligner, align_by_group, mean_cov
from ssmi.datasets import load, load_own
from ssmi.evals.calibration_curve import run as calib_curve


def test_alignment_whitens_each_group():
    r = load("ss", "synthetic_shift", n_per_class=5)
    Xa = align_by_group(r.X, r.groups)
    for g in np.unique(r.groups):
        C = mean_cov(Xa[r.groups == g], reg=0)
        assert np.allclose(C, np.eye(8), atol=1e-3)  # small reg in fit


def test_aligner_needs_no_labels():
    r = synthetic.make_mi(n_per_class=5, n_subjects=1, mix=0.8)
    al = EuclideanAligner().fit(r.X)  # no y
    assert al.transform(r.X[:2]).shape == r.X[:2].shape


def test_profile_roundtrip(tmp_path):
    r = synthetic.make_ss(n_per_class=8, n_sessions=1)
    prof = calibrate_user("P99", "ss", r, "td_features_lda", mode="subject")
    path = prof.save(tmp_path)
    loaded = UserProfile.load("P99", "ss", tmp_path)
    assert path.exists() and loaded.predict_label(r.X[0]) in r.class_names
    assert loaded.calib_cv_accuracy is not None


def test_pooled_rejects_channel_mismatch():
    mi, ss = synthetic.make_mi(n_per_class=4, n_subjects=1), synthetic.make_ss(n_per_class=4)
    try:
        fit_calibrated(ss, "td_features_lda", source=mi, mode="pooled")
    except ValueError as e:
        assert "channels" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_load_own_groups_and_exclude(tmp_path):
    for who, seed in [("P01", 0), ("P01", 1), ("P02", 2)]:
        r = synthetic.make_ss(n_per_class=3, n_sessions=1, seed=seed)
        np.savez(
            tmp_path / f"{who}_s{seed}.npz",
            X=r.X,
            y=r.y,
            fs=r.fs,
            ch_names=np.array(r.ch_names),
            classes=np.array(r.class_names),
            groups=np.zeros(len(r.y), int),
            participant=who,
        )
    rec = load_own(root=tmp_path)
    assert len(np.unique(rec.groups)) == 3  # one group per session file
    assert len(load_own(root=tmp_path, exclude_participant="P01").y) == 18


def test_calibration_curve_alignment_helps_on_shifted_data():
    out = calib_curve("ss", "synthetic_shift", "td_features_lda", ks=(5,))
    acc = out["balanced_accuracy"]
    assert acc["source_only_ea"][5] >= acc["source_only_raw"][5]
    assert acc["pooled_ea"][5] > out["chance"]
