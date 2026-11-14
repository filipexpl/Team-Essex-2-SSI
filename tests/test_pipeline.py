import numpy as np
import pytest

from ssmi.acquisition import synthetic
from ssmi.evals.metrics import itr_bits_per_min
from ssmi.evals.run import run
from ssmi.models import REGISTRY, get_model
from ssmi.preprocessing import bandpass, resample


def test_synthetic_shapes():
    r = synthetic.make_mi(n_per_class=5, n_subjects=2)
    assert r.X.shape[1:] == (8, 750) and set(r.y) == {0, 1, 2, 3}
    s = synthetic.make_ss(n_per_class=5, n_sessions=2)
    assert s.X.shape[1] == 8


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_models_fit_predict(name):
    r = synthetic.make_mi(n_per_class=10, n_subjects=1)
    m = get_model(name, r.fs).fit(r.X, r.y)
    assert m.predict(r.X[:3]).shape == (3,)


def test_bandpass_respects_nyquist():
    x = np.random.default_rng(0).standard_normal((2, 8, 500))
    assert bandpass(x, 250.0, 20, 450).shape == x.shape  # hi clipped below 125 Hz


def test_resample_gaddy_to_device():
    x = np.zeros((1, 8, 1000))
    assert resample(x, 1000, 250).shape[-1] == 250


def test_itr():
    assert itr_bits_per_min(0.25, 4, 4.0) == 0.0
    assert itr_bits_per_min(1.0, 4, 4.0) == pytest.approx(30.0, rel=1e-3)


def test_eval_no_leakage_flag():
    out = run("mi", "synthetic", "csp_lda")
    assert out["groups_independent"] is True
    assert out["balanced_accuracy"] > out["chance"]
