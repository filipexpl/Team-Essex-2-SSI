"""Real-time loop on a simulated person, in virtual time (fast, no hardware)."""

import numpy as np
import pytest

from ssmi.acquisition import Recording
from ssmi.acquisition.sources import SyntheticSource
from ssmi.acquisition.stream_sim import SimulatedStream
from ssmi.calibration import calibrate_user
from ssmi.realtime import make_decoder, run
from ssmi.realtime.vad import envelope, first_onset, threshold_from_rest

FS = 500.0
WORDS = ["yes", "no", "up", "down", "left", "right"]
MI = ["left_hand", "right_hand", "both_hands", "feet"]


def _profile(task, classes, n=12):
    src = SyntheticSource(task, classes, fs=FS, seed=123)
    X, y = [], []
    for c in np.repeat(np.arange(len(classes)), n):
        x = src.read(1.5 if task == "ss" else 4.0, cue=classes[c])
        X.append(x if task == "ss" else x[:, int(0.5 * FS) : int(3.5 * FS)])
        y.append(c)
    rec = Recording(np.array(X), np.array(y), FS, list(src.ch_names), classes)
    model = "td_features_lda" if task == "ss" else "csp_lda"
    return calibrate_user("SIM", task, rec, model, mode="subject")


def test_onset_detector_finds_burst():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, int(3 * FS)))
    x[:, int(1.2 * FS) : int(1.6 * FS)] *= 8
    env = envelope(x, FS)
    thr = threshold_from_rest(envelope(rng.standard_normal((8, int(2 * FS))), FS))
    o = first_onset(env, thr, FS)
    assert o is not None and abs(o / FS - 1.2) < 0.1


def test_profile_stores_onset_offset():
    prof = _profile("ss", WORDS, n=6)
    assert prof.onset_offset_s is not None and 0 <= prof.onset_offset_s < 1.5


def test_realtime_silent_speech_detects_every_word():
    prof = _profile("ss", WORDS)
    st = SimulatedStream("ss", WORDS, fs=FS, seconds=40)
    ev = run(
        st,
        make_decoder(prof, FS),
        hop_s=0.1,
        duration_s=st.duration - 6,
        clock=st.clock,
        sleep=st.sleep,
        say=lambda *a: None,
    )
    truth = [(t, w) for t, w in st.truth if t >= 5.0]
    matched = [next((e for e in ev if t <= e.t <= t + 3.0), None) for t, _ in truth]
    assert all(m is not None for m in matched)  # every word detected
    assert len(ev) == len(truth)  # no false triggers in rest
    correct = sum(m.label == w for m, (_, w) in zip(matched, truth, strict=True))
    assert correct / len(truth) > 1 / len(WORDS) * 2  # well above chance
    assert max(e.latency_ms for e in ev) < 100


@pytest.mark.parametrize("hop", [0.25])
def test_realtime_motor_imagery_reports_changes(hop):
    prof = _profile("mi", MI, n=10)
    st = SimulatedStream("mi", MI, fs=FS, seconds=60)
    ev = run(
        st,
        make_decoder(prof, FS),
        hop_s=hop,
        duration_s=st.duration - 6,
        clock=st.clock,
        sleep=st.sleep,
        say=lambda *a: None,
    )
    assert ev and all(e.kind == "mi" for e in ev)
    assert all(a.label != b.label for a, b in zip(ev, ev[1:], strict=False))  # only changes
