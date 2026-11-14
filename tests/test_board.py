"""BoardSource against a fake `brainaccess_board` module (no hardware, no Board app)."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from ssmi.acquisition.board import BoardControl, BoardSource, load_board_db


class FakeRaw:
    def __init__(self, data, fs, names, types):
        self._d, self.info, self.ch_names, self._t = data, {"sfreq": fs}, names, types
        self.n_times = data.shape[1]

    def get_channel_types(self):
        return self._t

    def get_data(self, picks=None):
        idx = [self.ch_names.index(p) for p in picks] if picks else slice(None)
        return self._d[idx]


class FakeDB:
    """Signal = sample index in volts*1e-6 on every channel, so windows are checkable."""

    def __init__(self, fs=500.0):
        self.fs, self.t0 = fs, time.time()
        self.names = [f"EMG{i}" for i in range(1, 9)] + ["ACC_X", "Sync"]
        self.types = ["emg"] * 8 + ["bio", "stim"]

    def get_mne(self, duration=None):
        n_all = int((time.time() - self.t0 + 10) * self.fs)
        n = n_all if duration is None else int(duration * self.fs)
        idx = np.arange(n_all - n, n_all, dtype=float)
        return {"EXG8-1234": FakeRaw(np.tile(idx * 1e-6, (10, 1)), self.fs, self.names, self.types)}


def fake_bb(log):
    def command(cmd):
        log.append(cmd)
        return {"message": "ok", **cmd}

    commands = {
        k: {"Command": k}
        for k in [
            "test",
            "get_devices",
            "status",
            "connect_device",
            "start_recording",
            "stop_recording",
        ]
    }
    return SimpleNamespace(
        db_connect=lambda filename=None: (FakeDB(), True),
        msg_connect=lambda: (SimpleNamespace(command=command), commands, True),
    )


def test_board_source_reads_signal_channels_in_uv():
    src = BoardSource(markers=False, lag_s=0.0, bb_module=fake_bb([]))
    assert src.fs == 500.0 and src.n_channels == 8 and "Sync" not in src.ch_names
    x = src.read(0.2, cue="yes")
    assert x.shape == (8, 100)
    assert np.allclose(np.diff(x[0]), 1.0)  # contiguous samples, converted to uV
    assert src.marker_log[-1][1] == "go:yes"


def test_device_selection_by_name():
    src = BoardSource(device="EXG8", markers=False, bb_module=fake_bb([]))
    assert src.device_id == "EXG8-1234"
    with pytest.raises(KeyError):
        BoardSource(device="MINI", markers=False, bb_module=fake_bb([]))


def test_board_control_copies_command_and_sets_fields():
    log = []
    ctl = BoardControl(bb_module=fake_bb(log))
    ctl.start_recording("/tmp/rec", "P07", "s1", 1, "ss-iitk")
    assert log[-1] == {
        "Command": "start_recording",
        "dir": "/tmp/rec",
        "subject": "P07",
        "session": "s1",
        "run": 1,
        "task": "ss-iitk",
    }
    assert ctl.commands["start_recording"] == {"Command": "start_recording"}  # not mutated
    with pytest.raises(KeyError):
        ctl.send("no_such_command")


def test_load_board_db():
    raws = load_board_db("x.db", bb_module=fake_bb([]))
    assert list(raws) == ["EXG8-1234"]
