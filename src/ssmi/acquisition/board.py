"""BrainAccess Board adapter (Board Python API 1.0.2, package `brainaccess_board`).

Documented calls used here (brainaccess.ai/documentation/board-python-api/1.0.2):
  bb.db_connect()                -> (db, ok); db.get_mne(duration=s) -> {device_id: mne.io.RawArray}
  bb.db_connect(filename=...)    -> read a recorded .db file
  bb.msg_connect()               -> (board_control, commands, ok)
  board_control.command(dict)    -> reply dict
  commands: test, get_devices, status, connect_device, disconnect_device,
            start_recording(dir, subject, session, run, task), stop_recording

Markers: the docs attach a marker stream as an LSL device via `connect_device` (port = LSL uid)
but do not show how to push a marker, so we create our own pylsl marker outlet and attach it the
same way. Board must be running with the amplifier connected before any of this.

NOT verified on hardware (check with `python scripts/board_check.py`):
  - units returned by get_mne (MNE convention is volts; we convert to microvolts),
  - how far the live database lags behind real time (`lag_s`),
  - whether recorded markers appear as annotations in get_mne() of a recorded .db.
"""

from __future__ import annotations

import importlib
import time
import warnings

import numpy as np

from ssmi.acquisition.channels import select_channels

SIGNAL_TYPES = ("eeg", "emg", "ecg", "eog", "seeg", "misc")


def _import_bb(bb_module=None):
    return bb_module or importlib.import_module("brainaccess_board")


def _signal_channels(raw) -> list[str]:
    """Signal channels only (drop stim/marker/accelerometer-like channels if typed as such)."""
    try:
        types = raw.get_channel_types()
    except Exception:  # noqa: BLE001 - older/mock objects
        return list(raw.ch_names)
    keep = [n for n, t in zip(raw.ch_names, types, strict=True) if t in SIGNAL_TYPES]
    return keep or list(raw.ch_names)


class BoardControl:
    """Thin wrapper over msg_connect(): recording control and device commands."""

    def __init__(self, bb_module=None):
        bb = _import_bb(bb_module)
        self.bc, self.commands, ok = bb.msg_connect()
        if not ok:
            raise RuntimeError("Could not connect to BrainAccess Board (is it running?)")

    def send(self, name: str, **fields) -> dict:
        if name not in self.commands:
            raise KeyError(f"Board has no command '{name}'. Available: {sorted(self.commands)}")
        cmd = dict(self.commands[name])  # copy before changing fields (as in the docs)
        cmd.update(fields)
        return self.bc.command(cmd)

    def start_recording(self, directory: str | None, subject, session, run, task: str) -> dict:
        fields = {"subject": subject, "session": session, "run": run, "task": task}
        if directory:
            fields["dir"] = directory
        return self.send("start_recording", **fields)

    def stop_recording(self) -> dict:
        return self.send("stop_recording")


class MarkerOutlet:
    """Our LSL marker stream, attached to Board so markers are saved with the recording."""

    def __init__(self, control: BoardControl | None, name: str = "SSMI_Markers"):
        from pylsl import StreamInfo, StreamOutlet, local_clock

        info = StreamInfo(name, "Markers", 1, 0, "string", f"ssmi_{name}")
        self.outlet = StreamOutlet(info)
        self.uid = self.outlet.get_info().uid()
        self.clock = local_clock
        self.attached = False
        if control is not None:
            reply = control.send("connect_device", port=self.uid)
            self.attached = True
            self.reply = reply

    def push(self, label: str) -> float:
        t = self.clock()
        self.outlet.push_sample([label])
        return t


class BoardSource:
    """Live source with the same interface as the other sources:
    attributes fs, n_channels, ch_names; read(seconds, cue) -> (n_channels, n_samples) in uV.

    `channels`: None/'auto' keeps electrode channels only (drops contact flags, IMU, counters,
    battery and constant channels); or a list of names / 're:<pattern>'.

    read() pushes a 'go:<cue>' marker, waits `seconds + lag_s`, then takes the window that started
    at the marker from the newest data. Increase `lag_s` if the database lags behind real time.
    """

    def __init__(
        self,
        device: str | int | None = None,
        channels=None,
        markers: bool = True,
        lag_s: float = 0.25,
        to_microvolts: bool = True,
        bb_module=None,
    ):
        bb = _import_bb(bb_module)
        self.db, ok = bb.db_connect()
        if not ok:
            raise RuntimeError("Could not open the live Board database (is Board running?)")
        raws = self.db.get_mne(duration=1)
        if not raws:
            raise RuntimeError("Board reports no connected device: connect the amplifier first")
        ids = list(raws)
        if device is None:
            self.device_id = ids[0]
        elif isinstance(device, int):
            self.device_id = ids[device]
        else:
            matches = [i for i in ids if str(device) in str(i)]
            if not matches:
                raise KeyError(f"No device matching '{device}'. Devices: {ids}")
            self.device_id = matches[0]
        raw = raws[self.device_id]
        self.fs = float(raw.info["sfreq"])
        typed = _signal_channels(raw)
        x1 = raw.get_data(picks=typed)
        self.ch_names = [typed[i] for i in select_channels(typed, channels, x1)]
        self.n_channels = len(self.ch_names)
        self.lag_s, self.scale = lag_s, (1e6 if to_microvolts else 1.0)
        self.devices = ids
        self.control = None
        self.markers = None
        self.marker_log: list[tuple[float, str]] = []
        if markers:
            try:
                self.control = BoardControl(bb)
                self.markers = MarkerOutlet(self.control)
            except Exception as e:  # noqa: BLE001 - markers are optional
                warnings.warn(f"Markers disabled: {e}", stacklevel=2)

    def mark(self, label: str) -> None:
        t = self.markers.push(label) if self.markers else time.time()
        self.marker_log.append((t, label))

    def latest(self, seconds: float) -> np.ndarray:
        raw = self.db.get_mne(duration=seconds)[self.device_id]
        return raw.get_data(picks=self.ch_names) * self.scale

    def read(self, seconds: float, cue: str | None = None) -> np.ndarray:
        n = int(round(seconds * self.fs))
        self.mark(f"go:{cue}" if cue else "rest")
        time.sleep(seconds + self.lag_s)
        x = self.latest(seconds + self.lag_s)
        if x.shape[1] < n:
            raise RuntimeError(
                f"Got {x.shape[1]} samples, expected {n}: packet loss or wrong lag_s/fs"
            )
        return x[:, -int(round((seconds + self.lag_s) * self.fs)) :][:, :n]


def load_board_db(filename: str, bb_module=None) -> dict:
    """Recorded Board database -> {device_id: mne Raw} (documented db_connect(filename=...))."""
    bb = _import_bb(bb_module)
    db, ok = bb.db_connect(filename=filename)
    if ok is False:
        raise RuntimeError(f"Could not open {filename}")
    return db.get_mne()
