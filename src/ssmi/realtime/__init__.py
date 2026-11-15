"""Real-time decoding.

Every source exposes `latest(seconds) -> (channels, samples)` (newest data) and `fs`. The loop
polls it every `hop_s` seconds; decoders are stateless with respect to sample indices, so this
works with the BrainAccess Board database API (which returns the last N seconds) as well as LSL.

  Silent speech  : wait for an EMG onset, cut [onset - offset, + window] (offset learned at
                   calibration), classify, report label + confidence, then a refractory period.
  Motor imagery  : classify the last `epoch` seconds every hop, smooth probabilities (EMA),
                   report a class when it is confident and stable for `stable_hops` hops.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from ssmi.realtime.vad import envelope, first_onset, threshold_from_rest


@dataclass
class Event:
    t: float  # wall-clock (or virtual) time of the decision
    label: str  # class name, or "?" if below the confidence threshold
    confidence: float
    latency_ms: float  # processing time of this decision
    kind: str = "ss"


def _proba(profile, x: np.ndarray) -> np.ndarray:
    X = x[None]
    if profile.aligner is not None:
        X = profile.aligner.transform(X)
    if hasattr(profile.model, "predict_proba"):
        return profile.model.predict_proba(X)[0]
    p = np.zeros(len(profile.class_names))
    p[int(profile.model.predict(X)[0])] = 1.0
    return p


class SilentSpeechDecoder:
    def __init__(
        self,
        profile,
        fs: float,
        window_s: float = 1.5,
        k: float = 6.0,
        min_conf: float = 0.4,
        refractory_s: float | None = None,
    ):
        self.p, self.fs, self.window_s, self.k, self.min_conf = profile, fs, window_s, k, min_conf
        self.offset_s = getattr(profile, "onset_offset_s", None) or 0.3
        self.refractory_s = refractory_s if refractory_s is not None else window_s
        self.thr: float | None = None
        self.last_onset_t = -np.inf
        # how much history each poll needs: enough to hold a full window before the onset
        self.buffer_s = window_s + self.offset_s + 1.0

    def calibrate_rest(self, rest: np.ndarray) -> float:
        self.thr = threshold_from_rest(envelope(rest, self.fs), self.k)
        return self.thr

    def step(self, buf: np.ndarray, now: float) -> Event | None:
        assert self.thr is not None, "call calibrate_rest() first"
        env = envelope(buf, self.fs)
        n, off = int(self.window_s * self.fs), int(self.offset_s * self.fs)
        t_of = lambda i: now - (buf.shape[1] - i) / self.fs  # noqa: E731
        start = 0
        while True:
            o = first_onset(env, self.thr, self.fs, start=start)
            if o is None:
                return None
            if t_of(o) <= self.last_onset_t + self.refractory_s:
                start = o + 1  # already reported (or too close): look further
                continue
            a = o - off
            if a < 0:  # onset too close to buffer start: skip it
                start = o + 1
                continue
            if a + n > buf.shape[1]:
                return None  # window not complete yet: wait for the next poll
            t0 = time.perf_counter()
            pr = _proba(self.p, buf[:, a : a + n])
            ms = (time.perf_counter() - t0) * 1000
            self.last_onset_t = t_of(o)
            c = int(np.argmax(pr))
            label = self.p.class_names[c] if pr[c] >= self.min_conf else "?"
            return Event(now, label, float(pr[c]), ms, "ss")


class MotorImageryDecoder:
    def __init__(
        self,
        profile,
        fs: float,
        epoch_s: float = 3.0,
        alpha: float = 0.3,
        min_conf: float = 0.5,
        stable_hops: int = 3,
    ):
        self.p, self.fs, self.epoch_s = profile, fs, epoch_s
        self.alpha, self.min_conf, self.stable_hops = alpha, min_conf, stable_hops
        self.buffer_s = epoch_s
        self.ema: np.ndarray | None = None
        self.current, self.streak, self.reported = None, 0, None

    def calibrate_rest(self, rest: np.ndarray) -> None:  # interface symmetry
        return None

    def step(self, buf: np.ndarray, now: float) -> Event | None:
        n = int(self.epoch_s * self.fs)
        if buf.shape[1] < n:
            return None
        t0 = time.perf_counter()
        pr = _proba(self.p, buf[:, -n:])
        ms = (time.perf_counter() - t0) * 1000
        self.ema = pr if self.ema is None else self.alpha * pr + (1 - self.alpha) * self.ema
        c = int(np.argmax(self.ema))
        conf = float(self.ema[c])
        label = self.p.class_names[c] if conf >= self.min_conf else "rest/unsure"
        self.streak = self.streak + 1 if label == self.current else 1
        self.current = label
        if self.streak >= self.stable_hops and label != self.reported:
            self.reported = label
            return Event(now, label, conf, ms, "mi")
        return None


def make_decoder(profile, fs: float, **kw):
    if profile.task == "ss":
        return SilentSpeechDecoder(profile, fs, **kw)
    return MotorImageryDecoder(profile, fs, **kw)


def run(
    source,
    decoder,
    *,
    hop_s: float = 0.1,
    duration_s: float | None = None,
    rest_s: float = 5.0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    on_event=None,
    log_path=None,
    say=print,
) -> list[Event]:
    """Main loop. `clock`/`sleep` are injectable so tests can run on virtual time."""
    say(f"Stay relaxed for {rest_s:.0f} s (rest baseline)...")
    sleep(rest_s)
    decoder.calibrate_rest(source.latest(rest_s))
    say("Ready. Decoding continuously (Ctrl+C to stop).")
    events: list[Event] = []
    log = open(log_path, "a", encoding="utf-8") if log_path else None  # noqa: SIM115
    t_start = clock()
    try:
        while duration_s is None or clock() - t_start < duration_s:
            sleep(hop_s)
            ev = decoder.step(source.latest(decoder.buffer_s), clock())
            if ev:
                events.append(ev)
                if on_event:
                    on_event(ev)
                if log:
                    log.write(json.dumps(asdict(ev)) + "\n")
                    log.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if log:
            log.close()
    return events


class LSLOutput:
    """Publishes decoded labels as an LSL string stream 'SSMI_Decoded' for a demo UI."""

    def __init__(self, name: str = "SSMI_Decoded"):
        from pylsl import StreamInfo, StreamOutlet

        self.outlet = StreamOutlet(StreamInfo(name, "Markers", 1, 0, "string", f"ssmi_{name}"))

    def __call__(self, ev: Event) -> None:
        self.outlet.push_sample([ev.label])
