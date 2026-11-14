"""Channel selection: keep the electrode channels, drop everything else the device streams.

The BrainAccess Board reports every stream channel with a signal type, including contact flags
(`contact_F1`...), IMU (`Accel_*`, `Gyro_*`), a sample counter, a streaming flag and the battery
(seen on EXG8, 2026-11-14: 25 channels, of which F1-F8 are EMG). Those must never reach a model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

NON_SIGNAL = re.compile(
    r"^(contact|impedance|imp_|accel|acc_|gyro|sample|streaming|battery|bat_|marker|trigger|"
    r"trig|sync|stim|time|counter|status|digital)",
    re.IGNORECASE,
)


def parse_spec(spec: str | Sequence[str] | None) -> list[str] | str | None:
    """CLI helper: 'auto' / None -> auto; 'F1,F2' -> list; 're:^F' -> regex string."""
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("", "auto")):
        return None
    if isinstance(spec, str) and spec.startswith("re:"):
        return spec
    if isinstance(spec, str):
        return [s.strip() for s in spec.split(",") if s.strip()]
    return list(spec)


def select_channels(ch_names: Sequence[str], spec=None, X: np.ndarray | None = None) -> list[int]:
    """Indices of channels to keep.

    spec None/'auto': names not matching NON_SIGNAL and, if X is given, not constant.
    spec list of names: exactly those, in that order (KeyError if one is missing).
    spec 're:<pattern>': names matching the regular expression.
    X: data with channels on axis -2, i.e. (channels, samples) or (trials, channels, samples).
    """
    names = list(ch_names)
    spec = parse_spec(spec)
    if isinstance(spec, list):
        missing = [s for s in spec if s not in names]
        if missing:
            raise KeyError(f"Channels {missing} not found. Available: {names}")
        return [names.index(s) for s in spec]
    if isinstance(spec, str):
        rx = re.compile(spec[3:])
        idx = [i for i, n in enumerate(names) if rx.search(n)]
    else:
        idx = [i for i, n in enumerate(names) if not NON_SIGNAL.match(n)]
    if X is not None and idx:
        Xc = np.moveaxis(np.asarray(X), -2, 0).reshape(len(names), -1)
        idx = [i for i in idx if np.isfinite(Xc[i]).all() and Xc[i].std() > 0]
    if not idx:
        raise ValueError(f"No signal channels left after selection from {names}")
    return idx
