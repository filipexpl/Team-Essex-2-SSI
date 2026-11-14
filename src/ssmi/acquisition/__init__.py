from dataclasses import dataclass, field

import numpy as np


@dataclass
class Recording:
    """Common container for epoched data from any source."""

    X: np.ndarray  # (n_trials, n_channels, n_samples)
    y: np.ndarray  # (n_trials,) int labels
    fs: float
    ch_names: list[str]
    class_names: list[str]
    groups: np.ndarray | None = None  # subject/session id per trial, used for splits
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.X.ndim == 3, "X must be (trials, channels, samples)"
        assert len(self.X) == len(self.y)
        if self.groups is None:
            self.groups = np.zeros(len(self.y), dtype=int)
