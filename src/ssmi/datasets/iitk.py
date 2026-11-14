"""IITK silent speech sEMG (Jain et al.).

Source: github.com/DarshanJain295/Silent-Speech-recognition-dataset

OpenBCI Cyton, 8 monopolar channels @ 250 Hz, one speaker, 22-word English vocabulary.
CSV columns: 8 x EMG (uV) + marker (1 = utterance start, 2 = end; sentences: 1 = boundaries).

  words-rec/<word>.csv          ~20 isolated repetitions per word, ONE FILE PER WORD
  real-sentence-rec/<a_b_c>.csv one sentence, markers at the (n_words + 1) word boundaries

Caveat: in words-rec each class is a separate recording, so slow drift differs per class and
can leak label information. Use `iitk_sent` (words cut out of sentences) as the honest test:
`python -m ssmi.evals.transfer --task ss --train iitk --test iitk_sent`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ssmi.acquisition import Recording

FS = 250.0
VOCAB = [
    "all", "anyone", "anything", "are", "back", "can", "come", "could", "did", "doing", "find",
    "help", "here", "in", "now", "out", "please", "right", "stop", "that", "tomorrow", "you",
]  # fmt: skip
CHANNELS = [f"IITK{i}" for i in range(1, 9)]  # exact muscle order per channel not documented


def _root(root) -> Path:
    return Path(root or Path(__file__).resolve().parents[3] / "data" / "raw" / "iitk")


def _cut(emg: np.ndarray, onset: int, n: int, pre: int) -> np.ndarray | None:
    a = onset - pre
    if a < 0 or a + n > len(emg):
        return None
    return emg[a : a + n].T


def load_iitk_words(
    root=None, window_s: float = 1.2, pre_s: float = 0.1, n_blocks: int = 4
) -> Recording:
    """Isolated words. groups = chronological repetition block (0..n_blocks-1) within each word,
    so leave-one-group-out tests on repetitions recorded at another time."""
    root = _root(root) / "words-rec"
    if not root.exists():
        raise FileNotFoundError(f"{root} missing — run `make data` (clones the IITK repo)")
    n, pre = int(window_s * FS), int(pre_s * FS)
    X, y, g = [], [], []
    for c, word in enumerate(VOCAB):
        a = np.loadtxt(root / f"{word}.csv", delimiter=",")
        emg, m = a[:, :8], a[:, 8]
        starts = np.flatnonzero(m == 1)
        trials = [t for s in starts if (t := _cut(emg, s, n, pre)) is not None]
        for i, t in enumerate(trials):
            X.append(t)
            y.append(c)
            g.append(min(i * n_blocks // len(trials), n_blocks - 1))
    return Recording(
        np.asarray(X),
        np.asarray(y),
        FS,
        CHANNELS,
        VOCAB,
        np.asarray(g),
        meta={"source": "iitk_words", "window_s": window_s},
    )


def load_iitk_sentences(root=None, window_s: float = 1.2, pre_s: float = 0.1) -> Recording:
    """Words cut out of continuously mouthed sentences (word boundaries from markers).
    Files whose marker count != n_words + 1 are skipped. groups = sentence id."""
    root = _root(root) / "real-sentence-rec"
    n, pre = int(window_s * FS), int(pre_s * FS)
    X, y, g, skipped = [], [], [], []
    for si, f in enumerate(sorted(root.glob("*.csv"))):
        words = f.stem.split("_")
        a = np.loadtxt(f, delimiter=",")
        b = np.flatnonzero(a[:, 8] != 0)
        if len(b) != len(words) + 1 or any(w not in VOCAB for w in words):
            skipped.append(f.stem)
            continue
        for w, onset in zip(words, b[:-1], strict=True):
            t = _cut(a[:, :8], onset, n, pre)
            if t is not None:
                X.append(t)
                y.append(VOCAB.index(w))
                g.append(si)
    return Recording(
        np.asarray(X),
        np.asarray(y),
        FS,
        CHANNELS,
        VOCAB,
        np.asarray(g),
        meta={"source": "iitk_sentences", "skipped": skipped},
    )
