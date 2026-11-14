import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, tf2sos


def bandpass(X: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    hi = min(hi, 0.45 * fs)  # never exceed Nyquist (important: EMG at fs=250)
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def notch(X: np.ndarray, fs: float, f0: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Remove mains (50 Hz in Poland) and its harmonics below Nyquist."""
    out = X
    k = 1
    while f0 * k < fs / 2 - 1:
        b, a = iirnotch(f0 * k, q, fs=fs)
        out = sosfiltfilt(tf2sos(b, a), out, axis=-1)
        k += 1
    return out


def resample(X: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample along time, e.g. Gaddy 1000 Hz -> device fs before transfer learning."""
    from fractions import Fraction

    fr = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(X, fr.numerator, fr.denominator, axis=-1)
