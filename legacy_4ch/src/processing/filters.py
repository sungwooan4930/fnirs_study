from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal


def bandpass_filter(
    data: np.ndarray,
    low_hz: float,
    high_hz: float,
    sampling_rate_hz: float,
    order: int = 6,  # default 6: sosfiltfilt doubles effective order (→12), needed for 0.01-0.5 Hz fNIRS bandpass
) -> np.ndarray:
    """Butterworth 밴드패스 필터. 1D 또는 2D (n_channels, n_samples) 배열 지원."""
    if low_hz <= 0:
        raise ValueError(f"low_hz must be positive, got {low_hz}")
    if high_hz <= low_hz:
        raise ValueError(f"high_hz ({high_hz}) must be greater than low_hz ({low_hz})")
    if high_hz >= sampling_rate_hz / 2:
        raise ValueError(
            f"high_hz ({high_hz}) must be less than Nyquist frequency ({sampling_rate_hz / 2})"
        )
    nyq = sampling_rate_hz / 2.0
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    sos = sp_signal.butter(order, [low, high], btype="band", output="sos")
    if data.ndim == 1:
        return sp_signal.sosfiltfilt(sos, data)
    return sp_signal.sosfiltfilt(sos, data, axis=1)


def baseline_correct(
    data: np.ndarray,
    baseline_sec: float,
    sampling_rate_hz: float,
) -> np.ndarray:
    """초반 baseline_sec 구간의 평균을 빼서 베이스라인을 보정한다."""
    n_baseline = int(baseline_sec * sampling_rate_hz)
    if n_baseline < 1:
        raise ValueError(
            f"baseline_sec={baseline_sec} with sampling_rate_hz={sampling_rate_hz} "
            f"yields fewer than 1 sample for baseline window"
        )
    if data.ndim == 1:
        baseline_mean = np.mean(data[:n_baseline])
        return data - baseline_mean
    baseline_mean = np.mean(data[:, :n_baseline], axis=1, keepdims=True)
    return data - baseline_mean
