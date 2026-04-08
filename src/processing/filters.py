from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal


def bandpass_filter(
    data: np.ndarray,
    low_hz: float,
    high_hz: float,
    sampling_rate_hz: float,
    order: int = 6,
) -> np.ndarray:
    """Butterworth 밴드패스 필터. 1D 또는 2D (n_channels, n_samples) 배열 지원."""
    nyq = sampling_rate_hz / 2.0
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    sos = sp_signal.butter(order, [low, high], btype="band", output="sos")
    if data.ndim == 1:
        return sp_signal.sosfiltfilt(sos, data)
    return np.apply_along_axis(
        lambda ch: sp_signal.sosfiltfilt(sos, ch), axis=1, arr=data
    )


def baseline_correct(
    data: np.ndarray,
    baseline_sec: float,
    sampling_rate_hz: float,
) -> np.ndarray:
    """초반 baseline_sec 구간의 평균을 빼서 베이스라인을 보정한다."""
    n_baseline = int(baseline_sec * sampling_rate_hz)
    if data.ndim == 1:
        baseline_mean = np.mean(data[:n_baseline])
        return data - baseline_mean
    baseline_mean = np.mean(data[:, :n_baseline], axis=1, keepdims=True)
    return data - baseline_mean
