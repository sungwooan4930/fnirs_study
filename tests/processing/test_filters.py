import numpy as np
import pytest
from src.processing.filters import bandpass_filter, baseline_correct


def test_bandpass_removes_dc():
    """DC 성분(0Hz)을 포함한 신호에서 밴드패스 후 DC가 제거되어야 한다."""
    sr = 100.0
    t = np.arange(0, 10, 1 / sr)
    signal = 5.0 + np.sin(2 * np.pi * 0.1 * t)
    filtered = bandpass_filter(signal, low_hz=0.01, high_hz=0.5, sampling_rate_hz=sr)
    assert abs(np.mean(filtered)) < 0.5


def test_bandpass_preserves_in_band_signal():
    """밴드 내 신호(0.1Hz)는 필터 후에도 보존되어야 한다."""
    sr = 100.0
    t = np.arange(0, 30, 1 / sr)
    signal = np.sin(2 * np.pi * 0.1 * t)
    filtered = bandpass_filter(signal, low_hz=0.01, high_hz=0.5, sampling_rate_hz=sr)
    assert np.std(filtered[500:]) > 0.3


def test_bandpass_attenuates_high_freq():
    """밴드 외 고주파(10Hz)는 크게 감쇠되어야 한다."""
    sr = 100.0
    t = np.arange(0, 10, 1 / sr)
    signal = np.sin(2 * np.pi * 10.0 * t)
    filtered = bandpass_filter(signal, low_hz=0.01, high_hz=0.5, sampling_rate_hz=sr)
    assert np.std(filtered) < 0.1


def test_baseline_correct_removes_mean():
    """베이스라인 보정 후 초반 구간 평균이 0에 가까워야 한다."""
    sr = 10.0
    baseline_sec = 5.0
    signal = np.ones(100) * 3.0
    corrected = baseline_correct(signal, baseline_sec=baseline_sec, sampling_rate_hz=sr)
    assert abs(np.mean(corrected)) < 0.01


def test_filters_work_on_2d_array():
    """여러 채널(2D 배열)에 대해 필터가 동작해야 한다."""
    sr = 100.0
    t = np.arange(0, 10, 1 / sr)
    signals = np.stack([np.sin(2 * np.pi * 0.1 * t) for _ in range(4)])  # (4, 1000)
    filtered = bandpass_filter(signals, low_hz=0.01, high_hz=0.5, sampling_rate_hz=sr)
    assert filtered.shape == signals.shape
