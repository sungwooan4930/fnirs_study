"""EEG 진동 성분: 1/f 배경 + θ·α 진동.

인지부하가 오르면 전두 θ가 증가하고 두정 α가 감소한다는 계획서 §2.3의
후보 지표를 구현한다. 피험자의 theta(신경효율성)가 반응 크기를 개인별로
바꾼다.
"""

from __future__ import annotations

import numpy as np

from src.simulation.components.base import (
    ALPHA_COUPLING,
    N_MODULATED_EEG,
    THETA_COUPLING,
    pink_noise,
)

THETA_HZ = 6.0
ALPHA_HZ = 10.0
BASE_AMPLITUDE = 1.0
BACKGROUND_SCALE = 0.5


def generate_eeg_oscillation(
    timeline,
    subject,
    rng: np.random.Generator,
    *,
    sfreq: float,
    n_channels: int,
    effect_size: float,
) -> np.ndarray:
    """(n_channels, n_samples) EEG 신호를 만든다."""
    n_samples = int(round(timeline.duration_s * sfreq))
    t = np.arange(n_samples) / sfreq

    signal = BACKGROUND_SCALE * pink_noise(n_samples, n_channels, rng)

    load = timeline.effective_load(t)
    # 개인차: theta가 클수록 같은 부하에 더 크게 반응한다
    gain = 1.0 + subject.theta

    theta_amp = BASE_AMPLITUDE * (1.0 + effect_size * THETA_COUPLING * gain * load)
    alpha_amp = BASE_AMPLITUDE * (1.0 - effect_size * ALPHA_COUPLING * gain * load)
    alpha_amp = np.clip(alpha_amp, 0.05, None)

    n_mod = min(N_MODULATED_EEG, n_channels)
    for ch in range(n_mod):
        phase_t = rng.uniform(0, 2 * np.pi)
        phase_a = rng.uniform(0, 2 * np.pi)
        signal[ch] += theta_amp * np.sin(2 * np.pi * THETA_HZ * t + phase_t)
        signal[ch] += alpha_amp * np.sin(2 * np.pi * ALPHA_HZ * t + phase_a)

    return signal
