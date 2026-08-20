"""ERP 성분: P300 · N200.

계획서 §2.3의 ERP 후보 지표. 인지부하가 오르면 처리 자원이 줄어
P300 진폭이 감소한다. N200은 반대 부호의 초기 성분이다.
"""

from __future__ import annotations

import numpy as np

from src.simulation.components.base import N_MODULATED_EEG

N200_LATENCY_S = 0.20
P300_LATENCY_S = 0.30
N200_WIDTH_S = 0.04
P300_WIDTH_S = 0.06
N200_AMPLITUDE = -0.4
P300_AMPLITUDE = 1.0
LOAD_ATTENUATION = 0.6  # 최대 부하에서 P300이 이 비율만큼 줄어든다


def _gaussian(t: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def generate_eeg_erp(
    timeline,
    subject,
    rng: np.random.Generator,
    *,
    sfreq: float,
    n_channels: int,
    effect_size: float,
) -> np.ndarray:
    """자극 온셋에 정렬된 ERP를 (n_channels, n_samples)로 만든다."""
    n_samples = int(round(timeline.duration_s * sfreq))
    signal = np.zeros((n_channels, n_samples))

    # 자극당 1초짜리 ERP 템플릿을 겹쳐 놓는다
    template_len = int(round(1.0 * sfreq))
    t_local = np.arange(template_len) / sfreq

    n200 = N200_AMPLITUDE * _gaussian(t_local, N200_LATENCY_S, N200_WIDTH_S)
    p300_shape = _gaussian(t_local, P300_LATENCY_S, P300_WIDTH_S)

    loads = timeline.effective_load(timeline.stim_onsets)
    gain = 1.0 + subject.theta
    n_mod = min(N_MODULATED_EEG, n_channels)

    for onset, load in zip(timeline.stim_onsets, loads):
        start = int(round(onset * sfreq))
        end = min(start + template_len, n_samples)
        if start >= n_samples:
            continue
        span = end - start

        attenuation = 1.0 - effect_size * LOAD_ATTENUATION * gain * load
        attenuation = float(np.clip(attenuation, 0.05, None))
        wave = n200[:span] + P300_AMPLITUDE * attenuation * p300_shape[:span]

        signal[:n_mod, start:end] += wave

    return signal
