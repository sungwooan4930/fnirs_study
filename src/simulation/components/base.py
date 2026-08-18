"""신호 성분 공통 정의.

결합 상수는 스펙 §6.2를 구현한다. 성분별 기여 비율을 코드에 고정하고
config의 effect_size가 그 전체를 스케일한다. 이렇게 두면 "effect_size를
키우면 모든 모달이 함께 강해진다"는 단순한 해석이 가능해진다.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

# 부하에 반응하는 EEG 채널 수. 실제로도 전전두 일부 채널만 반응하므로
# 전 채널을 변조하면 실제보다 쉬운 문제가 된다.
N_MODULATED_EEG: int = 10

# 성분별 기여 비율 (스펙 §6.2)
THETA_COUPLING: float = 1.0
ALPHA_COUPLING: float = 0.8
HBO_COUPLING: float = 0.8
BEHAV_COUPLING: float = 0.6


class SignalComponent(Protocol):
    """신호 성분의 호출 규약.

    모든 성분은 인지상태 궤적과 피험자 프로파일을 받아
    (n_channels, n_samples) 배열을 만든다.
    """

    def __call__(
        self,
        timeline,
        subject,
        rng: np.random.Generator,
        *,
        sfreq: float,
        n_channels: int,
        effect_size: float,
    ) -> np.ndarray: ...


def load_fraction(load_level: np.ndarray, n_levels: int = 3) -> np.ndarray:
    """부하 인덱스(0..n-1)를 0..1 비율로 바꾼다."""
    return load_level.astype(float) / max(n_levels - 1, 1)


def pink_noise(n_samples: int, n_channels: int, rng: np.random.Generator) -> np.ndarray:
    """1/f 배경 활동. 주파수 영역에서 1/sqrt(f)로 스케일해 생성한다."""
    white = rng.normal(size=(n_channels, n_samples))
    spectrum = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(n_samples)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    return np.fft.irfft(spectrum * scale, n=n_samples, axis=1)
