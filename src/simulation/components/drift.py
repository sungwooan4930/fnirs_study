"""드리프트 적용 — 깨끗한 신호에 측정 왜곡을 얹는다.

기존 성분 플러그인은 손대지 않는다. 그들은 계속 '깨끗한 신호'를 만들고,
드리프트는 그 뒤에 곱해진다. 이 분리가 스펙 §5.1의 층 구분을 코드로
지키는 방식이다 — 여기서 하는 일은 전부 신호 변환이며 인지상태는
건드리지 않는다.

곱셈 이득이 핵심이다. 광 결합도는 곱셈적이라, 가산 오프셋만 심으면
정규화 난이도가 비현실적으로 쉬워진다(평균 빼기로 끝난다).
"""

from __future__ import annotations

import numpy as np


def apply_channel_drift(
    x: np.ndarray,
    *,
    gain: np.ndarray,
    offset: np.ndarray,
    within_rate: float,
) -> np.ndarray:
    """`y = gain(t) * x + offset`, `gain(t) = gain * (1 + within_rate * t/T)`.

    오프셋은 램프의 대상이 아니다. 세션 내 드리프트는 옵토드가 서서히
    밀리며 결합도가 변하는 현상이므로 곱셈 항에만 걸린다.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (n_channels, n_samples), got shape {x.shape}")
    n_ch, n_samples = x.shape
    if len(gain) != n_ch:
        raise ValueError(f"gain length {len(gain)} != n_channels {n_ch}")
    if len(offset) != n_ch:
        raise ValueError(f"offset length {len(offset)} != n_channels {n_ch}")

    ramp = 1.0 + within_rate * (np.arange(n_samples) / max(n_samples - 1, 1))
    return x * np.asarray(gain)[:, None] * ramp[None, :] + np.asarray(offset)[:, None]


def apply_noise_scaling(
    x: np.ndarray,
    *,
    noise_scale: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """채널별 잡음을 더한다 — 임피던스 상승 모사.

    임피던스가 오르면 진폭이 아니라 **잡음**이 커진다. 그래서 이득과
    분리해 가산으로 얹는다.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (n_channels, n_samples), got shape {x.shape}")
    if len(noise_scale) != x.shape[0]:
        raise ValueError(
            f"noise_scale length {len(noise_scale)} != n_channels {x.shape[0]}"
        )
    noise = rng.normal(0.0, 1.0, size=x.shape) * np.asarray(noise_scale)[:, None]
    return x + noise
