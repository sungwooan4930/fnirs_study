"""fNIRS 성분: 신경활성 → HRF 컨볼루션 → HbO/HbR.

의도적 단순화 (스펙 §6.3):
- hbr = hbr_coupling * hbo + 독립잡음. 실제 HbR은 진폭이 HbO의 약 1/3이면서
  시간 지연도 다르지만, 여기서는 지연 차이를 모델링하지 않는다.
- 정준 HRF 하나만 쓴다. 실제 HRF는 개인·부위별로 다르다.
- 채널 간 공간 상관을 넣지 않는다.
이 단순화들은 테스트베드 목적(하네스 검증)에는 무해하다. 검증 대상은
신호의 현실성이 아니라 "심은 효과를 하네스가 정직하게 회수하는가"이기 때문이다.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.simulation.components.base import HBO_COUPLING

HRF_DURATION_S = 32.0
NOISE_SD = 0.15


def canonical_hrf(sfreq: float) -> np.ndarray:
    """이중 감마 정준 혈류반응함수. 최대치가 1이 되도록 정규화한다."""
    t = np.arange(0.0, HRF_DURATION_S, 1.0 / sfreq)
    peak = stats.gamma.pdf(t, a=6.0)
    undershoot = stats.gamma.pdf(t, a=16.0)
    hrf = peak - undershoot / 6.0
    return hrf / np.max(hrf)


def generate_fnirs(
    timeline,
    subject,
    rng: np.random.Generator,
    *,
    sfreq: float,
    n_channels: int,
    effect_size: float,
    hbr_coupling: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(hbo, hbr)을 각각 (n_channels, n_samples)로 만든다."""
    n_samples = int(round(timeline.duration_s * sfreq))
    t = np.arange(n_samples) / sfreq

    neural = timeline.effective_load(t)
    hrf = canonical_hrf(sfreq)

    gain = 1.0 + subject.theta
    response = np.convolve(neural, hrf)[:n_samples]
    response = response / max(len(hrf) / 4.0, 1.0)  # 컨볼루션 누적을 정규화
    response = effect_size * HBO_COUPLING * gain * response

    hbo = np.tile(response, (n_channels, 1))
    hbo += rng.normal(0.0, NOISE_SD, size=hbo.shape)

    hbr = hbr_coupling * hbo + rng.normal(0.0, NOISE_SD * 0.5, size=hbo.shape)

    return hbo, hbr
