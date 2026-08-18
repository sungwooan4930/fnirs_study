"""행동 성분: 정답 여부와 반응시간.

시각 u의 행동은 시각 (u - lead_delta_s)의 인지상태가 결정한다.
따라서 시각 w의 뇌신호로 시각 (w + lead_delta_s)의 행동을 예측할 수 있다.
이것이 계획서 가설 2("행동 오류 발현 약 1.2초 전 선행 신호")의 구현이며,
행동을 입력 특징이자 동시점 예측 타깃으로 쓸 때 생기는 라벨 누수를 피한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulation.components.base import BEHAV_COUPLING, load_fraction

BASE_ACCURACY = 0.95
ACCURACY_DROP = 0.35   # 최대 부하에서 정답률이 이만큼 떨어진다
BASE_RT_S = 0.55
RT_RISE_S = 0.40       # 최대 부하에서 RT가 이만큼 늘어난다
RT_NOISE_SD = 0.08


@dataclass(frozen=True)
class BehaviorLog:
    onsets: np.ndarray   # (n_stim,) 자극 제시 시각(초)
    correct: np.ndarray  # (n_stim,) 0/1
    rt: np.ndarray       # (n_stim,) 초


def generate_behavior(
    timeline,
    subject,
    rng: np.random.Generator,
    *,
    effect_size: float,
    lead_delta_s: float,
) -> BehaviorLog:
    """자극별 정답 여부와 반응시간을 만든다."""
    if lead_delta_s < 0:
        raise ValueError(f"lead_delta_s must be >= 0, got {lead_delta_s}")

    onsets = timeline.stim_onsets
    # 행동을 구동하는 것은 Δ만큼 앞선 시점의 상태다
    driving_load = load_fraction(timeline.load_at(onsets - lead_delta_s))

    gain = 1.0 + subject.theta
    scale = effect_size * BEHAV_COUPLING * gain * driving_load

    p_correct = np.clip(BASE_ACCURACY - ACCURACY_DROP * scale, 0.05, 0.99)
    correct = (rng.random(len(onsets)) < p_correct).astype(int)

    rt = BASE_RT_S + RT_RISE_S * scale + rng.normal(0.0, RT_NOISE_SD, size=len(onsets))
    rt = np.clip(rt, 0.05, None)

    return BehaviorLog(onsets=onsets, correct=correct, rt=rt)
