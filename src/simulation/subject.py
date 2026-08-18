"""피험자 랜덤효과.

theta는 계획서 가설 2의 "신경효율성 개인차"다. 동일한 인지부하에서도
피험자마다 전전두엽 활성 크기가 다른 현상을 하나의 스칼라로 요약한다.
theta가 0이면 모든 피험자가 동일하게 반응하므로 LOSO가 within-subject만큼
쉬워진다. T4가 이 성질을 검증한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubjectProfile:
    subject_id: str
    theta: float


def make_subjects(
    n_subjects: int,
    subject_variance: float,
    rng: np.random.Generator,
) -> list[SubjectProfile]:
    """피험자 프로파일 목록을 만든다.

    theta ~ N(0, subject_variance). subject_variance는 분산이므로
    표준편차는 sqrt를 취한다.
    """
    if n_subjects < 1:
        raise ValueError(f"n_subjects must be >= 1, got {n_subjects}")
    if subject_variance < 0:
        raise ValueError(f"subject_variance must be >= 0, got {subject_variance}")

    sd = float(np.sqrt(subject_variance))
    thetas = rng.normal(0.0, sd, size=n_subjects) if sd > 0 else np.zeros(n_subjects)

    return [
        SubjectProfile(subject_id=f"sub-{i + 1:02d}", theta=float(t))
        for i, t in enumerate(thetas)
    ]
