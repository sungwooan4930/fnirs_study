"""시드 고정. 모든 난수는 여기서 만든 Generator를 통해서만 생성한다."""

from __future__ import annotations

import random

import numpy as np


def set_all_seeds(seed: int) -> np.random.Generator:
    """시드를 고정하고 numpy Generator를 반환한다.

    전역 np.random과 random도 함께 고정하지만, 프로젝트 코드는 반환된
    Generator만 사용해야 한다. 전역 고정은 서드파티가 내부적으로 전역
    상태를 쓰는 경우를 위한 방어일 뿐이다.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
