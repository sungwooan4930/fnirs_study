from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class ConcentrationIndex(ABC):
    """집중도 지수 계산 플러그인 인터페이스.

    논문 기반 알고리즘이 확정되면 이 클래스를 상속해 구현한다.
    """

    @abstractmethod
    def compute(self, hbo: np.ndarray, hbr: np.ndarray) -> float:
        """HbO/HbR 배열로부터 집중도 지수 [0.0, 1.0]을 계산한다.

        Args:
            hbo: shape (n_channels,), 현재 시점 HbO 값 (μmol/L)
            hbr: shape (n_channels,), 현재 시점 HbR 값 (μmol/L)

        Returns:
            집중도 지수 [0.0, 1.0]
        """
        ...


class SimpleHbOIndex(ConcentrationIndex):
    """단순 HbO 평균 기반 집중도 지수 (placeholder).

    전전두엽 HbO 증가 = 집중도 증가 가정.
    논문 기반 알고리즘 확정 후 교체 예정.
    """

    _MAX_HBO_UMOL = 10.0  # 정규화 기준 (μmol/L)

    def compute(self, hbo: np.ndarray, hbr: np.ndarray) -> float:
        mean_hbo = float(np.mean(hbo))
        normalized = mean_hbo / self._MAX_HBO_UMOL
        return float(np.clip(normalized, 0.0, 1.0))
