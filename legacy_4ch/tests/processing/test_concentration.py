import numpy as np
import pytest
from src.processing.concentration import ConcentrationIndex, SimpleHbOIndex


def test_concentration_index_is_abstract():
    with pytest.raises(TypeError):
        ConcentrationIndex()


def test_simple_hbo_index_returns_float():
    idx = SimpleHbOIndex()
    hbo = np.array([0.5, 0.3, 0.1, 0.4])  # 4 channels per actual spec
    hbr = np.array([-0.1, -0.05, 0.0, -0.2])
    result = idx.compute(hbo, hbr)
    assert isinstance(result, float)


def test_simple_hbo_index_range():
    """결과값은 [0.0, 1.0] 범위 이내여야 한다."""
    idx = SimpleHbOIndex()
    hbo = np.array([1.0, 2.0, 3.0, 4.0])  # 4 channels
    hbr = np.array([-1.0] * 4)
    result = idx.compute(hbo, hbr)
    assert 0.0 <= result <= 1.0


def test_simple_hbo_high_hbo_gives_high_index():
    """HbO가 높을수록 집중도 지수가 높아야 한다."""
    idx = SimpleHbOIndex()
    hbo_high = np.array([2.0] * 4)
    hbo_low = np.array([0.1] * 4)
    hbr = np.zeros(4)
    assert idx.compute(hbo_high, hbr) > idx.compute(hbo_low, hbr)


def test_plugin_is_swappable():
    """ConcentrationIndex를 상속하면 compute 메서드를 통해 동일하게 사용 가능."""
    class CustomIndex(ConcentrationIndex):
        def compute(self, hbo: np.ndarray, hbr: np.ndarray) -> float:
            return 0.42

    idx: ConcentrationIndex = CustomIndex()
    result = idx.compute(np.zeros(4), np.zeros(4))
    assert result == 0.42


def test_simple_hbo_clips_at_one():
    """HbO가 _MAX_HBO_UMOL 초과해도 1.0을 반환해야 한다."""
    idx = SimpleHbOIndex()
    hbo = np.array([100.0] * 4)  # way above max
    hbr = np.zeros(4)
    assert idx.compute(hbo, hbr) == 1.0


def test_simple_hbo_clips_at_zero():
    """HbO가 음수여도 0.0을 반환해야 한다."""
    idx = SimpleHbOIndex()
    hbo = np.array([-5.0] * 4)
    hbr = np.zeros(4)
    assert idx.compute(hbo, hbr) == 0.0
