import numpy as np
import pytest

from src.common.seeding import set_all_seeds


def test_same_seed_gives_same_numbers():
    g1 = set_all_seeds(42)
    g2 = set_all_seeds(42)
    assert np.array_equal(g1.normal(size=10), g2.normal(size=10))


def test_different_seed_gives_different_numbers():
    g1 = set_all_seeds(42)
    g2 = set_all_seeds(43)
    assert not np.array_equal(g1.normal(size=10), g2.normal(size=10))


def test_rejects_non_integer_seed():
    with pytest.raises(TypeError):
        set_all_seeds("42")


def test_rejects_negative_seed():
    with pytest.raises(ValueError):
        set_all_seeds(-1)
