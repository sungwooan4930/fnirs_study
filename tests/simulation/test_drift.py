import numpy as np
import pytest

from src.simulation.components.drift import apply_channel_drift, apply_noise_scaling


def test_multiplicative_gain_is_applied_per_channel():
    x = np.ones((3, 100))
    out = apply_channel_drift(
        x, gain=np.array([1.0, 2.0, 0.5]), offset=np.zeros(3), within_rate=0.0
    )
    assert np.allclose(out[0], 1.0)
    assert np.allclose(out[1], 2.0)
    assert np.allclose(out[2], 0.5)


def test_additive_offset_is_applied_per_channel():
    x = np.zeros((2, 50))
    out = apply_channel_drift(
        x, gain=np.ones(2), offset=np.array([1.5, -0.5]), within_rate=0.0
    )
    assert np.allclose(out[0], 1.5)
    assert np.allclose(out[1], -0.5)


def test_within_session_drift_ramps_the_gain_over_time():
    x = np.ones((1, 101))
    out = apply_channel_drift(
        x, gain=np.ones(1), offset=np.zeros(1), within_rate=0.5
    )
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, -1] == pytest.approx(1.5)
    assert out[0, 50] == pytest.approx(1.25)


def test_zero_within_rate_leaves_the_signal_flat_in_time():
    x = np.ones((1, 101))
    out = apply_channel_drift(
        x, gain=np.full(1, 2.0), offset=np.zeros(1), within_rate=0.0
    )
    assert np.allclose(out, 2.0)


def test_offset_is_not_ramped():
    """오프셋은 DC이므로 세션 내 이동의 대상이 아니다 — 이득만 이동한다."""
    x = np.zeros((1, 101))
    out = apply_channel_drift(
        x, gain=np.ones(1), offset=np.full(1, 3.0), within_rate=0.5
    )
    assert np.allclose(out, 3.0)


def test_shape_is_preserved():
    x = np.random.default_rng(0).normal(size=(4, 77))
    out = apply_channel_drift(
        x, gain=np.ones(4), offset=np.zeros(4), within_rate=0.2
    )
    assert out.shape == x.shape


def test_gain_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="gain length"):
        apply_channel_drift(
            np.ones((3, 10)), gain=np.ones(2), offset=np.zeros(3), within_rate=0.0
        )


def test_noise_scaling_raises_per_channel_variance():
    x = np.zeros((2, 4000))
    out = apply_noise_scaling(
        x, noise_scale=np.array([0.1, 1.0]), rng=np.random.default_rng(0)
    )
    assert out[0].std() == pytest.approx(0.1, rel=0.1)
    assert out[1].std() == pytest.approx(1.0, rel=0.1)


def test_noise_scaling_does_not_shift_the_mean():
    x = np.zeros((1, 8000))
    out = apply_noise_scaling(
        x, noise_scale=np.array([1.0]), rng=np.random.default_rng(0)
    )
    assert abs(out.mean()) < 0.05
