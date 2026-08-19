import inspect

import numpy as np
import pytest

from src.preprocessing.baseline import (
    MODALITY_KIND,
    NORMALIZATION_KINDS,
    SessionBaseline,
    compute_drift,
    flag_drift,
)


def test_fit_signature_cannot_receive_another_session():
    """경계를 규율이 아니라 서명으로 막는다 (스펙 §6.1).

    다른 세션·다른 피험자를 넘길 인자가 존재하지 않아야 한다.
    """
    params = list(inspect.signature(SessionBaseline.fit).parameters)
    assert params == ["start_baseline", "kind"]


def test_concentration_delta_subtracts_the_reference():
    base = np.array([[1.0, 2.0], [3.0, 4.0]])          # 평균 (2.0, 3.0)
    sb = SessionBaseline.fit(base, "concentration_delta")
    out = sb.apply(np.array([[2.0, 3.0], [5.0, 3.0]]))
    assert np.allclose(out, [[0.0, 0.0], [3.0, 0.0]])


def test_band_power_db_is_zero_at_the_reference_and_3db_when_doubled():
    base = np.array([[4.0, 9.0]])
    sb = SessionBaseline.fit(base, "band_power_db")
    assert np.allclose(sb.apply(np.array([[4.0, 9.0]])), 0.0)
    assert np.allclose(sb.apply(np.array([[8.0, 18.0]])), 10 * np.log10(2.0))


def test_absolute_leaves_the_values_alone():
    base = np.array([[0.9, 0.5]])
    sb = SessionBaseline.fit(base, "absolute")
    x = np.array([[0.7, 0.4], [0.95, 0.6]])
    assert np.allclose(sb.apply(x), x)


def test_absolute_returns_a_copy_not_the_same_object():
    sb = SessionBaseline.fit(np.array([[1.0]]), "absolute")
    x = np.array([[2.0]])
    out = sb.apply(x)
    out[0, 0] = 99.0
    assert x[0, 0] == 2.0


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown normalization kind"):
        SessionBaseline.fit(np.array([[1.0]]), "zscore")


def test_empty_baseline_is_rejected():
    with pytest.raises(ValueError, match="baseline block produced no windows"):
        SessionBaseline.fit(np.empty((0, 3)), "concentration_delta")


def test_modality_kind_covers_the_three_modalities():
    assert set(MODALITY_KIND) == {"eeg", "fnirs", "behavior"}
    assert set(MODALITY_KIND.values()) <= set(NORMALIZATION_KINDS)


def test_drift_is_a_relative_ratio():
    start = np.array([[10.0, 100.0]])
    end = np.array([[11.0, 100.0]])
    d = compute_drift(start, end, zero_atol=1e-8)
    assert np.allclose(d.per_channel, [0.1, 0.0])
    assert d.aggregate == pytest.approx(0.05)
    assert d.n_excluded == 0


def test_near_zero_baseline_channels_are_excluded_and_counted():
    start = np.array([[10.0, 0.0]])
    end = np.array([[11.0, 5.0]])
    d = compute_drift(start, end, zero_atol=1e-8)
    assert d.n_excluded == 1
    assert np.isnan(d.per_channel[1])
    assert d.aggregate == pytest.approx(0.1)


def test_all_channels_excluded_yields_nan_not_a_quiet_zero():
    start = np.zeros((1, 2))
    end = np.ones((1, 2))
    d = compute_drift(start, end, zero_atol=1e-8)
    assert d.n_excluded == 2
    assert np.isnan(d.aggregate)


def test_nan_drift_counts_as_flagged():
    """평가 불가는 '문제 없음'이 아니다. 조용히 통과시키면 안 된다."""
    assert flag_drift({"fnirs": float("nan")}, threshold=0.2) is True


def test_flag_fires_only_above_threshold():
    assert flag_drift({"fnirs": 0.1, "eeg": 0.05}, threshold=0.2) is False
    assert flag_drift({"fnirs": 0.1, "eeg": 0.31}, threshold=0.2) is True
