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


def test_concentration_delta_subtracts_and_scales_by_baseline_spread():
    """2026-08-19 정정(Task 14/T3): 뺄셈만으로는 곱셈 이득(gain)이 남는다
    (docstring 참조). 이제 베이스라인 산포(ddof=1)로도 나눈다.
    """
    base = np.array([[1.0, 2.0], [3.0, 4.0]])          # 평균 (2.0, 3.0), std(ddof=1) = √2
    sb = SessionBaseline.fit(base, "concentration_delta")
    out = sb.apply(np.array([[2.0, 3.0], [5.0, 3.0]]))
    scale = np.sqrt(2.0)
    assert np.allclose(out, [[0.0, 0.0], [3.0 / scale, 0.0]])


def test_concentration_delta_cancels_multiplicative_gain():
    """이 정정의 존재 이유: y = gain·x + offset일 때, gain이 세션마다 달라도
    (양수인 한) 정규화 결과가 같아야 한다. 이게 안 되면 세션 간 분류기가
    진짜 인지상태가 아니라 잔류 gain 차이를 학습한다 (Task 14 T3 실측:
    회복률 0.268 → fnirs_gain_sigma=0일 때 0.378로 상승, 원인이 이것임을 확인).
    """
    rng = np.random.default_rng(0)
    x_base = rng.normal(size=(20, 3))
    x_task = rng.normal(size=(5, 3)) + 1.5  # 진짜 과제 반응

    def normalize_under(gain: float, offset: float) -> np.ndarray:
        y_base = gain * x_base + offset
        y_task = gain * x_task + offset
        sb = SessionBaseline.fit(y_base, "concentration_delta")
        return sb.apply(y_task)

    out_gain1 = normalize_under(1.0, 0.0)
    out_gain5 = normalize_under(5.0, -3.0)
    out_gain_small = normalize_under(0.2, 10.0)
    assert np.allclose(out_gain1, out_gain5, atol=1e-9)
    assert np.allclose(out_gain1, out_gain_small, atol=1e-9)


def test_concentration_delta_without_scale_falls_back_to_subtraction_only():
    """`fit`을 거치지 않고 `reference`만 직접 구성하는 기존 훅(예: 결함 주입
    테스트)과의 호환성. `scale=None`이면 옛 동작(뺄셈만)으로 되돌아간다."""
    sb = SessionBaseline(reference=np.array([2.0, 3.0]), kind="concentration_delta")
    out = sb.apply(np.array([[5.0, 3.0]]))
    assert np.allclose(out, [[3.0, 0.0]])


def test_near_constant_baseline_channel_does_not_divide_by_zero():
    """σ≈0인 채널(거의 상수)이 있어도 발산하지 않는다 — `_SCALE_FLOOR` 하한."""
    base = np.array([[5.0], [5.0], [5.0]])  # std(ddof=1) == 0
    sb = SessionBaseline.fit(base, "concentration_delta")
    out = sb.apply(np.array([[6.0]]))
    assert np.isfinite(out).all()


def test_single_baseline_window_does_not_produce_nan_scale():
    """베이스라인 창이 하나뿐이면 ddof=1 표준편차가 정의되지 않는다(0/0) —
    NaN이 아니라 `_SCALE_FLOOR`로 클램프돼야 한다."""
    base = np.array([[5.0, 2.0]])  # n=1
    sb = SessionBaseline.fit(base, "concentration_delta")
    out = sb.apply(np.array([[6.0, 2.0]]))
    assert np.isfinite(out).all()


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
