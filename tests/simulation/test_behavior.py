import numpy as np
import pytest

from src.common.seeding import set_all_seeds
from src.simulation.components.behavior import generate_behavior
from src.simulation.state import build_timeline
from src.simulation.subject import make_subjects

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "baseline_duration_s": 20,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}


def _make(effect_size, lead_delta_s=1.2, seed=0):
    rng = set_all_seeds(seed)
    tl = build_timeline(TASK_CFG, rng)
    sub = make_subjects(1, 0.0, rng)[0]
    return tl, generate_behavior(tl, sub, rng, effect_size=effect_size, lead_delta_s=lead_delta_s)


def test_one_entry_per_stimulus():
    tl, log = _make(0.8)
    assert len(log.onsets) == len(tl.stim_onsets)
    assert len(log.correct) == len(log.rt) == len(log.onsets)


def test_correct_is_binary():
    _, log = _make(0.8)
    assert set(np.unique(log.correct)).issubset({0, 1})


def test_rt_is_positive():
    _, log = _make(0.8)
    assert (log.rt > 0).all()


def test_accuracy_drops_with_load_measured_at_lead_time():
    tl, log = _make(0.8, lead_delta_s=1.2)
    driving_load = tl.load_at(log.onsets - 1.2)
    acc_low = log.correct[driving_load == 0].mean()
    acc_high = log.correct[driving_load == 2].mean()
    assert acc_high < acc_low


def test_rt_rises_with_load_measured_at_lead_time():
    tl, log = _make(0.8, lead_delta_s=1.2)
    driving_load = tl.load_at(log.onsets - 1.2)
    assert log.rt[driving_load == 2].mean() > log.rt[driving_load == 0].mean()


def test_lead_delta_shifts_which_state_drives_behavior():
    # Δ가 크게 다르면 같은 시드에서도 행동 계열이 달라져야 한다
    _, a = _make(0.8, lead_delta_s=0.0)
    _, b = _make(0.8, lead_delta_s=10.0)
    assert not np.array_equal(a.rt, b.rt)


def test_zero_effect_size_removes_load_dependence():
    tl, log = _make(0.0)
    driving_load = tl.load_at(log.onsets - 1.2)
    lo = log.rt[driving_load == 0].mean()
    hi = log.rt[driving_load == 2].mean()
    assert hi == pytest.approx(lo, abs=0.05)


def test_rejects_negative_lead_delta():
    with pytest.raises(ValueError):
        _make(0.8, lead_delta_s=-1.0)


class _LoadAtSpy:
    """generate_behavior가 load_at에 실제로 넘기는 시각을 기록한다."""

    def __init__(self, timeline):
        self._tl = timeline
        self.queries = []

    def __getattr__(self, name):
        return getattr(self._tl, name)

    def load_at(self, t):
        self.queries.append(np.asarray(t).copy())
        return self._tl.load_at(t)


def test_load_at_called_with_onset_minus_lead_delta():
    """generate_behavior는 정확히 onset - lead_delta_s를 load_at에 전달해야 한다."""
    rng = set_all_seeds(0)
    tl = build_timeline(TASK_CFG, rng)
    sub = make_subjects(1, 0.0, rng)[0]

    spy = _LoadAtSpy(tl)
    lead_delta_s = 1.2
    generate_behavior(spy, sub, rng, effect_size=0.8, lead_delta_s=lead_delta_s)

    # 정확히 한 번 호출되어야 함
    assert len(spy.queries) == 1
    query = spy.queries[0]

    # 정답: onsets - lead_delta_s
    expected = tl.stim_onsets - lead_delta_s
    assert np.allclose(query, expected), \
        f"Query should be onsets - {lead_delta_s}, got array differing from expected"

    # 오답: onsets + lead_delta_s (부호 역전을 감지)
    wrong = tl.stim_onsets + lead_delta_s
    assert not np.allclose(query, wrong), \
        "Query matches onsets + lead_delta_s; sign is inverted"


def test_accuracy_drops_with_lagged_load_short_blocks():
    """짧은 블록 사용으로 onset - Δ와 onset + Δ이 다른 블록에 떨어지도록.
    이 조건에서 부호 역전이 정확도·RT에 눈에 띄는 영향을 준다."""
    short_block_cfg = {
        "nback_levels": [0, 2, 3],
        "block_duration_s": 3.0,      # 짧은 블록
        "baseline_duration_s": 5.0,
        "n_blocks_per_level": 4,      # 더 많은 블록 → 더 많은 경계 전환
        "stim_interval_s": 0.5,       # 자극 더 촘촘히
    }
    rng = set_all_seeds(42)
    tl = build_timeline(short_block_cfg, rng)
    sub = make_subjects(1, 0.0, rng)[0]

    lead_delta_s = 1.2
    log = generate_behavior(tl, sub, rng, effect_size=0.8, lead_delta_s=lead_delta_s)

    # 정확히 onset - 1.2의 상태를 기준으로 정확도와 RT를 검증
    driving_load = tl.load_at(log.onsets - lead_delta_s)

    # 높은 부하일 때 정확도 저하
    acc_low = log.correct[driving_load == 0].mean()
    acc_high = log.correct[driving_load == 2].mean()
    # 통계 잡음이 있을 수 있지만, 올바른 부호면 일관된 경향
    assert acc_high < acc_low, \
        f"With short blocks and correct sign, accuracy should drop at high load. " \
        f"Got acc_low={acc_low:.3f}, acc_high={acc_high:.3f}"
