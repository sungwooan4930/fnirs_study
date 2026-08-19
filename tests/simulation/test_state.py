import numpy as np
import pytest

from src.common.seeding import set_all_seeds
from src.simulation.state import (
    BASELINE,
    BASELINE_LOAD_SENTINEL,
    STATE_SFREQ,
    TASK,
    build_timeline,
)

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "baseline_duration_s": 20,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}


def test_duration_is_blocks_times_duration():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    # 세션은 베이스라인(양끝) + 과제 블록(6개)이다. 2 * baseline_duration_s가
    # 과제 시간(3 levels * 2 blocks * 30s)에 더해진다.
    assert tl.duration_s == pytest.approx(2 * 20 + 3 * 2 * 30)


def test_samples_at_state_sfreq():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    assert len(tl.t) == int(tl.duration_s * STATE_SFREQ)
    assert tl.t[1] - tl.t[0] == pytest.approx(1.0 / STATE_SFREQ)


def test_all_load_levels_appear_equally_often():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    # 베이스라인 구간(-1 센티넬)은 과제 조건이 아니므로 제외하고 셈한다.
    task_loads = tl.load_level[tl.block_kind == TASK]
    counts = np.bincount(task_loads, minlength=3)
    assert counts[0] == counts[1] == counts[2]


def test_block_order_distribution_is_balanced():
    # 카운터밸런스: 첫 "과제" 블록에서 레벨 분포가 균형잡혀야 한다.
    # 인덱스 0은 이제 시작 베이스라인이므로 첫 과제 블록의 레벨을 봐야 한다.
    # 30개 시드에서 모든 레벨이 최소 1회 이상 나타나고,
    # 어느 레벨도 60% 이상 차지하지 않아야 한다.
    first_levels = []
    for seed in range(30):
        tl = build_timeline(TASK_CFG, set_all_seeds(seed))
        first_level = tl.load_level[tl.block_kind == TASK][0]
        first_levels.append(first_level)

    first_levels = np.array(first_levels)
    counts = np.bincount(first_levels, minlength=3)

    # 모든 레벨이 최소 1회 이상 나타나야 함
    assert np.all(counts > 0), f"Some levels never appeared first: {counts}"

    # 어느 레벨도 60% 이상을 차지하면 안 됨 (편향 방지)
    max_fraction = np.max(counts) / len(first_levels)
    assert max_fraction < 0.6, f"Level over-represented: {counts}, fraction={max_fraction}"


def test_fatigue_increases_monotonically():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    assert np.all(np.diff(tl.fatigue) >= 0)
    assert tl.fatigue[0] == pytest.approx(0.0)
    assert tl.fatigue[-1] < 1.0 + 1e-9


def test_trial_id_is_one_per_block():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    # 과제 블록 6개(3 levels * 2) + 시작·종료 베이스라인 블록 2개 = 8
    assert len(np.unique(tl.trial_id)) == 8


def test_stim_onsets_spaced_by_interval():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    gaps = np.diff(tl.stim_onsets)
    assert np.allclose(gaps, 2.0)
    assert tl.stim_onsets[-1] < tl.duration_s


def test_load_at_uses_step_interpolation():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    # 블록 중앙에서 조회한 값은 그 블록의 라벨과 같아야 한다.
    # 인덱스 0은 이제 시작 베이스라인(20초)이므로, 첫 "과제" 블록(20~50초)의
    # 중앙인 35초를 사용한다.
    first_task_level = tl.load_level[tl.block_kind == TASK][0]
    mid_of_first_task_block = 20.0 + 15.0
    assert tl.load_at(np.array([mid_of_first_task_block]))[0] == first_task_level


def test_load_at_clamps_beyond_end():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    assert tl.load_at(np.array([tl.duration_s + 100.0]))[0] == tl.load_level[-1]


def test_arrays_stay_same_length_for_non_integral_block_duration():
    # 배열 길이 불변: 블록 지속시간이 정수배가 아니어도 모든 배열이 같은 길이여야 한다.
    # 이는 duration_s가 샘플 그리드로부터 도출되어야 함을 보장한다.
    cfg = {
        "nback_levels": [0, 2, 3],
        "block_duration_s": 20.05,  # 비정수 → 샘플링 결과도 비정수
        "baseline_duration_s": 5.0,
        "n_blocks_per_level": 2,
        "stim_interval_s": 2.0,
    }
    tl = build_timeline(cfg, set_all_seeds(42))
    assert len(tl.t) == len(tl.load_level), \
        f"t and load_level length mismatch: {len(tl.t)} vs {len(tl.load_level)}"
    assert len(tl.t) == len(tl.block_kind), \
        f"t and block_kind length mismatch: {len(tl.t)} vs {len(tl.block_kind)}"
    assert len(tl.t) == len(tl.fatigue), \
        f"t and fatigue length mismatch: {len(tl.t)} vs {len(tl.fatigue)}"
    assert len(tl.t) == len(tl.trial_id), \
        f"t and trial_id length mismatch: {len(tl.t)} vs {len(tl.trial_id)}"


def test_session_starts_and_ends_with_a_baseline_block():
    tl = build_timeline(TASK_CFG, np.random.default_rng(0))
    assert tl.block_kind[0] == BASELINE
    assert tl.block_kind[-1] == BASELINE
    # 가운데 어딘가는 과제여야 한다
    assert (tl.block_kind == TASK).any()


def test_baseline_blocks_are_two_and_bookend_the_task_blocks():
    tl = build_timeline(TASK_CFG, np.random.default_rng(0))
    baseline_trials = np.unique(tl.trial_id[tl.block_kind == BASELINE])
    assert len(baseline_trials) == 2
    assert baseline_trials[0] == tl.trial_id.min()
    assert baseline_trials[1] == tl.trial_id.max()


def test_baseline_samples_carry_the_load_sentinel():
    tl = build_timeline(TASK_CFG, np.random.default_rng(0))
    assert (tl.load_level[tl.block_kind == BASELINE] == BASELINE_LOAD_SENTINEL).all()
    assert (tl.load_level[tl.block_kind == TASK] >= 0).all()


def test_no_stimulus_lands_inside_a_baseline_block():
    tl = build_timeline(TASK_CFG, np.random.default_rng(0))
    assert len(tl.stim_onsets) > 0
    assert (tl.kind_at(tl.stim_onsets) == TASK).all()


def test_baseline_duration_matches_config():
    tl = build_timeline(TASK_CFG, np.random.default_rng(0))
    n_baseline = int((tl.block_kind == BASELINE).sum())
    expected = 2 * int(round(TASK_CFG["baseline_duration_s"] * 10.0))
    assert n_baseline == expected


def test_effective_load_scales_by_practice_gain():
    tl_a = build_timeline(TASK_CFG, np.random.default_rng(0), practice_gain=1.0)
    tl_b = build_timeline(TASK_CFG, np.random.default_rng(0), practice_gain=0.5)
    t = tl_a.t
    assert np.allclose(tl_b.effective_load(t), 0.5 * tl_a.effective_load(t))


def test_effective_load_treats_baseline_as_rest_not_negative():
    tl = build_timeline(TASK_CFG, np.random.default_rng(0))
    base_t = tl.t[tl.block_kind == BASELINE]
    assert (tl.effective_load(base_t) == 0.0).all()


def test_session_idx_is_recorded():
    tl = build_timeline(TASK_CFG, np.random.default_rng(0), session_idx=2)
    assert tl.session_idx == 2
