import numpy as np
import pytest

from src.common.seeding import set_all_seeds
from src.simulation.state import STATE_SFREQ, build_timeline

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}


def test_duration_is_blocks_times_duration():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    assert tl.duration_s == pytest.approx(3 * 2 * 30)


def test_samples_at_state_sfreq():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    assert len(tl.t) == int(tl.duration_s * STATE_SFREQ)
    assert tl.t[1] - tl.t[0] == pytest.approx(1.0 / STATE_SFREQ)


def test_all_load_levels_appear_equally_often():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    counts = np.bincount(tl.load_level, minlength=3)
    assert counts[0] == counts[1] == counts[2]


def test_block_order_is_shuffled_not_sorted():
    # 카운터밸런스: 블록 순서가 0,0,1,1,2,2 처럼 정렬돼 있으면 안 된다
    orders = set()
    for seed in range(10):
        tl = build_timeline(TASK_CFG, set_all_seeds(seed))
        first_of_each_block = tl.load_level[:: int(30 * STATE_SFREQ)]
        orders.add(tuple(first_of_each_block))
    assert len(orders) > 1


def test_fatigue_increases_monotonically():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    assert np.all(np.diff(tl.fatigue) >= 0)
    assert tl.fatigue[0] == pytest.approx(0.0)
    assert tl.fatigue[-1] < 1.0 + 1e-9


def test_trial_id_is_one_per_block():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    assert len(np.unique(tl.trial_id)) == 6


def test_stim_onsets_spaced_by_interval():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    gaps = np.diff(tl.stim_onsets)
    assert np.allclose(gaps, 2.0)
    assert tl.stim_onsets[-1] < tl.duration_s


def test_load_at_uses_step_interpolation():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    # 블록 중앙에서 조회한 값은 그 블록의 라벨과 같아야 한다
    mid_of_first_block = 15.0
    assert tl.load_at(np.array([mid_of_first_block]))[0] == tl.load_level[0]


def test_load_at_clamps_beyond_end():
    tl = build_timeline(TASK_CFG, set_all_seeds(0))
    assert tl.load_at(np.array([tl.duration_s + 100.0]))[0] == tl.load_level[-1]
