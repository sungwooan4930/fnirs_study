import numpy as np
import pytest

from src.common.seeding import set_all_seeds
from src.datasets.windowing import make_windows
from src.simulation.state import build_timeline

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "baseline_duration_s": 20,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}


def _timeline():
    return build_timeline(TASK_CFG, set_all_seeds(0))


def test_windows_have_requested_length():
    w = make_windows(_timeline(), window_s=5.0, step_s=1.0)
    assert np.allclose(w.end_s - w.start_s, 5.0)


def test_windows_advance_by_step():
    w = make_windows(_timeline(), window_s=5.0, step_s=1.0)
    within_first_block = w.start_s[w.trial_id == w.trial_id[0]]
    assert np.allclose(np.diff(within_first_block), 1.0)


def test_no_window_crosses_a_block_boundary():
    tl = _timeline()
    w = make_windows(tl, window_s=5.0, step_s=1.0)
    # 창 시작과 끝 직전이 같은 블록이어야 한다
    start_trial = tl.trial_at(w.start_s)
    end_trial = tl.trial_at(w.end_s - 1e-6)
    assert np.array_equal(start_trial, end_trial)


def test_label_constant_across_window():
    """Load must be constant across the entire window for the label to be unambiguous."""
    tl = _timeline()
    w = make_windows(tl, window_s=5.0, step_s=1.0)
    # Check that load at start == load at end (within window) == recorded label
    load_at_start = tl.load_at(w.start_s)
    load_at_end = tl.load_at(w.end_s - 1e-6)
    assert np.array_equal(load_at_start, load_at_end), "Load must be constant within window"
    assert np.array_equal(w.load_level, load_at_start), "Label must match load at start"


def test_count_matches_expected():
    # 과제 블록 30초, 창 5초, 스텝 1초 -> 블록당 26개, 과제 블록 6개.
    # 베이스라인 블록(20초, 시작·종료 2개)도 make_windows에는 그냥 하나의
    # 블록이므로 블록당 16개씩 더 생긴다. kind로 베이스라인 창을 걸러내는
    # 것은 Task 7(kind_at 도입) 몫이다 — 여기서는 현재의 정직한 개수를 확인한다.
    w = make_windows(_timeline(), window_s=5.0, step_s=1.0)
    assert len(w.start_s) == 6 * 26 + 2 * 16


def test_all_windows_inside_recording():
    tl = _timeline()
    w = make_windows(tl, window_s=5.0, step_s=1.0)
    assert w.start_s.min() >= 0.0
    assert w.end_s.max() <= tl.duration_s


def test_rejects_step_larger_than_window():
    with pytest.raises(ValueError, match="step_s"):
        make_windows(_timeline(), window_s=5.0, step_s=6.0)


def test_rejects_window_longer_than_block():
    with pytest.raises(ValueError, match="no windows"):
        make_windows(_timeline(), window_s=100.0, step_s=1.0)


def test_discards_boundary_crossing_windows():
    """Windows that straddle block boundaries must be explicitly discarded."""
    tl = _timeline()
    # The first block is now the *start baseline* (20 seconds, CLAUDE.md §2.4),
    # not a task block. For a 20-second block and 5-second windows, starts at
    # 16-19 cross into the next block (end at 21-24, outside [0, 20)).
    w = make_windows(tl, window_s=5.0, step_s=1.0)

    block_boundary_starts = np.array([16.0, 17.0, 18.0, 19.0])
    safe_start = 10.0  # Within block, ends at 15.0

    for start in block_boundary_starts:
        assert not np.any(np.isclose(w.start_s, start)), \
            f"Window starting at {start} crosses block boundary and should be discarded"

    assert np.any(np.isclose(w.start_s, safe_start)), \
        f"Safe window starting at {safe_start} should be retained"


from src.datasets.windowing import make_windows
from src.simulation.state import BASELINE, TASK, build_timeline

_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
    "baseline_duration_s": 20,
    "n_blocks_per_level": 2,
    "stim_interval_s": 2.0,
}


def test_windows_carry_block_kind():
    tl = build_timeline(_CFG, np.random.default_rng(0))
    w = make_windows(tl, window_s=5.0, step_s=1.0)
    assert len(w.block_kind) == len(w.start_s)
    assert set(np.unique(w.block_kind)) == {BASELINE, TASK}


def test_baseline_windows_lie_entirely_inside_baseline_blocks():
    tl = build_timeline(_CFG, np.random.default_rng(0))
    w = make_windows(tl, window_s=5.0, step_s=1.0)
    base = w.block_kind == BASELINE
    assert base.any()
    assert (tl.kind_at(w.start_s[base]) == BASELINE).all()
    assert (tl.kind_at(w.end_s[base] - 1e-6) == BASELINE).all()


def test_both_baseline_blocks_produce_windows():
    """시작·종료 베이스라인 둘 다 창을 내야 드리프트를 잴 수 있다."""
    tl = build_timeline(_CFG, np.random.default_rng(0))
    w = make_windows(tl, window_s=5.0, step_s=1.0)
    base_trials = np.unique(w.trial_id[w.block_kind == BASELINE])
    assert len(base_trials) == 2


def test_baseline_shorter_than_window_is_rejected():
    cfg = dict(_CFG, baseline_duration_s=3)
    tl = build_timeline(cfg, np.random.default_rng(0))
    with pytest.raises(ValueError, match="baseline block"):
        make_windows(tl, window_s=5.0, step_s=1.0)
