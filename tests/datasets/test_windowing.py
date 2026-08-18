import numpy as np
import pytest

from src.common.seeding import set_all_seeds
from src.datasets.windowing import make_windows
from src.simulation.state import build_timeline

TASK_CFG = {
    "nback_levels": [0, 2, 3],
    "block_duration_s": 30,
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
    # 블록 30초, 창 5초, 스텝 1초 -> 블록당 26개, 블록 6개
    w = make_windows(_timeline(), window_s=5.0, step_s=1.0)
    assert len(w.start_s) == 6 * 26


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
    # For 30-second blocks and 5-second windows, starts at 26-29 within
    # each block cross into the next (ends at 31-34).
    w = make_windows(tl, window_s=5.0, step_s=1.0)

    # The first block spans [0, 30). Windows starting at 26, 27, 28, 29
    # would end at 31, 32, 33, 34 (outside the block). They should be absent.
    block_boundary_starts = np.array([26.0, 27.0, 28.0, 29.0])
    safe_start = 20.0  # Within block, ends at 25.0

    for start in block_boundary_starts:
        assert not np.any(np.isclose(w.start_s, start)), \
            f"Window starting at {start} crosses block boundary and should be discarded"

    assert np.any(np.isclose(w.start_s, safe_start)), \
        f"Safe window starting at {safe_start} should be retained"
