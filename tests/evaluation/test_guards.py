import numpy as np
import pytest

from src.datasets.contract import LeakageError
from src.evaluation.guards import (
    check_normalization_source,
    check_session_overlap,
    check_subject_overlap,
    check_window_overlap,
)


def test_subject_overlap_passes_when_disjoint():
    check_subject_overlap(np.array(["sub-01", "sub-02"]), np.array(["sub-03"]))


def test_subject_overlap_raises_when_shared():
    with pytest.raises(LeakageError, match="sub-02"):
        check_subject_overlap(np.array(["sub-01", "sub-02"]), np.array(["sub-02"]))


def test_window_overlap_passes_for_different_subjects_at_same_time():
    # 다른 피험자의 같은 시각 창은 누수가 아니다
    check_window_overlap(
        np.array(["sub-01"]), np.array([0]), np.array([[0.0, 5.0]]),
        np.array(["sub-02"]), np.array([0]), np.array([[0.0, 5.0]]),
    )


def test_window_overlap_passes_for_same_subject_disjoint_times():
    check_window_overlap(
        np.array(["sub-01"]), np.array([0]), np.array([[0.0, 5.0]]),
        np.array(["sub-01"]), np.array([0]), np.array([[10.0, 15.0]]),
    )


def test_window_overlap_raises_for_same_subject_overlapping_times():
    with pytest.raises(LeakageError, match="overlap"):
        check_window_overlap(
            np.array(["sub-01"]), np.array([0]), np.array([[0.0, 5.0]]),
            np.array(["sub-01"]), np.array([0]), np.array([[4.0, 9.0]]),
        )


def test_window_overlap_treats_touching_edges_as_disjoint():
    # [0,5)와 [5,10)은 겹치지 않는다
    check_window_overlap(
        np.array(["sub-01"]), np.array([0]), np.array([[0.0, 5.0]]),
        np.array(["sub-01"]), np.array([0]), np.array([[5.0, 10.0]]),
    )


def test_session_overlap_fires_on_a_shared_subject_session_pair():
    with pytest.raises(LeakageError, match="sessions appear in both"):
        check_session_overlap(
            np.array(["a"]), np.array([0]),
            np.array(["a"]), np.array([0]),
        )


def test_session_overlap_ignores_the_same_index_under_different_subjects():
    """session_idx는 피험자를 가로질러 값이 겹친다. LOSO가 걸리면 안 된다."""
    check_session_overlap(
        np.array(["a", "a"]), np.array([0, 1]),
        np.array(["b", "b"]), np.array([0, 1]),
    )


def test_window_overlap_no_longer_false_positives_across_sessions():
    """window_times는 세션마다 0부터 다시 센다.

    피험자만으로 묶으면 물리적으로 다른 녹화의 같은 시각이 겹침으로
    판정되어 CrossSessionSplitter가 아예 돌지 못한다.
    """
    times = np.array([[10.0, 15.0]])
    check_window_overlap(
        np.array(["a"]), np.array([0]), times,
        np.array(["a"]), np.array([1]), times,
    )


def test_window_overlap_still_fires_inside_one_session():
    with pytest.raises(LeakageError, match="window overlap"):
        check_window_overlap(
            np.array(["a"]), np.array([0]), np.array([[10.0, 15.0]]),
            np.array(["a"]), np.array([0]), np.array([[12.0, 17.0]]),
        )


def test_normalization_source_guard_fires_when_a_masked_window_reaches_a_fold():
    with pytest.raises(LeakageError, match="normalization source"):
        check_normalization_source(
            np.array([True, False]), np.array([False, False])
        )


def test_normalization_source_guard_passes_when_no_masked_window_is_present():
    check_normalization_source(np.array([False, False]), np.array([False]))
