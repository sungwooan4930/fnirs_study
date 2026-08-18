import numpy as np
import pytest

from src.datasets.contract import LeakageError
from src.evaluation.guards import check_subject_overlap, check_window_overlap


def test_subject_overlap_passes_when_disjoint():
    check_subject_overlap(np.array(["sub-01", "sub-02"]), np.array(["sub-03"]))


def test_subject_overlap_raises_when_shared():
    with pytest.raises(LeakageError, match="sub-02"):
        check_subject_overlap(np.array(["sub-01", "sub-02"]), np.array(["sub-02"]))


def test_window_overlap_passes_for_different_subjects_at_same_time():
    # 다른 피험자의 같은 시각 창은 누수가 아니다
    check_window_overlap(
        np.array(["sub-01"]), np.array([[0.0, 5.0]]),
        np.array(["sub-02"]), np.array([[0.0, 5.0]]),
    )


def test_window_overlap_passes_for_same_subject_disjoint_times():
    check_window_overlap(
        np.array(["sub-01"]), np.array([[0.0, 5.0]]),
        np.array(["sub-01"]), np.array([[10.0, 15.0]]),
    )


def test_window_overlap_raises_for_same_subject_overlapping_times():
    with pytest.raises(LeakageError, match="overlap"):
        check_window_overlap(
            np.array(["sub-01"]), np.array([[0.0, 5.0]]),
            np.array(["sub-01"]), np.array([[4.0, 9.0]]),
        )


def test_window_overlap_treats_touching_edges_as_disjoint():
    # [0,5)와 [5,10)은 겹치지 않는다
    check_window_overlap(
        np.array(["sub-01"]), np.array([[0.0, 5.0]]),
        np.array(["sub-01"]), np.array([[5.0, 10.0]]),
    )
