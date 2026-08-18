"""누수 가드.

계약(contract.py)이 전역 fit을 막는다면, 여기서는 분할 자체가
train/test 경계를 침범하지 않았는지 매 fold마다 확인한다.
"""

from __future__ import annotations

import numpy as np

from src.datasets.contract import LeakageError


def check_subject_overlap(
    train_subjects: np.ndarray,
    test_subjects: np.ndarray,
) -> None:
    """같은 피험자가 train과 test에 동시에 있으면 실패시킨다."""
    shared = sorted(set(train_subjects.tolist()) & set(test_subjects.tolist()))
    if shared:
        raise LeakageError(
            f"subjects appear in both train and test: {shared}. "
            "분할 단위는 항상 피험자여야 한다 (CLAUDE.md 5.1)."
        )


def check_window_overlap(
    train_subjects: np.ndarray,
    train_times: np.ndarray,
    test_subjects: np.ndarray,
    test_times: np.ndarray,
) -> None:
    """같은 피험자 안에서 train 창과 test 창이 시간상 겹치면 실패시킨다.

    서로 다른 피험자의 창은 시각이 같아도 다른 녹화이므로 검사 대상이 아니다.
    5초 창·1초 스텝은 80% 오버랩이므로 창 단위 무작위 분할은 여기서 걸린다.
    """
    for subject in set(train_subjects.tolist()) & set(test_subjects.tolist()):
        tr = train_times[train_subjects == subject]
        te = test_times[test_subjects == subject]
        # 반열린 구간 [start, end) 기준 겹침
        overlaps = (tr[:, None, 0] < te[None, :, 1]) & (te[None, :, 0] < tr[:, None, 1])
        if overlaps.any():
            i, j = np.argwhere(overlaps)[0]
            raise LeakageError(
                f"window overlap for {subject}: train {tr[i].tolist()} "
                f"overlaps test {te[j].tolist()}. "
                "인접 윈도우가 train/test에 동시 존재하면 그 결과는 폐기 대상이다."
            )
