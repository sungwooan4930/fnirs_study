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


def _session_keys(subjects: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    """(피험자, 세션)을 하나의 문자열 키로 만든다.

    session_idx만으로 비교하면 피험자 A의 세션 0과 피험자 B의 세션 0이 같은
    세션으로 취급되어 LOSO가 자기 가드에 걸린다. 반대로 피험자만으로 묶으면
    세션마다 0부터 다시 세는 window_times가 겹침으로 오판된다.
    """
    return np.array([f"{s}/ses-{int(k)}" for s, k in zip(subjects, sessions)])


def check_session_overlap(
    train_subjects: np.ndarray,
    train_sessions: np.ndarray,
    test_subjects: np.ndarray,
    test_sessions: np.ndarray,
) -> None:
    """같은 (피험자, 세션)이 train과 test에 동시에 있으면 실패시킨다."""
    train_keys = set(_session_keys(train_subjects, train_sessions).tolist())
    test_keys = set(_session_keys(test_subjects, test_sessions).tolist())
    shared = sorted(train_keys & test_keys)
    if shared:
        raise LeakageError(
            f"sessions appear in both train and test: {shared}. "
            "세션 간 일반화를 측정한다고 주장하는 분할에서 세션이 공유되면 "
            "그 주장 자체가 무너진다 (CLAUDE.md §5.4)."
        )


def check_window_overlap(
    train_subjects: np.ndarray,
    train_sessions: np.ndarray,
    train_times: np.ndarray,
    test_subjects: np.ndarray,
    test_sessions: np.ndarray,
    test_times: np.ndarray,
) -> None:
    """같은 (피험자, 세션) 안에서 train 창과 test 창이 시간상 겹치면 실패시킨다.

    서로 다른 피험자, 또는 같은 피험자라도 서로 다른 세션의 창은 물리적으로
    다른 녹화이므로 시각이 같아도 검사 대상이 아니다. window_times가 세션마다
    0부터 다시 세기 때문에, 피험자만으로 묶으면 세션 간 분할이 전부 오탐이 된다.

    5초 창·1초 스텝은 80% 오버랩이므로 창 단위 무작위 분할은 여기서 걸린다.
    """
    train_keys = _session_keys(train_subjects, train_sessions)
    test_keys = _session_keys(test_subjects, test_sessions)

    for key in set(train_keys.tolist()) & set(test_keys.tolist()):
        tr = train_times[train_keys == key]
        te = test_times[test_keys == key]
        # 반열린 구간 [start, end) 기준 겹침
        overlaps = (tr[:, None, 0] < te[None, :, 1]) & (te[None, :, 0] < tr[:, None, 1])
        if overlaps.any():
            i, j = np.argwhere(overlaps)[0]
            raise LeakageError(
                f"window overlap for {key}: train {tr[i].tolist()} "
                f"overlaps test {te[j].tolist()}. "
                "인접 윈도우가 train/test에 동시 존재하면 그 결과는 폐기 대상이다."
            )


def check_normalization_source(
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    """세션 베이스라인 fit에 쓰인 창이 fold에 들어오면 실패시킨다.

    "베이스라인은 train도 test도 아니다"는 설계상의 사실이 아니라 여기서
    지키는 불변식이다. 미래에 누가 베이스라인을 타깃에 옵트인하면 이 가드가
    먼저 터지고, 그 사람은 "이 타깃에는 다른 정규화 기준을 쓰겠다"를 명시적으로
    결정하게 된다 — 조용히 통과하지 않는다.
    """
    n_train = int(np.asarray(train_mask).sum())
    n_test = int(np.asarray(test_mask).sum())
    if n_train or n_test:
        raise LeakageError(
            f"{n_train + n_test} window(s) used as the normalization source "
            f"reached a fold (train={n_train}, test={n_test}). "
            "정규화 기준 구간이 분석 데이터에 들어가면 그 표본은 정규화 정의상 "
            "0 근처가 되어 공짜 클래스가 생긴다 (스펙 §6.4·§6.5)."
        )
