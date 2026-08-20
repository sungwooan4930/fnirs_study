import numpy as np
import pytest

from src.datasets.contract import SplitKeys
from src.evaluation.splitters import (
    CrossSessionSplitter,
    LosoSplitter,
    WithinSubjectSplitter,
    get_splitter,
)

SUBJ = np.array(["sub-01"] * 12 + ["sub-02"] * 12 + ["sub-03"] * 12)
TRIAL = np.tile(np.repeat(np.arange(3), 4), 3)
# 기존 테스트는 세션을 다루지 않으므로 단일 세션(전부 0)으로 채운다.
SESS = np.zeros(len(SUBJ), dtype=int)

# within_subject 전용 픽스처: 피험자당 9블록(수준당 3개) × 창 2개 = 18행.
# 파일럿(n_blocks_per_level=3, n_splits=3) 구조를 그대로 축소한 것이다.
WS_SUBJ = np.repeat(["sub-01", "sub-02", "sub-03"], 18)
WS_TRIAL = np.tile(np.repeat(np.arange(9), 2), 3)
WS_LEVELS = [0, 1, 1, 2, 0, 2, 1, 2, 0]
WS_LABEL = np.tile(np.repeat(WS_LEVELS, 2), 3)
WS_SESS = np.zeros(len(WS_SUBJ), dtype=int)

# GroupKFold(3)가 블록을 stride로 자르면 test fold가 {0,3,6}·{1,4,7}·{2,5,8}이
# 되는 배열. 수준이 [0,1,2] 반복이면 그 셋이 통째로 같은 수준이라 test는
# 단일 클래스, train에는 그 수준이 하나도 없어 정확도가 구조적으로 0이 된다.
DEGENERATE_LEVELS = [0, 1, 2, 0, 1, 2, 0, 1, 2]
DEGENERATE_LABEL = np.tile(np.repeat(DEGENERATE_LEVELS, 2), 3)


def _keys(subjects, sessions, trials, labels=None):
    return SplitKeys(
        subject_ids=np.array(subjects),
        session_ids=np.array(sessions),
        trial_ids=np.array(trials),
        labels=None if labels is None else np.array(labels),
    )


def test_loso_yields_one_fold_per_subject():
    folds = list(get_splitter("loso").split(_keys(SUBJ, SESS, TRIAL)))
    assert len(folds) == 3


def test_loso_test_fold_is_exactly_one_subject():
    for train_idx, test_idx in get_splitter("loso").split(_keys(SUBJ, SESS, TRIAL)):
        assert len(np.unique(SUBJ[test_idx])) == 1


def test_loso_train_and_test_subjects_are_disjoint():
    for train_idx, test_idx in get_splitter("loso").split(_keys(SUBJ, SESS, TRIAL)):
        assert not (set(SUBJ[train_idx]) & set(SUBJ[test_idx]))


def test_loso_covers_every_row_exactly_once_as_test():
    seen = np.zeros(len(SUBJ), dtype=int)
    for _, test_idx in get_splitter("loso").split(_keys(SUBJ, SESS, TRIAL)):
        seen[test_idx] += 1
    assert (seen == 1).all()


def test_within_subject_keeps_each_fold_inside_one_subject():
    for train_idx, test_idx in get_splitter("within_subject").split(
        _keys(WS_SUBJ, WS_SESS, WS_TRIAL, WS_LABEL)
    ):
        assert len(np.unique(WS_SUBJ[np.concatenate([train_idx, test_idx])])) == 1


def test_within_subject_splits_by_trial_not_by_window():
    for train_idx, test_idx in get_splitter("within_subject").split(
        _keys(WS_SUBJ, WS_SESS, WS_TRIAL, WS_LABEL)
    ):
        assert not (set(WS_TRIAL[train_idx]) & set(WS_TRIAL[test_idx]))


def test_within_subject_test_folds_carry_every_class():
    """각 fold의 test에 세 수준이 모두 들어가야 한다.

    들어가지 않으면 그 fold의 정확도는 판별력이 아니라 다수 클래스
    예측률이 되고, `accuracy_worst`(CLAUDE.md §5.4 필수 지표)가 구조적으로
    바닥에 고정된다.
    """
    all_classes = set(np.unique(WS_LABEL).tolist())
    folds = list(
        get_splitter("within_subject").split(_keys(WS_SUBJ, WS_SESS, WS_TRIAL, WS_LABEL))
    )
    assert len(folds) == 9  # 피험자 3명 × fold 3개
    for _, test_idx in folds:
        assert set(WS_LABEL[test_idx].tolist()) == all_classes


def test_within_subject_train_folds_carry_every_class():
    all_classes = set(np.unique(WS_LABEL).tolist())
    for train_idx, _ in get_splitter("within_subject").split(
        _keys(WS_SUBJ, WS_SESS, WS_TRIAL, WS_LABEL)
    ):
        assert set(WS_LABEL[train_idx].tolist()) == all_classes


def test_within_subject_survives_stride_degenerate_block_order():
    """GroupKFold라면 무너지는 블록 배열에서도 단일 클래스 fold가 없어야 한다.

    수준이 [0,1,2] 반복으로 놓이면 stride 분할은 test fold를 통째로 한
    수준으로 채운다. 층화 배정은 이 배열에서도 fold마다 수준당 1블록씩
    가져가므로 test에 세 수준이 모두 남는다.
    """
    all_classes = set(np.unique(DEGENERATE_LABEL).tolist())
    for train_idx, test_idx in get_splitter("within_subject").split(
        _keys(WS_SUBJ, WS_SESS, WS_TRIAL, DEGENERATE_LABEL)
    ):
        assert set(DEGENERATE_LABEL[test_idx].tolist()) == all_classes
        assert set(DEGENERATE_LABEL[train_idx].tolist()) == all_classes


def test_within_subject_covers_every_row_exactly_once_as_test():
    seen = np.zeros(len(WS_SUBJ), dtype=int)
    for _, test_idx in get_splitter("within_subject").split(
        _keys(WS_SUBJ, WS_SESS, WS_TRIAL, WS_LABEL)
    ):
        seen[test_idx] += 1
    assert (seen == 1).all()


def test_within_subject_without_labels_raises():
    """라벨 없이 나누면 층화가 불가능하므로 조용히 되돌아가지 않고 거부한다."""
    with pytest.raises(ValueError, match="needs the label array"):
        list(get_splitter("within_subject").split(_keys(WS_SUBJ, WS_SESS, WS_TRIAL)))


def test_within_subject_raises_when_a_class_has_a_single_block():
    """수준당 블록이 1개면 그 블록이 test인 fold의 train에 그 수준이 없다."""
    subj = np.repeat(["sub-01"], 6)
    sess = np.zeros(6, dtype=int)
    trial = np.repeat(np.arange(3), 2)
    label = np.repeat([0, 0, 1], 2)   # 수준 1은 블록 1개뿐
    with pytest.raises(ValueError, match="at least 2 blocks per class"):
        list(get_splitter("within_subject").split(_keys(subj, sess, trial, label)))


def test_window_random_mixes_subjects_across_train_and_test():
    leaked = False
    for train_idx, test_idx in get_splitter("window_random", seed=0).split(
        _keys(SUBJ, SESS, TRIAL)
    ):
        if set(SUBJ[train_idx]) & set(SUBJ[test_idx]):
            leaked = True
    assert leaked, "window_random은 T3 시연을 위해 반드시 피험자를 섞어야 한다"


def test_unknown_splitter_name_raises():
    with pytest.raises(ValueError, match="unknown splitter"):
        get_splitter("bogus")


def test_loso_allows_same_subject_is_false():
    assert get_splitter("loso").allows_same_subject is False


def test_within_subject_allows_same_subject_is_true():
    assert get_splitter("within_subject").allows_same_subject is True


def test_window_random_allows_same_subject_is_false():
    assert get_splitter("window_random").allows_same_subject is False


def test_within_subject_raises_on_subject_with_single_trial():
    subj = np.array(["sub-01"] * 4 + ["sub-02"] * 4)
    sess = np.zeros(8, dtype=int)
    trial = np.array([0, 0, 0, 0] + [0, 1, 0, 1])
    label = np.array([0, 1, 0, 1] + [0, 1, 0, 1])
    with pytest.raises(ValueError, match="within-subject cross-validation needs at least 2 trials"):
        list(get_splitter("within_subject").split(_keys(subj, sess, trial, label)))


def test_cross_session_holds_out_one_session_index_per_fold():
    keys = _keys(
        subjects=["a"] * 4 + ["b"] * 4,
        sessions=[0, 0, 1, 1] * 2,
        trials=[0, 1, 2, 3, 4, 5, 6, 7],
    )
    folds = list(CrossSessionSplitter().split(keys))
    assert len(folds) == 2
    for train_idx, test_idx in folds:
        test_sessions = set(keys.session_ids[test_idx].tolist())
        train_sessions = set(keys.session_ids[train_idx].tolist())
        assert len(test_sessions) == 1
        assert not (test_sessions & train_sessions)


def test_cross_session_keeps_the_same_subjects_on_both_sides():
    """세션 간 비교는 같은 사람의 다른 날을 보는 것이다."""
    keys = _keys(["a"] * 4, [0, 0, 1, 1], [0, 1, 2, 3])
    train_idx, test_idx = next(iter(CrossSessionSplitter().split(keys)))
    assert set(keys.subject_ids[train_idx]) == set(keys.subject_ids[test_idx])


def test_cross_session_declares_its_allowances():
    assert CrossSessionSplitter.allows_same_subject is True
    assert CrossSessionSplitter.allows_same_session is False


def test_cross_session_needs_at_least_two_sessions():
    keys = _keys(["a", "a"], [0, 0], [0, 1])
    with pytest.raises(ValueError, match="needs at least 2 sessions"):
        list(CrossSessionSplitter().split(keys))


def test_within_subject_allows_the_same_session_on_both_sides():
    """세션 안에서 시행을 나누므로 같은 세션이 양쪽에 나타난다."""
    assert WithinSubjectSplitter.allows_same_session is True


def test_within_subject_never_crosses_a_session_boundary():
    keys = _keys(
        subjects=["a"] * 8,
        sessions=[0, 0, 0, 0, 1, 1, 1, 1],
        trials=[0, 1, 2, 3, 4, 5, 6, 7],
        labels=[0, 1, 0, 1, 0, 1, 0, 1],
    )
    for train_idx, test_idx in WithinSubjectSplitter(n_splits=2).split(keys):
        train_pairs = {
            (s, int(k)) for s, k in zip(keys.subject_ids[train_idx], keys.session_ids[train_idx])
        }
        test_pairs = {
            (s, int(k)) for s, k in zip(keys.subject_ids[test_idx], keys.session_ids[test_idx])
        }
        # 같은 세션이 양쪽에 있는 것은 정상이다. 다른 세션이 섞이는 것이 문제다.
        assert len(test_pairs) == 1
        assert test_pairs <= train_pairs


def test_loso_keeps_subjects_apart_even_with_shared_session_indices():
    """session_idx는 피험자를 가로질러 값이 겹친다. LOSO가 걸리면 안 된다."""
    keys = _keys(["a", "a", "b", "b"], [0, 1, 0, 1], [0, 1, 2, 3])
    for train_idx, test_idx in LosoSplitter().split(keys):
        assert not (
            set(keys.subject_ids[train_idx]) & set(keys.subject_ids[test_idx])
        )


def test_get_splitter_knows_cross_session():
    assert isinstance(get_splitter("cross_session"), CrossSessionSplitter)
