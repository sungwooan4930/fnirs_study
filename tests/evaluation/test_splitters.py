import numpy as np
import pytest

from src.evaluation.splitters import get_splitter

SUBJ = np.array(["sub-01"] * 12 + ["sub-02"] * 12 + ["sub-03"] * 12)
TRIAL = np.tile(np.repeat(np.arange(3), 4), 3)

# within_subject 전용 픽스처: 피험자당 9블록(수준당 3개) × 창 2개 = 18행.
# 파일럿(n_blocks_per_level=3, n_splits=3) 구조를 그대로 축소한 것이다.
WS_SUBJ = np.repeat(["sub-01", "sub-02", "sub-03"], 18)
WS_TRIAL = np.tile(np.repeat(np.arange(9), 2), 3)
WS_LEVELS = [0, 1, 1, 2, 0, 2, 1, 2, 0]
WS_LABEL = np.tile(np.repeat(WS_LEVELS, 2), 3)

# GroupKFold(3)가 블록을 stride로 자르면 test fold가 {0,3,6}·{1,4,7}·{2,5,8}이
# 되는 배열. 수준이 [0,1,2] 반복이면 그 셋이 통째로 같은 수준이라 test는
# 단일 클래스, train에는 그 수준이 하나도 없어 정확도가 구조적으로 0이 된다.
DEGENERATE_LEVELS = [0, 1, 2, 0, 1, 2, 0, 1, 2]
DEGENERATE_LABEL = np.tile(np.repeat(DEGENERATE_LEVELS, 2), 3)


def test_loso_yields_one_fold_per_subject():
    folds = list(get_splitter("loso").split(SUBJ, TRIAL))
    assert len(folds) == 3


def test_loso_test_fold_is_exactly_one_subject():
    for train_idx, test_idx in get_splitter("loso").split(SUBJ, TRIAL):
        assert len(np.unique(SUBJ[test_idx])) == 1


def test_loso_train_and_test_subjects_are_disjoint():
    for train_idx, test_idx in get_splitter("loso").split(SUBJ, TRIAL):
        assert not (set(SUBJ[train_idx]) & set(SUBJ[test_idx]))


def test_loso_covers_every_row_exactly_once_as_test():
    seen = np.zeros(len(SUBJ), dtype=int)
    for _, test_idx in get_splitter("loso").split(SUBJ, TRIAL):
        seen[test_idx] += 1
    assert (seen == 1).all()


def test_within_subject_keeps_each_fold_inside_one_subject():
    for train_idx, test_idx in get_splitter("within_subject").split(
        WS_SUBJ, WS_TRIAL, WS_LABEL
    ):
        assert len(np.unique(WS_SUBJ[np.concatenate([train_idx, test_idx])])) == 1


def test_within_subject_splits_by_trial_not_by_window():
    for train_idx, test_idx in get_splitter("within_subject").split(
        WS_SUBJ, WS_TRIAL, WS_LABEL
    ):
        assert not (set(WS_TRIAL[train_idx]) & set(WS_TRIAL[test_idx]))


def test_within_subject_test_folds_carry_every_class():
    """각 fold의 test에 세 수준이 모두 들어가야 한다.

    들어가지 않으면 그 fold의 정확도는 판별력이 아니라 다수 클래스
    예측률이 되고, `accuracy_worst`(CLAUDE.md §5.4 필수 지표)가 구조적으로
    바닥에 고정된다.
    """
    all_classes = set(np.unique(WS_LABEL).tolist())
    folds = list(get_splitter("within_subject").split(WS_SUBJ, WS_TRIAL, WS_LABEL))
    assert len(folds) == 9  # 피험자 3명 × fold 3개
    for _, test_idx in folds:
        assert set(WS_LABEL[test_idx].tolist()) == all_classes


def test_within_subject_train_folds_carry_every_class():
    all_classes = set(np.unique(WS_LABEL).tolist())
    for train_idx, _ in get_splitter("within_subject").split(
        WS_SUBJ, WS_TRIAL, WS_LABEL
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
        WS_SUBJ, WS_TRIAL, DEGENERATE_LABEL
    ):
        assert set(DEGENERATE_LABEL[test_idx].tolist()) == all_classes
        assert set(DEGENERATE_LABEL[train_idx].tolist()) == all_classes


def test_within_subject_covers_every_row_exactly_once_as_test():
    seen = np.zeros(len(WS_SUBJ), dtype=int)
    for _, test_idx in get_splitter("within_subject").split(
        WS_SUBJ, WS_TRIAL, WS_LABEL
    ):
        seen[test_idx] += 1
    assert (seen == 1).all()


def test_within_subject_without_labels_raises():
    """라벨 없이 나누면 층화가 불가능하므로 조용히 되돌아가지 않고 거부한다."""
    with pytest.raises(ValueError, match="needs the label array"):
        list(get_splitter("within_subject").split(WS_SUBJ, WS_TRIAL))


def test_within_subject_raises_when_a_class_has_a_single_block():
    """수준당 블록이 1개면 그 블록이 test인 fold의 train에 그 수준이 없다."""
    subj = np.repeat(["sub-01"], 6)
    trial = np.repeat(np.arange(3), 2)
    label = np.repeat([0, 0, 1], 2)   # 수준 1은 블록 1개뿐
    with pytest.raises(ValueError, match="at least 2 blocks per class"):
        list(get_splitter("within_subject").split(subj, trial, label))


def test_window_random_mixes_subjects_across_train_and_test():
    leaked = False
    for train_idx, test_idx in get_splitter("window_random", seed=0).split(SUBJ, TRIAL):
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
    trial = np.array([0, 0, 0, 0] + [0, 1, 0, 1])
    label = np.array([0, 1, 0, 1] + [0, 1, 0, 1])
    with pytest.raises(ValueError, match="within-subject cross-validation needs at least 2 trials"):
        list(get_splitter("within_subject").split(subj, trial, label))
