import numpy as np
import pytest

from src.evaluation.splitters import get_splitter

SUBJ = np.array(["sub-01"] * 12 + ["sub-02"] * 12 + ["sub-03"] * 12)
TRIAL = np.tile(np.repeat(np.arange(3), 4), 3)


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
    for train_idx, test_idx in get_splitter("within_subject").split(SUBJ, TRIAL):
        assert len(np.unique(SUBJ[np.concatenate([train_idx, test_idx])])) == 1


def test_within_subject_splits_by_trial_not_by_window():
    for train_idx, test_idx in get_splitter("within_subject").split(SUBJ, TRIAL):
        assert not (set(TRIAL[train_idx]) & set(TRIAL[test_idx]))


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
