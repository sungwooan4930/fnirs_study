import numpy as np
import pytest

from src.evaluation.harness import FoldResult
from src.evaluation.metrics import aggregate


def _fold(fold_id, subject, y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return FoldResult(
        fold_id=fold_id,
        test_subjects=[subject],
        n_train=100,
        n_test=len(y_true),
        accuracy=float((y_true == y_pred).mean()),
        y_true=y_true,
        y_pred=y_pred,
    )


PERFECT = [_fold(0, "sub-01", [0, 1, 2, 0], [0, 1, 2, 0]),
           _fold(1, "sub-02", [0, 1, 2, 0], [0, 1, 2, 0])]

MIXED = [_fold(0, "sub-01", [0, 1, 2, 0], [0, 1, 2, 0]),
         _fold(1, "sub-02", [0, 1, 2, 0], [1, 1, 1, 1])]


def test_records_cv_method_and_chance():
    m = aggregate(PERFECT, n_classes=3, cv_method="loso")
    assert m["cv_method"] == "loso"
    assert m["chance_level"] == pytest.approx(1 / 3)


def test_counts_folds_and_windows():
    m = aggregate(MIXED, n_classes=3, cv_method="loso")
    assert m["n_folds"] == 2
    assert m["n_windows_evaluated"] == 8


def test_mean_and_worst_accuracy():
    m = aggregate(MIXED, n_classes=3, cv_method="loso")
    assert m["accuracy_mean"] == pytest.approx(0.625)
    assert m["accuracy_worst"] == pytest.approx(0.25)
    assert m["worst_fold_subjects"] == ["sub-02"]


def test_pooled_accuracy_and_ci():
    m = aggregate(PERFECT, n_classes=3, cv_method="loso")
    assert m["pooled_accuracy"] == pytest.approx(1.0)
    assert m["pooled_ci_low"] <= m["pooled_accuracy"] <= m["pooled_ci_high"]


def test_chance_data_has_high_p_value():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=600)
    y_pred = rng.integers(0, 3, size=600)
    m = aggregate([_fold(0, "sub-01", y_true, y_pred)], n_classes=3, cv_method="loso")
    assert m["binomtest_p"] > 0.05
    assert m["pooled_ci_low"] <= 1 / 3 <= m["pooled_ci_high"]


def test_perfect_data_has_low_p_value():
    m = aggregate(PERFECT, n_classes=3, cv_method="loso")
    assert m["binomtest_p"] < 0.05


def test_confusion_matrix_shape_and_total():
    m = aggregate(MIXED, n_classes=3, cv_method="loso")
    conf = np.array(m["confusion"])
    assert conf.shape == (3, 3)
    assert conf.sum() == 8


def test_rejects_empty_results():
    with pytest.raises(ValueError, match="no folds"):
        aggregate([], n_classes=3, cv_method="loso")


def test_accuracy_std_is_sample_standard_deviation():
    """accuracy_std는 표본 표준편차(ddof=1)를 사용한다.

    3개 fold with accuracies [1.0, 0.0, 0.5]:
    - mean = 1.5 / 3 = 0.5
    - sample var = [(1-0.5)² + (0-0.5)² + (0.5-0.5)²] / (3-1)
                 = [0.25 + 0.25 + 0] / 2 = 0.25
    - sample std = sqrt(0.25) = 0.5
    """
    folds = [
        _fold(0, "sub-01", [0, 1], [0, 1]),           # accuracy 1.0
        _fold(1, "sub-02", [0, 1], [1, 0]),           # accuracy 0.0
        _fold(2, "sub-03", [0, 1], [0, 0]),           # accuracy 0.5
    ]
    m = aggregate(folds, n_classes=2, cv_method="loso")
    assert m["accuracy_mean"] == pytest.approx(0.5)
    assert m["accuracy_std"] == pytest.approx(0.5)


def test_accuracy_std_single_fold_returns_zero():
    """단일 fold는 accuracy_std = 0.0 (nan 방지)."""
    single = [_fold(0, "sub-01", [0, 1, 2, 0], [0, 1, 2, 0])]
    m = aggregate(single, n_classes=3, cv_method="loso")
    assert m["accuracy_std"] == 0.0


def test_pooled_accuracy_with_unequal_folds():
    """pooled accuracy는 window 가중치를 사용하고, accuracy_mean과 다르다.

    Fold 0: 10 windows, 9 correct → 90%
    Fold 1: 2 windows, 0 correct → 0%

    - pooled_accuracy = 9 / 12 = 0.75
    - accuracy_mean = (0.9 + 0.0) / 2 = 0.45
    """
    folds = [
        _fold(0, "sub-01", [0]*10, [0]*9 + [1]),  # 9/10
        _fold(1, "sub-02", [0, 0], [1, 1]),        # 0/2
    ]
    m = aggregate(folds, n_classes=2, cv_method="loso")
    assert m["pooled_accuracy"] == pytest.approx(9.0 / 12)
    assert m["accuracy_mean"] == pytest.approx(0.45)
    assert m["pooled_accuracy"] != m["accuracy_mean"]
