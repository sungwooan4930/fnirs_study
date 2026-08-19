import numpy as np
import pytest

from src.evaluation.harness import FoldResult
from src.evaluation.metrics import aggregate, aggregate_runs


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


def test_subject_ci_contains_the_mean_with_multiple_folds():
    """fold가 여럿이면 subject_ci는 accuracy_mean을 감싸는 정상적인 t-구간이다."""
    m = aggregate(MIXED, n_classes=3, cv_method="loso")
    assert m["subject_ci_low"] < m["accuracy_mean"] < m["subject_ci_high"]


def test_subject_ci_is_wider_than_pooled_ci_under_window_overlap():
    """subject_ci가 pooled_ci보다 넓어야 한다 — 이것이 이 변경의 존재 이유다.

    5초 창·1초 스텝의 80% 오버랩 때문에 pooled_ci는 창을 독립 시행으로
    취급해 불확실성을 과소평가한다. subject_ci는 fold(≈피험자) 단위로
    세므로 오버랩 상관을 피하고, 그래서 더 넓다(=더 정직하다).

    MIXED는 fold별로 정확도가 크게 갈리는(1.0 vs 0.25) 소규모 데이터라
    fold 표준편차가 크고, pooled_ci는 window 8개를 독립으로 취급해
    좁게 잡힌다 — 두 구간 폭의 차이가 뚜렷하게 드러난다.
    """
    m = aggregate(MIXED, n_classes=3, cv_method="loso")
    pooled_width = m["pooled_ci_high"] - m["pooled_ci_low"]
    subject_width = m["subject_ci_high"] - m["subject_ci_low"]
    assert subject_width > pooled_width, (
        f"subject_ci 폭({subject_width:.4f})이 pooled_ci 폭({pooled_width:.4f})보다 "
        "넓지 않다 — 오버랩된 창을 독립으로 세는 pooled_ci가 여전히 더 넓거나 "
        "같다면 subject_ci를 추가한 목적이 성립하지 않는다"
    )


def test_subject_ci_is_nan_with_a_single_fold():
    """fold가 1개면 표준오차를 정의할 수 없다 — 구간을 지어내지 않는다."""
    single = [_fold(0, "sub-01", [0, 1, 2, 0], [0, 1, 2, 0])]
    m = aggregate(single, n_classes=3, cv_method="loso")
    assert np.isnan(m["subject_ci_low"])
    assert np.isnan(m["subject_ci_high"])


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


def test_aggregate_runs_refuses_to_mix_cv_methods():
    """LOSO와 세션 간 결과를 같은 표에 섞으면 CLAUDE.md §5.4 위반이다."""
    runs = [
        {"cv_method": "loso", "accuracy_mean": 0.65, "chance_level": 1 / 3},
        {"cv_method": "cross_session", "accuracy_mean": 0.71, "chance_level": 1 / 3},
    ]
    with pytest.raises(ValueError, match="cannot mix cv_method"):
        aggregate_runs(runs)


def test_aggregate_runs_combines_same_scheme():
    runs = [
        {"cv_method": "loso", "accuracy_mean": 0.60, "chance_level": 1 / 3},
        {"cv_method": "loso", "accuracy_mean": 0.70, "chance_level": 1 / 3},
    ]
    out = aggregate_runs(runs)
    assert out["cv_method"] == "loso"
    assert out["n_runs"] == 2
    assert out["accuracy_mean"] == pytest.approx(0.65)
    assert out["chance_level"] == pytest.approx(1 / 3)


def test_aggregate_runs_rejects_an_empty_list():
    with pytest.raises(ValueError, match="no runs"):
        aggregate_runs([])
