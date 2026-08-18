"""지표 집계.

CLAUDE.md 5.4가 요구하는 대로 chance level과 CV 방식을 항상 함께 담고,
평균만이 아니라 최악 피험자 성능도 담는다. 평균만 보고하면 특정 피험자에서
완전히 실패하는 모델이 좋아 보인다.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix


def aggregate(fold_results, *, n_classes: int, cv_method: str) -> dict:
    """fold 결과를 하나의 지표 dict으로 모은다.

    accuracy_std는 fold들을 표본으로 간주한 표본 표준편차(ddof=1)를 사용한다.
    fold들은 모집단 학습자에서 추출한 표본이므로, 보고되는 "45% ± 8%"는
    사람 간 변동성으로 해석된다. 모집단 공식(ddof=0)은 그 변동성을
    약 √(N/(N−1))만큼 과소평가한다. 단일 fold(N=1)에서는 nan을 피하고
    0.0을 반환한다.
    """
    if not fold_results:
        raise ValueError("no folds to aggregate")

    accuracies = np.array([f.accuracy for f in fold_results])
    y_true = np.concatenate([f.y_true for f in fold_results])
    y_pred = np.concatenate([f.y_pred for f in fold_results])

    n_correct = int((y_true == y_pred).sum())
    n_total = int(len(y_true))
    chance = 1.0 / n_classes

    test = stats.binomtest(n_correct, n_total, chance, alternative="two-sided")
    ci = test.proportion_ci(confidence_level=0.95)

    worst = int(np.argmin(accuracies))

    # accuracy_std: 단일 fold 시 nan 방지, 샘플 표준편차 사용
    if len(fold_results) == 1:
        accuracy_std = 0.0
    else:
        accuracy_std = float(accuracies.std(ddof=1))

    return {
        "cv_method": cv_method,
        "chance_level": float(chance),
        "n_folds": len(fold_results),
        "n_windows_evaluated": n_total,
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": accuracy_std,
        "accuracy_worst": float(accuracies[worst]),
        "worst_fold_subjects": list(fold_results[worst].test_subjects),
        "pooled_accuracy": n_correct / n_total,
        "pooled_ci_low": float(ci.low),
        "pooled_ci_high": float(ci.high),
        "binomtest_p": float(test.pvalue),
        "confusion": confusion_matrix(
            y_true, y_pred, labels=list(range(n_classes))
        ).tolist(),
    }
