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

    두 종류의 신뢰구간을 함께 담는다 — 서로 다른 것을 잰다.

    - `pooled_ci_*`: 모든 창을 독립 베르누이 시행으로 취급한 이항 신뢰구간
      (n = 창 개수). **5초 창·1초 스텝은 80% 오버랩이므로 창은 독립이 아니다**
      — 한 표본(피험자·구간)이 창 5개에 걸쳐 들어간다. 이 구간은 그 상관을
      무시하므로 **불확실성을 체계적으로 과소평가**한다. 세션을 늘려 창
      개수(n)가 커질수록 구간은 더 좁아지지만 실제 정보량은 그만큼 늘지
      않으므로, n이 클수록 이 구간은 더 위험해진다 — 우연한 잡음을
      "통계적으로 유의한 효과"로 착시시키는 방향으로 작동한다.
    - `subject_ci_*`: fold(≈피험자) 정확도를 표본으로 삼은 t-신뢰구간
      (자유도 = n_folds − 1). 오버랩 상관을 자동으로 피한다 — fold 하나는
      창 개수와 무관하게 관측치 하나다. **분포(널 데이터가 chance를
      포함하는지 등) 판정에는 이 구간을 쓴다.** `n_folds < 2`면 표준오차를
      추정할 수 없으므로 구간을 지어내지 않고 `nan`을 둘 다 채운다.
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

    n_folds = len(fold_results)

    # accuracy_std: 단일 fold 시 nan 방지, 샘플 표준편차 사용
    if n_folds == 1:
        accuracy_std = 0.0
    else:
        accuracy_std = float(accuracies.std(ddof=1))

    # subject_ci: fold(≈피험자) 수준 t-신뢰구간. 오버랩된 창을 표본으로
    # 세지 않는다 — fold 하나가 관측치 하나다. n_folds < 2면 표준오차를
    # 정의할 수 없으므로 구간을 지어내지 않고 nan을 둔다.
    if n_folds < 2:
        subject_ci_low = float("nan")
        subject_ci_high = float("nan")
    else:
        sem = accuracies.std(ddof=1) / np.sqrt(n_folds)
        t_crit = float(stats.t.ppf(0.975, df=n_folds - 1))
        subject_ci_low = float(accuracies.mean() - t_crit * sem)
        subject_ci_high = float(accuracies.mean() + t_crit * sem)

    return {
        "cv_method": cv_method,
        "chance_level": float(chance),
        "n_folds": n_folds,
        "n_windows_evaluated": n_total,
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": accuracy_std,
        "accuracy_worst": float(accuracies[worst]),
        "worst_fold_subjects": list(fold_results[worst].test_subjects),
        "pooled_accuracy": n_correct / n_total,
        "pooled_ci_low": float(ci.low),
        "pooled_ci_high": float(ci.high),
        "subject_ci_low": subject_ci_low,
        "subject_ci_high": subject_ci_high,
        "binomtest_p": float(test.pvalue),
        "confusion": confusion_matrix(
            y_true, y_pred, labels=list(range(n_classes))
        ).tolist(),
    }


def aggregate_runs(runs: list[dict]) -> dict:
    """여러 실행의 지표를 합친다. CV 방식이 다르면 거부한다.

    LOSO(교차 피험자)와 within-subject·cross-session은 서로 다른 질문에
    답하는 수치다. 같은 표에 넣으면 평균 자체가 의미를 잃고, 읽는 사람은
    그 사실을 알 수 없다 (CLAUDE.md §5.4).
    """
    if not runs:
        raise ValueError("no runs to aggregate")

    methods = sorted({r["cv_method"] for r in runs})
    if len(methods) > 1:
        raise ValueError(
            f"cannot mix cv_method values {methods} in one aggregate; "
            "LOSO 결과와 within-subject·cross-session 결과를 혼용 표기하는 것은 "
            "CLAUDE.md §5.4가 금지한다. 스킴별로 따로 집계하라"
        )

    chances = sorted({round(float(r["chance_level"]), 6) for r in runs})
    if len(chances) > 1:
        raise ValueError(
            f"cannot mix chance levels {chances} in one aggregate; "
            "클래스 수가 다른 실행들이다"
        )

    accuracies = np.array([float(r["accuracy_mean"]) for r in runs])
    return {
        "cv_method": methods[0],
        "chance_level": chances[0],
        "n_runs": len(runs),
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": float(accuracies.std(ddof=1)) if len(runs) > 1 else 0.0,
        "accuracy_worst": float(accuracies.min()),
    }
