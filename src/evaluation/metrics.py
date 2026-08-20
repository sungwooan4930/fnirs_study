"""지표 집계.

CLAUDE.md 5.4가 요구하는 대로 chance level과 CV 방식을 항상 함께 담고,
평균만이 아니라 최악 fold 성능도 담는다. 평균만 보고하면 특정 fold에서
완전히 실패하는 모델이 좋아 보인다.

**주의: `accuracy_worst`·`worst_fold_subjects`가 가리키는 단위는 `cv_method`마다
다르다** (아래 `aggregate()`의 `fold_ci_*` 설명과 같은 이유). `loso`에서는 fold가
피험자이므로 "최악 피험자"가 맞지만, `cross_session`에서는 fold가 **세션**이라
`accuracy_worst`는 최악 **세션**이고 `worst_fold_subjects`는 그 세션에 낀 피험자
전원(보통 여러 명)을 나열한다 — "최악 피험자 1명"으로 읽지 마라.
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

    두 종류의 신뢰구간과 하나의 p값을 함께 담는다 — 모두 같은 결함을 공유하는
    쪽(`pooled_ci_*`, `binomtest_p`)과 그것을 피하는 쪽(`fold_ci_*`)으로 갈린다.

    - `pooled_ci_*`, `binomtest_p`: 모든 창을 독립 베르누이 시행으로 취급한
      이항 신뢰구간·검정 (n = 창 개수). **5초 창·1초 스텝은 80% 오버랩이므로
      창은 독립이 아니다** — 한 표본(피험자·구간)이 창 5개에 걸쳐 들어간다.
      이 둘은 그 상관을 무시하므로 **불확실성을 체계적으로 과소평가**한다.
      세션을 늘려 창 개수(n)가 커질수록 구간은 더 좁아지고 p값은 더 작아지지만
      실제 정보량은 그만큼 늘지 않으므로, n이 클수록 둘 다 더 위험해진다 —
      우연한 잡음을 "통계적으로 유의한 효과"로 착시시키는 방향으로 작동한다.
      `binomtest_p`가 작다는 것을 효과의 증거로 쓰지 마라.
    - `fold_ci_*`: fold 정확도를 표본으로 삼은 t-신뢰구간(자유도 = n_folds − 1).
      오버랩 상관을 자동으로 피한다 — fold 하나는 창 개수와 무관하게 관측치
      하나다. `n_folds < 2`면 표준오차를 추정할 수 없으므로 구간을 지어내지
      않고 `nan`을 둘 다 채운다. **CV 스킴마다 fold가 가리키는 단위가 다르므로
      해석도 다르다 — 스킴이 다른 실행끼리 같은 표에 넣지 마라 (CLAUDE.md §5.4).**
      - `loso`: fold가 곧 피험자다. 이 구간이 **피험자 수준** 불확실성이다.
      - `cross_session`: fold가 세션이다. 보통 세션 수가 적어 t 배수가 커지므로
        이 구간은 **보수적**(넓게 잡힘)이다.
      - `within_subject`: fold가 (피험자 × 세션 내 블록)이라 같은 피험자의
        fold끼리 **서로 상관돼 있다.** 이 구간은 진짜 불확실성보다 좁게 나올
        수 있으므로 **하한으로만** 읽어야 하며, `loso`의 `fold_ci`와 나란히
        비교하면 안 된다.

    `accuracy_worst`·`worst_fold_subjects`도 같은 이유로 스킴별 해석이 다르다
    — fold가 가리키는 단위가 `cv_method`마다 다르므로 "최악의 무엇"인지가
    함께 바뀐다. `loso`는 최악 피험자, `cross_session`은 최악 세션(그리고
    `worst_fold_subjects`는 그 세션의 피험자 전원), `within_subject`는 최악
    (피험자 × 블록)이다. 스킴이 다른 두 실행의 `accuracy_worst`를 "같은 종류의
    최악값"으로 나란히 놓지 마라.
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

    # fold_ci: fold 수준 t-신뢰구간. 오버랩된 창을 표본으로 세지 않는다 —
    # fold 하나가 관측치 하나다. n_folds < 2면 표준오차를 정의할 수 없으므로
    # 구간을 지어내지 않고 nan을 둔다. fold가 무엇을 가리키는지(피험자·세션·
    # 피험자×블록)는 cv_method에 달렸다 — 위 docstring 참조.
    if n_folds < 2:
        fold_ci_low = float("nan")
        fold_ci_high = float("nan")
    else:
        sem = accuracies.std(ddof=1) / np.sqrt(n_folds)
        t_crit = float(stats.t.ppf(0.975, df=n_folds - 1))
        fold_ci_low = float(accuracies.mean() - t_crit * sem)
        fold_ci_high = float(accuracies.mean() + t_crit * sem)

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
        "fold_ci_low": fold_ci_low,
        "fold_ci_high": fold_ci_high,
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
