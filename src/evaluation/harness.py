"""fold 실행 하네스.

계약이 넘겨주는 FoldView마다 가드를 걸고, train에서만 학습한 모델로
test를 예측한다. sklearn Pipeline을 쓰므로 scaler는 자동으로 train에만
fit된다 — 계약이 막는 것과 같은 규칙을 파이프라인이 지킨다.

피험자 중첩 가드는 분할기의 `allows_same_subject` 선언을 존중한다.
within-subject CV는 같은 피험자가 train/test 양쪽에 있는 것이 설계
그 자체이므로, 그런 분할기에는 이 가드를 적용하지 않는다 — 가드는
"train/test에 같은 피험자가 있으면 무조건 실패"가 아니라 "그 분할기가
주장하는 것과 다른 결과가 나오면 실패"로 동작해야 한다. 창 단위 중첩
가드는 이 예외와 무관하게 항상 실행된다: within-subject 분할은 시행
경계를 따라 나뉘므로 창이 경계를 넘지 않아 가드를 통과하고, 의도적으로
누수를 일으키는 window_random 분할기는 여기서 걸린다.

단일 클래스 fold는 train 쪽뿐 아니라 **test 쪽도** 거부한다. 스펙 §9의
문구는 "특정 fold에서 클래스가 1개뿐 → 실행 거부"이지 "train fold"가
아니다. test가 단일 클래스면 그 fold의 정확도는 판별력이 아니라 다수
클래스 예측률이 되어, 평균과 최악 피험자 지표를 동시에 오염시킨다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.guards import check_subject_overlap, check_window_overlap


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    test_subjects: list[str]
    n_train: int
    n_test: int
    accuracy: float
    y_true: np.ndarray
    y_pred: np.ndarray


def make_model(name: str, seed: int) -> Pipeline:
    """이름으로 sklearn 파이프라인을 만든다."""
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
    raise ValueError(f"unknown model '{name}'; expected 'logistic_regression'")


def run_folds(
    dataset,
    splitter,
    *,
    target: str,
    modalities: list[str],
    guards: dict,
    seed: int,
    model_name: str = "logistic_regression",
) -> list[FoldResult]:
    """모든 fold를 실행하고 결과 목록을 돌려준다."""
    results: list[FoldResult] = []

    allows_same_subject = getattr(splitter, "allows_same_subject", False)

    for fold in dataset.iter_folds(splitter, stratify_target=target):
        train_subj = fold.train.subject_ids()
        test_subj = fold.test.subject_ids()

        if guards.get("check_subject_overlap", True) and not allows_same_subject:
            check_subject_overlap(train_subj, test_subj)
        if guards.get("check_window_overlap", True):
            check_window_overlap(
                train_subj, fold.train.window_times(),
                test_subj, fold.test.window_times(),
            )

        y_train = fold.train.y(target)
        if len(np.unique(y_train)) < 2:
            raise ValueError(
                f"fold {fold.fold_id} train has a single class for target "
                f"'{target}'; accuracy would be meaningless"
            )

        y_true = fold.test.y(target)
        if len(np.unique(y_true)) < 2:
            raise ValueError(
                f"fold {fold.fold_id} test has a single class for target "
                f"'{target}'; accuracy would be meaningless. "
                "스펙 §9는 'train fold'가 아니라 'fold에서 클래스가 1개뿐'이면 "
                "실행을 거부하라고 적고 있다 — test가 단일 클래스인 fold의 "
                "정확도는 다수 클래스 예측률일 뿐 판별력이 아니다."
            )

        model = make_model(model_name, seed)
        model.fit(fold.train.X(modalities), y_train)

        y_pred = model.predict(fold.test.X(modalities))

        results.append(
            FoldResult(
                fold_id=fold.fold_id,
                test_subjects=sorted(set(test_subj.tolist())),
                n_train=len(train_subj),
                n_test=len(test_subj),
                accuracy=float((y_true == y_pred).mean()),
                y_true=y_true,
                y_pred=y_pred,
            )
        )

    return results
