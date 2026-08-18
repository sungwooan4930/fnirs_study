"""분할기.

sklearn의 split(X, y, groups) 시그니처를 쓰지 않는 이유는
within_subject가 피험자 ID와 블록 ID를 둘 다 필요로 하기 때문이다.
내부적으로는 sklearn의 LeaveOneGroupOut·KFold를 쓴다.
within_subject의 블록 배정은 라벨 층화가 필요해 직접 구현한다.

각 분할기는 클래스 속성 `allows_same_subject`로 "같은 피험자가
train/test 양쪽에 나타나는 것이 이 분할 방식의 정당한 특성인지"를
선언한다. Task 16의 피험자 중첩 가드는 이 속성을 읽어
(`getattr(splitter, "allows_same_subject", False)`) 위반 여부를 판단한다 —
가드는 "train/test에 같은 피험자가 있으면 무조건 실패"가 아니라
"그 분할기가 주장하는 것과 다른 결과가 나오면 실패"로 동작해야 하기 때문이다.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut

from src.datasets.contract import Splitter


class LosoSplitter:
    """Leave-One-Subject-Out. 교차 피험자 일반화를 측정한다.

    train과 test는 절대 같은 피험자를 공유하지 않는다고 주장하는
    분할기다 (`allows_same_subject = False`). 공유가 발생하면 그 자체가
    버그이며 누수로 취급해야 한다.
    """

    allows_same_subject: ClassVar[bool] = False

    def split(self, subject_ids, trial_ids, labels=None):
        # labels는 계약이 전달하지만 LOSO는 쓰지 않는다 (층화 대상이 아님).
        yield from LeaveOneGroupOut().split(
            np.zeros(len(subject_ids)), groups=subject_ids
        )


class WithinSubjectSplitter:
    """피험자별로 블록 단위 K-fold. 블록 배정은 라벨 층화(stratified)다.

    창 단위가 아니라 블록 단위로 나눈다. 창 단위로 나누면 80% 오버랩
    때문에 같은 피험자 안에서도 누수가 생긴다.

    같은 피험자가 train과 test 양쪽에 있는 것이 이 분할 방식의 목적
    그 자체다 (`allows_same_subject = True`). 대신 분리는 시행(trial)
    수준에서 강제되며, 창 단위 중첩 가드는 여전히 실행되어 검증한다.

    **왜 층화가 필요한가.** `build_timeline`은 9개 블록(수준당 3개)의
    순서를 무작위로 섞는다. 그 위에 `GroupKFold(3)`를 얹으면 블록이
    인덱스 stride로 잘려 test fold가 {i, i+3, i+6}을 받는다. 그 셋이
    우연히 같은 n-back 수준이면 — 수준당 블록이 정확히 3개뿐이므로 —
    train에 그 수준의 예시가 하나도 남지 않고 test는 단일 클래스가 되어
    정확도가 구조적으로 0.0000이 된다. 파일럿(12명·3 fold = 36 fold)에서
    실제로 3개 fold가 이렇게 무너졌고, 그 결과 `accuracy_worst`가
    within_subject 전 조건에서 0.0000에 고정됐다 — CLAUDE.md §5.4가
    필수로 요구하는 지표가 아무 정보도 담지 못하게 된 것이다.

    그래서 블록을 라벨별로 모아 fold에 라운드로빈으로 배정한다. 수준당
    블록 3개·fold 3개면 각 fold의 test에 수준당 정확히 1개 블록이 들어간다.

    라벨을 분할에 쓰는 것은 누수가 아니다. 여기서 쓰는 라벨은 과제 설계
    (n-back 수준 = `task_condition`)이며 실험 전에 이미 확정된 정보다.
    sklearn의 `StratifiedKFold`가 y를 쓰는 것과 같은 의미이고, 모델은
    여전히 train fold 안에서만 fit된다.
    """

    allows_same_subject: ClassVar[bool] = True

    def __init__(self, n_splits: int = 3) -> None:
        self.n_splits = n_splits

    @staticmethod
    def _dominant_label(values: np.ndarray):
        """블록 하나의 대표 라벨. 동률이면 작은 라벨을 택해 결정론을 유지한다."""
        uniq, counts = np.unique(values, return_counts=True)
        return uniq[int(np.argmax(counts))]

    def split(self, subject_ids, trial_ids, labels=None):
        if labels is None:
            raise ValueError(
                "WithinSubjectSplitter needs the label array to stratify blocks; "
                "call WindowedDataset.iter_folds(splitter, stratify_target=<target>). "
                "층화 없이 나누면 특정 fold의 train에서 한 수준이 통째로 사라진다."
            )
        labels = np.asarray(labels)

        for subject in np.unique(subject_ids):
            rows = np.flatnonzero(subject_ids == subject)
            trials = trial_ids[rows]
            y = labels[rows]
            unique_trials = np.unique(trials)
            if len(unique_trials) < 2:
                raise ValueError(
                    f"subject '{subject}' has only {len(unique_trials)} unique "
                    "trial(s); within-subject cross-validation needs at "
                    "least 2 trials per subject"
                )

            by_label: dict = {}
            for t in unique_trials:
                by_label.setdefault(self._dominant_label(y[trials == t]), []).append(t)

            min_per_label = min(len(v) for v in by_label.values())
            n_splits = min(self.n_splits, min_per_label)
            if n_splits < 2:
                scarce = sorted(
                    str(k) for k, v in by_label.items() if len(v) == min_per_label
                )
                raise ValueError(
                    f"subject '{subject}' has only {min_per_label} block(s) of "
                    f"class {scarce[0]}; stratified within-subject CV needs at "
                    "least 2 blocks per class so that both train and test carry "
                    "every class in every fold"
                )

            fold_of_trial: dict = {}
            for label in sorted(by_label, key=str):
                for i, t in enumerate(sorted(by_label[label])):
                    fold_of_trial[t] = i % n_splits

            assignment = np.array([fold_of_trial[t] for t in trials])
            for f in range(n_splits):
                is_test = assignment == f
                yield rows[~is_test], rows[is_test]


class WindowRandomSplitter:
    """⚠ 의도적으로 누수를 일으키는 분할기.

    창을 무작위로 섞어 나눈다. 5초 창·1초 스텝은 80% 오버랩이므로
    인접 창이 train과 test에 동시에 들어가고, 같은 피험자가 양쪽에
    존재하게 된다. T3(누수 검출) 시연 전용이며 실제 실험에 쓰면 안 된다.

    이것은 교차 피험자 일반화를 측정한다고 "주장"하는 분할기의 naive한
    대역이므로 (`allows_same_subject = False`), 피험자 중첩 가드는
    반드시 이 분할기에서 실패해야 한다 — 그것이 가드가 실제로 누수를
    잡아낸다는 증거다.
    """

    allows_same_subject: ClassVar[bool] = False

    def __init__(self, n_splits: int = 5, seed: int = 0) -> None:
        self.n_splits = n_splits
        self.seed = seed

    def split(self, subject_ids, trial_ids, labels=None):
        # labels는 계약이 전달하지만 누수 시연용 분할기는 쓰지 않는다.
        yield from KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        ).split(np.zeros(len(subject_ids)))


def get_splitter(name: str, *, seed: int = 0) -> Splitter:
    """이름으로 분할기를 만든다."""
    if name == "loso":
        return LosoSplitter()
    if name == "within_subject":
        return WithinSubjectSplitter()
    if name == "window_random":
        return WindowRandomSplitter(seed=seed)
    raise ValueError(
        f"unknown splitter '{name}'; expected one of "
        "'loso', 'within_subject', 'window_random'"
    )
