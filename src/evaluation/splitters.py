"""분할기.

sklearn의 split(X, y, groups) 시그니처를 쓰지 않는 이유는
within_subject가 피험자 ID와 블록 ID를 둘 다 필요로 하기 때문이다.
내부적으로는 sklearn의 LeaveOneGroupOut·GroupKFold·KFold를 쓴다.

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
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

from src.datasets.contract import Splitter


class LosoSplitter:
    """Leave-One-Subject-Out. 교차 피험자 일반화를 측정한다.

    train과 test는 절대 같은 피험자를 공유하지 않는다고 주장하는
    분할기다 (`allows_same_subject = False`). 공유가 발생하면 그 자체가
    버그이며 누수로 취급해야 한다.
    """

    allows_same_subject: ClassVar[bool] = False

    def split(self, subject_ids, trial_ids):
        yield from LeaveOneGroupOut().split(
            np.zeros(len(subject_ids)), groups=subject_ids
        )


class WithinSubjectSplitter:
    """피험자별로 블록 단위 K-fold.

    창 단위가 아니라 블록 단위로 나눈다. 창 단위로 나누면 80% 오버랩
    때문에 같은 피험자 안에서도 누수가 생긴다.

    같은 피험자가 train과 test 양쪽에 있는 것이 이 분할 방식의 목적
    그 자체다 (`allows_same_subject = True`). 대신 분리는 시행(trial)
    수준에서 `GroupKFold(groups=trial_ids)`로 강제되며, 창 단위 중첩
    가드는 여전히 실행되어 검증한다.
    """

    allows_same_subject: ClassVar[bool] = True

    def __init__(self, n_splits: int = 3) -> None:
        self.n_splits = n_splits

    def split(self, subject_ids, trial_ids):
        for subject in np.unique(subject_ids):
            rows = np.flatnonzero(subject_ids == subject)
            trials = trial_ids[rows]
            n_unique_trials = len(np.unique(trials))
            if n_unique_trials < 2:
                raise ValueError(
                    f"subject '{subject}' has only {n_unique_trials} unique "
                    "trial(s); within-subject cross-validation needs at "
                    "least 2 trials per subject"
                )
            n_splits = min(self.n_splits, n_unique_trials)
            for tr, te in GroupKFold(n_splits=n_splits).split(
                np.zeros(len(rows)), groups=trials
            ):
                yield rows[tr], rows[te]


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

    def split(self, subject_ids, trial_ids):
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
