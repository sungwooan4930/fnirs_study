"""데이터셋 계약 — 누수를 구조적으로 차단하는 계층.

WindowedDataset은 X·y를 직접 노출하는 프로퍼티를 두지 않는다.
iter_folds()를 거치지 않으면 특징과 라벨에 닿을 수 없으므로
"전체 데이터에 scaler를 fit"하는 코드를 작성하는 것 자체가 불가능해진다.

TestView는 fit 계열 호출에서 LeakageError를 던진다. 이것은 설정으로
끌 수 없는 구조적 장치다 (스펙 §8 "가드 비활성화에 대하여").

fold 내부에서는 sklearn Pipeline·GridSearchCV를 평소대로 쓴다.
이 계층은 sklearn을 대체하지 않고 감싼다.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


class LeakageError(RuntimeError):
    """train/test 경계를 넘는 정보 흐름이 감지됐을 때 발생."""


class _View:
    """fold 한쪽 절반에 대한 읽기 전용 뷰."""

    def __init__(self, dataset: WindowedDataset, indices: np.ndarray, kind: str) -> None:
        self._ds = dataset
        self._idx = indices
        self._kind = kind

    def __len__(self) -> int:
        return len(self._idx)

    def X(self, modalities: list[str]) -> np.ndarray:
        """지정한 모달리티를 주어진 순서대로 가로로 이어붙여 반환한다."""
        blocks = []
        for name in modalities:
            if name not in self._ds._X:
                raise KeyError(f"unknown modality '{name}'; have {sorted(self._ds._X)}")
            blocks.append(self._ds._X[name][self._idx])
        return np.hstack(blocks)

    def y(self, target: str) -> np.ndarray:
        if target not in self._ds._y:
            raise KeyError(f"unknown target '{target}'; have {sorted(self._ds._y)}")
        return self._ds._y[target][self._idx]

    def subject_ids(self) -> np.ndarray:
        return self._ds._subject_ids[self._idx]

    def window_times(self) -> np.ndarray:
        return self._ds._window_times[self._idx]


class TrainView(_View):
    """fold의 학습 절반. sklearn 관용구를 그대로 사용할 수 있다."""

    def groups(self) -> np.ndarray:
        """중첩 CV용 그룹 벡터. 항상 피험자 ID다."""
        return self.subject_ids()


class TestView(_View):
    """fold의 평가 절반. 어떤 fit도 허용하지 않는다."""

    def transform(self, fitted_transformer: Any, modalities: list[str]) -> np.ndarray:
        """train에서 이미 fit된 변환기를 적용한다."""
        return fitted_transformer.transform(self.X(modalities))

    def fit(self, *args: Any, **kwargs: Any) -> None:
        raise LeakageError(
            "cannot fit on the test view — fit only on TrainView. "
            "test 데이터로 fit하면 그 fold의 성능 수치는 폐기 대상이다."
        )

    def fit_transform(self, *args: Any, **kwargs: Any) -> None:
        raise LeakageError(
            "cannot fit_transform on the test view — fit on TrainView, "
            "then call TestView.transform(fitted, modalities)."
        )


@dataclass(frozen=True)
class FoldView:
    fold_id: int
    train: TrainView
    test: TestView


class WindowedDataset:
    """윈도우 단위 특징·라벨 묶음.

    생성자는 X/y/... 를 인자로 받지만 인스턴스에는 밑줄 이름으로만 보관한다.
    외부에서 ds.X 로 접근할 수 없다는 것이 이 클래스의 요점이므로
    dataclass를 쓰지 않고 __init__을 직접 정의한다.
    """

    def __init__(
        self,
        X: dict[str, np.ndarray],
        y: dict[str, np.ndarray],
        subject_ids: np.ndarray,
        window_times: np.ndarray,
        trial_ids: np.ndarray,
    ) -> None:
        n = len(subject_ids)
        if n == 0:
            raise ValueError("dataset is empty")

        for name, arr in X.items():
            if len(arr) != n:
                raise ValueError(f"modality '{name}' length {len(arr)} != {n}")
        for name, arr in y.items():
            if len(arr) != n:
                raise ValueError(f"target '{name}' length {len(arr)} != {n}")
        if len(window_times) != n:
            raise ValueError(f"window_times length {len(window_times)} != {n}")
        if len(trial_ids) != n:
            raise ValueError(f"trial_ids length {len(trial_ids)} != {n}")

        self._X = dict(X)
        self._y = dict(y)
        self._subject_ids = np.asarray(subject_ids)
        self._window_times = np.asarray(window_times, dtype=float)
        self._trial_ids = np.asarray(trial_ids)

    @property
    def n_windows(self) -> int:
        return len(self._subject_ids)

    @property
    def modalities(self) -> list[str]:
        return sorted(self._X)

    @property
    def targets(self) -> list[str]:
        return sorted(self._y)

    def get_subject_ids(self) -> np.ndarray:
        return self._subject_ids.copy()

    def get_trial_ids(self) -> np.ndarray:
        return self._trial_ids.copy()

    def iter_folds(self, splitter: Any) -> Iterator[FoldView]:
        """분할기가 내놓는 fold를 뷰로 감싸 하나씩 내보낸다.

        데이터에 접근하는 유일한 경로다.
        """
        for fold_id, (train_idx, test_idx) in enumerate(
            splitter.split(self._subject_ids, self._trial_ids)
        ):
            yield FoldView(
                fold_id=fold_id,
                train=TrainView(self, np.asarray(train_idx), "train"),
                test=TestView(self, np.asarray(test_idx), "test"),
            )
