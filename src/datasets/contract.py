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
from typing import Any, ClassVar, Protocol

import numpy as np


class LeakageError(RuntimeError):
    """train/test 경계를 넘는 정보 흐름이 감지됐을 때 발생."""


@dataclass(frozen=True)
class SplitKeys:
    """분할기가 보는 축들.

    위치 인자를 하나씩 늘리지 않는 이유는, 축이 또 생기면(run, 과제 유형)
    모든 분할기 시그니처가 다시 깨지기 때문이다.
    """

    subject_ids: np.ndarray
    session_ids: np.ndarray
    trial_ids: np.ndarray
    labels: np.ndarray | None


class Splitter(Protocol):
    """분할기 규약. iter_folds가 기대하는 호출 형태다."""

    #: 같은 피험자가 train과 test에 동시에 나타나도 되는 CV 방식인지.
    #: 생략하면 False로 간주한다(피험자 중첩은 위반). within-subject 계열만 True.
    allows_same_subject: ClassVar[bool]

    #: 같은 (피험자, 세션)이 train과 test에 동시에 나타나도 되는지.
    #: 생략하면 False. within-subject는 세션 안에서 시행을 나누므로 True이고,
    #: cross-session은 세션을 가르는 것이 목적이므로 False다.
    allows_same_session: ClassVar[bool]

    def split(self, keys: SplitKeys) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """(train_idx, test_idx) 인덱스 쌍을 fold마다 하나씩 내놓는다.

        `keys.labels`는 층화(stratification)가 필요한 분할기만 쓴다. 라벨을
        분할에 쓰는 것은 누수가 아니다 — 여기 들어오는 라벨은 과제 설계
        (n-back 수준)에서 나온 사전 확정 정보이고, sklearn의
        `StratifiedKFold`가 y를 쓰는 것과 같은 용법이다. 모델 학습은
        여전히 train fold 안에서만 일어난다.
        """
        ...


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

    def session_ids(self) -> np.ndarray:
        return self._ds._session_ids[self._idx]

    def normalization_source(self) -> np.ndarray:
        """이 창이 세션 베이스라인 fit에 쓰였는지."""
        return self._ds._normalization_source[self._idx]


class TrainView(_View):
    """fold의 학습 절반. sklearn 관용구를 그대로 사용할 수 있다."""

    def groups(self) -> np.ndarray:
        """중첩 CV용 그룹 벡터. 항상 피험자 ID다."""
        return self.subject_ids()


class TestView(_View):
    """fold의 평가 절반. 어떤 fit도 허용하지 않는다."""

    __test__ = False  # pytest 수집 대상이 아님

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
        session_ids: np.ndarray,
        window_times: np.ndarray,
        trial_ids: np.ndarray,
        normalization_source: np.ndarray,
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
        for label, arr in (
            ("session_ids", session_ids),
            ("window_times", window_times),
            ("trial_ids", trial_ids),
            ("normalization_source", normalization_source),
        ):
            if len(arr) != n:
                raise ValueError(f"{label} length {len(arr)} != {n}")

        self._X = dict(X)
        self._y = dict(y)
        self._subject_ids = np.asarray(subject_ids)
        self._session_ids = np.asarray(session_ids)
        self._window_times = np.asarray(window_times, dtype=float)
        self._trial_ids = np.asarray(trial_ids)
        self._normalization_source = np.asarray(normalization_source, dtype=bool)

        self._check_trial_ids_globally_unique()

    def _check_trial_ids_globally_unique(self) -> None:
        """한 trial_id가 두 개의 (피험자, 세션)에 걸쳐 있으면 거부한다.

        생성기의 trial_id는 세션마다 0부터 센다. 그대로 쌓으면 sub-01의
        세션0-블록3과 세션1-블록3이 같은 trial_id를 갖고, 시행 단위로
        나누는 분할기가 두 세션의 블록을 한 덩어리로 묶는다. 시행이 더
        이상 일관된 단위가 아니게 되므로, 조립하는 쪽에서 전역 고유하게
        만들 책임을 여기서 강제한다.
        """
        owner: dict = {}
        for trial, subject, session in zip(
            self._trial_ids, self._subject_ids, self._session_ids
        ):
            key = (str(subject), int(session))
            previous = owner.setdefault(trial, key)
            if previous != key:
                raise ValueError(
                    f"trial_id {trial!r} appears under both {previous} and {key}; "
                    "trial_ids must be globally unique across (subject, session). "
                    "조립하는 쪽에서 녹화마다 오프셋을 더해야 한다"
                )

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

    def get_session_ids(self) -> np.ndarray:
        return self._session_ids.copy()

    def class_labels(self, target: str) -> np.ndarray:
        """타깃의 고유 클래스 목록.

        chance level·혼동행렬 라벨 축을 실제 평가 대상에서 유도하기 위한
        메타데이터다. 행 단위 라벨은 여전히 노출하지 않으므로 계약이
        막으려는 것(전역 fit)은 그대로 막힌다.
        """
        if target not in self._y:
            raise KeyError(f"unknown target '{target}'; have {sorted(self._y)}")
        return np.unique(self._y[target])

    def iter_folds(
        self, splitter: Splitter, *, stratify_target: str | None = None
    ) -> Iterator[FoldView]:
        """분할기가 내놓는 fold를 뷰로 감싸 하나씩 내보낸다.

        데이터에 접근하는 유일한 경로다.

        splitter.split(keys)를 `SplitKeys` 객체 하나로 호출한다
        (subject_ids·session_ids·trial_ids·labels를 담는다). `stratify_target`을
        주면 그 타깃의 라벨 배열을 `keys.labels`에 채운다 — 층화
        분할기(within_subject)가 블록을 라벨별로 배정하는 데 쓴다. 주지
        않으면 `keys.labels`는 `None`이다. 분할기는 fold마다
        (train_idx, test_idx) 인덱스 쌍을 하나씩 내놓아야 한다.
        """
        if stratify_target is None:
            labels = None
        else:
            if stratify_target not in self._y:
                raise KeyError(
                    f"unknown target '{stratify_target}'; have {sorted(self._y)}"
                )
            labels = self._y[stratify_target]

        keys = SplitKeys(
            subject_ids=self._subject_ids,
            session_ids=self._session_ids,
            trial_ids=self._trial_ids,
            labels=labels,
        )
        pairs = splitter.split(keys)

        for fold_id, (train_idx, test_idx) in enumerate(pairs):
            yield FoldView(
                fold_id=fold_id,
                train=TrainView(self, np.asarray(train_idx), "train"),
                test=TestView(self, np.asarray(test_idx), "test"),
            )
