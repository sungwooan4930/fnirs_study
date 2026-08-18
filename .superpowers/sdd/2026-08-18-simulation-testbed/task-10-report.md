# Task 10 리포트 — 데이터셋 계약 (WindowedDataset / iter_folds)

## 요약

`src/datasets/contract.py`에 `WindowedDataset`, `TrainView`, `TestView`, `FoldView`, `LeakageError`를
브리프에 명시된 코드 그대로 구현했다. `tests/datasets/test_contract.py`도 브리프의 14개 테스트를
그대로 사용했다.

## 절차

1. `tests/datasets/test_contract.py` 작성 (브리프 Step 1 코드 그대로).
2. 구현 전 실패 확인: `ModuleNotFoundError: No module named 'src.datasets.contract'` — 로그 1~19줄.
3. `src/datasets/contract.py` 구현 (브리프 Step 3 코드 그대로, `field` import는 제외).
4. 재실행: 14 passed.
5. 전체 스위트 실행: 83 passed.

## 핵심 설계 — 왜 누수가 구조적으로 불가능한가

- `WindowedDataset`은 일반 클래스(`__init__` 직접 정의)로, 생성자가 `X`/`y`를 받지만
  인스턴스 속성은 `self._X`, `self._y`로만 저장한다. `ds.X`, `ds.y` 같은 공개 속성이 존재하지 않는다
  (`test_no_public_x_or_y_attribute`로 확인).
- 데이터에 닿는 유일한 경로는 `iter_folds(splitter)` → `FoldView(train: TrainView, test: TestView)`.
- `TestView.fit()` / `fit_transform()`은 무조건 `LeakageError`를 던진다. 설정 플래그로 끌 수 없다 —
  분기 없이 항상 예외를 발생시키는 코드이므로 "가드를 끄는" 경로 자체가 코드에 없다.
- `TestView.transform(fitted, modalities)`만 허용 — train에서 이미 fit된 transformer를 받아 적용.
- `get_subject_ids()` / `get_trial_ids()`는 `.copy()`를 반환해 호출자가 배열을 변형해도 내부 상태에
  영향이 없다 (`test_subject_ids_are_exposed_but_copied`로 확인).
- `TrainView.groups()`는 항상 피험자 ID를 반환 — LOSO/GroupKFold 등 중첩 CV에 쓰인다.

## 테스트 증거

로그 파일: `.superpowers/sdd/2026-08-18-simulation-testbed/task-10-testlog.txt`
(사전 실패 실행 + 구현 후 `test_contract.py` 단독 실행 + 전체 스위트 실행, 총 3개 pytest 호출의 출력을
같은 파일에 순서대로 append).

- `tests/datasets/test_contract.py`: **14 passed** (1.39s)
- 전체 스위트 (`pytest -q`): **83 passed** (3.40s), 1 warning

경미한 pytest 수집 경고 1건: `TestView` 클래스명이 `Test`로 시작해 pytest가 테스트 클래스로 오인하고
"cannot collect ... because it has a `__init__` constructor" 경고를 낸다. 실제 테스트 실행에는 영향
없음 (수집만 스킵, 에러 아님). 브리프가 지정한 클래스명이므로 리네임하지 않았다.

## 벗어난 점

없음. 브리프의 코드를 그대로 사용했다.

## 남은 우려 (최초 구현 시점)

- pytest 수집 경고(`TestView` 이름 충돌)는 기능에 영향 없으나, 향후 pytest 설정에서
  `python_classes` 패턴을 조정하거나 별도 conftest 처리로 억제할지는 미확정 — 이번 태스크 범위 밖.
  → **fix round 1에서 해소함 (아래 참조).**

---

## Fix round 1/5 — 스펙 리뷰 반영

리뷰 결과: 배리어 자체(공개 `_X`/`_y` 부재, 뷰의 얕은 복사 아님, `X(modalities)` 순서 보존)는
직접 프로빙 검증을 통과. 정합성에 영향 없는 3건의 개선 요청을 반영했다.

### 1. (Important) `Splitter` Protocol 추가

`iter_folds`가 받는 `splitter` 인자가 `Any`로만 타입되어 있고 호출 규약이 테스트 파일의
`DummySplitter`에만 암묵적으로 존재했다. `src/datasets/contract.py`에 다음을 추가:

```python
class Splitter(Protocol):
    """분할기 규약. iter_folds가 기대하는 호출 형태다."""

    def split(
        self, subject_ids: np.ndarray, trial_ids: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """(train_idx, test_idx) 인덱스 쌍을 fold마다 하나씩 내놓는다."""
        ...
```

- `iter_folds`의 인자 타입을 `Any` → `Splitter`로 변경.
- `iter_folds` 독스트링에 `split(subject_ids, trial_ids)`를 위치 인자 순서대로 호출한다는 것과,
  fold마다 `(train_idx, test_idx)` 쌍을 하나씩 내놓아야 한다는 것을 명시.
- `typing`에서 `Protocol`을 추가로 import.
- 이 Protocol은 (미래의) `splitters.py`가 아니라 `contract.py`에 둔다 — 소비자가 계약을 정의하고
  구현체가 그것을 import하는 방향. Task 14는 여기서 `Splitter`를 import해야 한다.

### 2. (Minor) `TestView`에 `__test__ = False` 추가

`TestView`가 `Test`로 시작해 pytest가 테스트 클래스로 오인, 매 실행마다
`PytestCollectionWarning: cannot collect test class 'TestView' because it has a __init__ constructor`
경고를 냈다. `TestView`에 `__test__ = False` 클래스 속성을 추가해 pytest 수집 대상에서 제외했다.
이것은 pytest 메타데이터일 뿐 데이터셋 계약의 공개 표면이 아니므로 브리프 코드를 그대로 쓴다는
원칙과 충돌하지 않는다.

### 3. (Minor) 호출자 지정 순서 보존 테스트 추가

기존 `test_view_x_concatenates_modalities_in_given_order`는 `["eeg", "fnirs"]`를 호출하는데
이는 정렬 순서와 우연히 같아서, 구현이 내부적으로 모달리티를 정렬해도 통과했을 것이다.
`test_view_x_respects_caller_order_not_sorted_order`를 추가: `["fnirs", "eeg"]`(정렬 역순, 폭도
다름 — fnirs 3열 vs eeg 4열)를 호출해 그 결과가 `X(["fnirs"])`와 `X(["eeg"])`를 그 순서로
hstack한 것과 일치하고, `["eeg", "fnirs"]` 호출 결과와는 다름을 확인한다. (사설 속성 접근을
피하기 위해 공개 API `fold.train.X([...])`만으로 기대값을 구성했다.)

### 결과

```bash
cd /d/Study_fNIRS
.venv/Scripts/pytest.exe tests/datasets/test_contract.py -v > .superpowers/sdd/2026-08-18-simulation-testbed/task-10-testlog.txt 2>&1
.venv/Scripts/pytest.exe -q >> .superpowers/sdd/2026-08-18-simulation-testbed/task-10-testlog.txt 2>&1
```

- `tests/datasets/test_contract.py`: **15 passed** (1.42s) — 새 테스트 1개 추가로 14 → 15.
- 전체 스위트: **84 passed** (3.41s) — **경고 0건** (이전 1건에서 감소).

로그 파일은 이 fix round의 pytest 호출 결과로 덮어써졌다 (coordinator 지정 명령이 `>`이므로).
최초 구현 시점의 사전 실패 로그·13개 통과 로그는 위 "절차"/"테스트 증거" 절 서술로 남아 있다.

### 남은 우려

없음. 세 가지 요청 모두 반영했고 경고 0건을 커맨드 실행으로 직접 확인했다.
