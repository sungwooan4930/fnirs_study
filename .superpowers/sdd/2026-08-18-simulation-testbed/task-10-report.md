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

## 남은 우려

- pytest 수집 경고(`TestView` 이름 충돌)는 기능에 영향 없으나, 향후 pytest 설정에서
  `python_classes` 패턴을 조정하거나 별도 conftest 처리로 억제할지는 미확정 — 이번 태스크 범위 밖.
