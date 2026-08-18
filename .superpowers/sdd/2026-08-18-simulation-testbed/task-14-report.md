# Task 14 리포트 — 분할기

## 구현
- `src/evaluation/splitters.py`: `LosoSplitter`, `WithinSubjectSplitter`, `WindowRandomSplitter`, `get_splitter(name, *, seed=0)`.
- `tests/evaluation/test_splitters.py`: 브리프의 8개 테스트 + 컨트롤러 결정 2에서 요구한 `allows_same_subject` 검증 3개 = 11개.

## 컨트롤러 결정 반영
1. **`Splitter` 재선언 금지** — `splitters.py`는 자체 `Protocol`을 선언하지 않고
   `from src.datasets.contract import Splitter`로 가져와 반환 타입 힌트에 사용한다.
2. **`allows_same_subject` 클래스 속성**
   - `LosoSplitter.allows_same_subject = False` (교차 피험자 주장 → 공유는 위반)
   - `WithinSubjectSplitter.allows_same_subject = True` (같은 피험자가 목적 그 자체;
     분리는 trial 단위 `GroupKFold`가 담당)
   - `WindowRandomSplitter.allows_same_subject = False` (교차 피험자 평가의 naive 대역;
     가드가 반드시 여기서 실패해야 함)
   - 세 클래스 docstring에 각각 의미를 명시했고, 세 클래스 모두 값을 검증하는 테스트를 추가했다.

## 테스트 결과
- `tests/evaluation/test_splitters.py`: 11 passed (사전 실패 확인: `ModuleNotFoundError` 1건 → 구현 후 11 passed)
- 전체 스위트: 125 passed (기존 114 + 신규 11), 경고 0건.

로그: `.superpowers/sdd/2026-08-18-simulation-testbed/task-14-testlog.txt`
(사전 실패 실행 → 구현 후 개별 파일 실행 → 전체 스위트 실행 순서로 이어붙임)

## 우려 사항
없음. `within_subject`는 브리프대로 블록(trial) 단위 `GroupKFold`를 쓰므로 80% 윈도우 오버랩 누수를 피한다.
`window_random`은 의도적으로 누수를 일으키는 분할기임을 docstring에 명시했다.

---

## Fix round 1/5 (coordinator spec review)

### 1 (Important) — `WithinSubjectSplitter` 조용한 실패 수정
- `src/evaluation/splitters.py`: trial 수가 2 미만인 피험자를 만나면 더 이상 `continue`로
  건너뛰지 않고, 어떤 피험자가 몇 개의 trial만 가졌는지 명시한 `ValueError`를 던진다
  ("within-subject cross-validation needs at least 2 trials per subject").
- 기존 `n_splits = min(self.n_splits, n_unique_trials)` 클램프는 그대로 유지 —
  trial이 2개인 피험자의 fold 수를 줄이는 것은 정당하며, 문제는 오직 "조용한 소멸"이었다.
- `tests/evaluation/test_splitters.py`에 `test_within_subject_raises_on_subject_with_single_trial`
  추가: trial이 1개뿐인 피험자를 만들어 `pytest.raises(ValueError, match=...)`로 검증.

### 2 (Minor) — `allows_same_subject`를 `Splitter` Protocol에 문서화
- `src/datasets/contract.py`: `typing` import에 `ClassVar` 추가, `Splitter(Protocol)`에
  `allows_same_subject: ClassVar[bool]`을 주석과 함께 선언 (생략 시 False로 간주,
  within-subject 계열만 True).
- Protocol에 선언한다고 구현체에 강제되지는 않는다 — Task 16의
  `getattr(splitter, "allows_same_subject", False)`가 여전히 안전한 기본값을 보장한다.
  순수 문서화 목적이며 새 요구사항이 아니다.
- 기존 `tests/datasets/test_contract.py` 27개 항목 중 관련 테스트 모두 그대로 통과 확인.

### 테스트 결과
- `tests/evaluation/test_splitters.py` + `tests/datasets/test_contract.py`: **27 passed**
  (분할기 12개 [기존 11 + 신규 1] + contract 15개)
- 전체 스위트: **126 passed** (fix round 이전 125 + 신규 1), 경고 0건.

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.0.2, pluggy-1.6.0 -- D:\Study_fNIRS\.venv\Scripts\python.exe
cachedir: .pytest_cache
PySide6 6.7.0 -- Qt runtime 6.7.0 -- Qt compiled 6.7.0
rootdir: D:\Study_fNIRS
configfile: pytest.ini
plugins: qt-4.5.0
collecting ... collected 27 items

tests/evaluation/test_splitters.py::test_loso_yields_one_fold_per_subject PASSED [  3%]
tests/evaluation/test_splitters.py::test_loso_test_fold_is_exactly_one_subject PASSED [  7%]
tests/evaluation/test_splitters.py::test_loso_train_and_test_subjects_are_disjoint PASSED [ 11%]
tests/evaluation/test_splitters.py::test_loso_covers_every_row_exactly_once_as_test PASSED [ 14%]
tests/evaluation/test_splitters.py::test_within_subject_keeps_each_fold_inside_one_subject PASSED [ 18%]
tests/evaluation/test_splitters.py::test_within_subject_splits_by_trial_not_by_window PASSED [ 22%]
tests/evaluation/test_splitters.py::test_window_random_mixes_subjects_across_train_and_test PASSED [ 25%]
tests/evaluation/test_splitters.py::test_unknown_splitter_name_raises PASSED [ 29%]
tests/evaluation/test_splitters.py::test_loso_allows_same_subject_is_false PASSED [ 33%]
tests/evaluation/test_splitters.py::test_within_subject_allows_same_subject_is_true PASSED [ 37%]
tests/evaluation/test_splitters.py::test_window_random_allows_same_subject_is_false PASSED [ 40%]
tests/evaluation/test_splitters.py::test_within_subject_raises_on_subject_with_single_trial PASSED [ 44%]
tests/datasets/test_contract.py::test_no_public_x_or_y_attribute PASSED  [ 48%]
tests/datasets/test_contract.py::test_basic_properties PASSED            [ 51%]
tests/datasets/test_contract.py::test_subject_ids_are_exposed_but_copied PASSED [ 55%]
tests/datasets/test_contract.py::test_iter_folds_yields_views PASSED     [ 59%]
tests/datasets/test_contract.py::test_view_x_concatenates_modalities_in_given_order PASSED [ 62%]
tests/datasets/test_contract.py::test_view_x_respects_caller_order_not_sorted_order PASSED [ 66%]
tests/datasets/test_contract.py::test_view_x_rejects_unknown_modality PASSED [ 70%]
tests/datasets/test_contract.py::test_view_y_rejects_unknown_target PASSED [ 74%]
tests/datasets/test_contract.py::test_train_and_test_have_disjoint_rows PASSED [ 77%]
tests/datasets/test_contract.py::test_testview_fit_raises_leakage_error PASSED [ 81%]
tests/datasets/test_contract.py::test_testview_fit_transform_raises_leakage_error PASSED [ 85%]
tests/datasets/test_contract.py::test_testview_transform_with_fitted_transformer_is_allowed PASSED [ 88%]
tests/datasets/test_contract.py::test_trainview_groups_returns_subject_ids PASSED [ 92%]
tests/datasets/test_contract.py::test_rejects_length_mismatch PASSED     [ 96%]
tests/datasets/test_contract.py::test_rejects_empty_dataset PASSED       [100%]

============================= 27 passed in 1.39s ==============================
........................................................................ [ 57%]
......................................................                   [100%]
126 passed in 5.13s
```

### 우려 사항
없음.
