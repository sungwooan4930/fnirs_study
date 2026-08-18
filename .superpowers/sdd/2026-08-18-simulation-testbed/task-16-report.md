# Task 16 리포트 — 평가 하네스

## 구현
- `src/evaluation/harness.py`: `FoldResult`(frozen dataclass), `make_model(name, seed)`, `run_folds(dataset, splitter, *, target, modalities, guards, seed, model_name="logistic_regression")`.
- `tests/evaluation/test_harness.py`: 브리프의 8개 테스트 + 컨트롤러 결정이 요구한 `within_subject` 통과 테스트 1개 = 9개.

## 컨트롤러 결정 반영 — `allows_same_subject` 존중
`run_folds`는 `getattr(splitter, "allows_same_subject", False)`를 읽어, 그 값이 `True`인
분할기(`WithinSubjectSplitter`)에는 피험자-중첩 가드(`check_subject_overlap`)를 적용하지
않는다. 속성이 없는 분할기는 안전한 기본값 `False`로 취급해 가드를 그대로 적용한다.
윈도우-겹침 가드(`check_window_overlap`)는 이 예외와 무관하게 항상 실행된다.

```python
allows_same_subject = getattr(splitter, "allows_same_subject", False)
...
if guards.get("check_subject_overlap", True) and not allows_same_subject:
    check_subject_overlap(train_subj, test_subj)
if guards.get("check_window_overlap", True):
    check_window_overlap(...)
```

## `within_subject` 테스트 — 픽스처 함정 발견
브리프의 `make_dataset`을 그대로 재사용해 `within_subject` 테스트를 처음 작성했더니
**실패**했다: `make_dataset`의 trial 블록(`i // 10`)은 시간축에서 연속적이어서
(윈도우 시각이 0..29 구간에 걸쳐 1초 스텝으로 이어짐), trial 경계에 걸친 인접 윈도우
(예: trial 0의 마지막 윈도우 `[9,14]`와 trial 1의 첫 윈도우 `[10,15]`)가 실제로 시간상
겹친다. 이는 `check_window_overlap`이 정확하게 잡아낸 **진짜 누수**이지 가드의 오탐이
아니다 — 컨트롤러 결정 텍스트가 전제한 "within-subject 분할은 시행 경계를 따라 나뉘므로
창이 경계를 넘지 않는다"는 가정은, 실제 실험처럼 **시행 블록 사이에 시간 간격이 있을 때만**
성립한다.

그래서 새 픽스처 `make_block_dataset`을 추가했다: trial마다 시간 기준점을 `t * 100.0`으로
크게 떨어뜨려(윈도우 폭 5초 대비 100초 간격) 블록 간 겹침이 구조적으로 불가능하게 만들었다.
이 픽스처로 재작성한 `test_within_subject_runs_with_both_guards_on`은 두 가드가 모두 켜진
채로 통과한다 — 이것이 컨트롤러 결정이 존재하는 바로 그 경우다.

## 테스트 결과
- `tests/evaluation/test_harness.py`: 9 passed (사전 실패 확인: `ModuleNotFoundError` 1건 → 구현 후 9 passed)
- 전체 스위트: 141 passed (기존 132 + 신규 9), 경고 0건.

로그: `.superpowers/sdd/2026-08-18-simulation-testbed/task-16-testlog.txt`
(개별 파일 실행 → 전체 스위트 실행 순서로 이어붙임. 사전 실패 실행은 별도로 확인만 하고
이 로그 파일에 처음 캡처된 것은 구현 후 결과다 — 사전 실패 원문은 이 리포트에 인용됨)

## leakage-check 스킬 결과
7개 항목 점검 완료. 발견된 누수 없음.
- 분할 단위: `run_folds`는 자체 분할을 하지 않고 `dataset.iter_folds(splitter)`에 위임 — 계약·분할기 계층에서 이미 검증됨.
- 전처리 fit: `model.fit`은 `TrainView`에만 호출됨. `TestView.fit`은 구조적으로 `LeakageError`.
- 윈도우 겹침: `check_window_overlap`이 `allows_same_subject`와 무관하게 항상 실행됨 — 위 픽스처 함정에서 실제로 가드가 살아있음을 재확인.
- 특징 선택·튜닝: 이 하네스에는 없음(N.A.).
- 보고 정합성: `FoldResult`는 raw accuracy만 담고 chance level·LOSO/within-subject 태그는 상위 리포팅 레이어 책임 — 하네스 자체의 결함은 아님.

## 우려 사항
`FoldResult`는 fold 단위 raw 결과만 반환한다. chance level 명시, LOSO vs within-subject 구분
표기(CLAUDE.md §5.4)는 이 결과를 소비하는 다음 단계(리포팅/집계 레이어)의 책임으로 남아있다 —
Task 16 범위 밖이라 여기서는 처리하지 않았다.
