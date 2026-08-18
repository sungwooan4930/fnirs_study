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
