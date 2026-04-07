# fNIRS 집중도 분석 시스템 - 프로젝트 지침

## 프로젝트 개요

8채널 커스텀 fNIRS(기능적 근적외선 분광법) 장치를 이용해 사용자의 전전두엽(PFC) 신호를 실시간으로 측정·분석하고, 시각화 및 집중도 분석 레포트를 생성하는 소프트웨어 시스템 개발.

---

## 핵심 기능

### 1. 실시간 데이터 수집
- 8채널 fNIRS 장치에서 원시 신호(raw signal) 수신
- 시리얼/USB 통신 인터페이스를 통한 데이터 스트리밍
- 타임스탬프 기반 데이터 동기화

### 2. 신호 전처리
- 모션 아티팩트 제거
- 베이스라인 드리프트 보정
- HbO(옥시헤모글로빈) / HbR(디옥시헤모글로빈) 농도 변화 계산 (수정된 Beer-Lambert 법칙)

### 3. 실시간 시각화
- 채널별 HbO/HbR 시계열 그래프 (실시간 업데이트)
- 전전두엽 활성화 토포그래피 맵 (선택 사항)
- 집중도 지수(Concentration Index) 실시간 표시

### 4. 측정 후 분석 레포트
- 세션 요약 (측정 시간, 채널 품질 등)
- 시간대별 집중도 변화 그래프
- 채널별 신호 품질 평가
- 집중 구간 / 분산 구간 자동 검출
- PDF 또는 HTML 형태 레포트 출력

---

## 기술 스택 (예정)

| 분류 | 도구/라이브러리 |
|------|----------------|
| 언어 | Python 3.10+ |
| 데이터 수집 | pyserial / 커스텀 드라이버 |
| 신호 처리 | NumPy, SciPy, MNE-NIRS |
| 실시간 시각화 | PyQtGraph 또는 Matplotlib (animation) |
| GUI 프레임워크 | PyQt5 / PySide6 |
| 레포트 생성 | ReportLab (PDF) 또는 Jinja2+HTML |
| 데이터 저장 | HDF5 (h5py) 또는 CSV |

---

## 디렉토리 구조 (계획)

```
D:/Study_fNIRS/
├── PROJECT_GUIDELINES.md       # 이 파일
├── run_logging.md              # 진행 로그
├── src/
│   ├── acquisition/            # 데이터 수집 모듈
│   ├── processing/             # 신호 처리 모듈
│   ├── visualization/          # 실시간 시각화 모듈
│   ├── report/                 # 레포트 생성 모듈
│   └── main.py                 # 진입점
├── config/
│   └── settings.yaml           # 장치 설정, 채널 매핑 등
├── data/                       # 측정 데이터 저장
├── reports/                    # 생성된 레포트
└── tests/                      # 단위 테스트
```

---

## 개발 프로세스 (필수)

모든 기능 개발은 아래 단계를 **순서대로** 거친다. 단계를 건너뛰지 않는다.

```
1. Brainstorm  →  2. Plan  →  3. TDD(테스트 작성)  →  4. Implement  →  5. Debug  →  6. Verify  →  7. Code Review
```

| 단계 | Superpowers 스킬 | 설명 |
|------|-----------------|------|
| Brainstorm | `superpowers:brainstorming` | 요구사항 탐색, 설계 의도 파악 |
| Plan | `superpowers:writing-plans` | 구체적 구현 계획 수립 |
| TDD | `superpowers:test-driven-development` | 구현 전 테스트 먼저 작성 |
| Implement | `superpowers:executing-plans` | 계획 기반 코드 구현 |
| Debug | `superpowers:systematic-debugging` | 버그/테스트 실패 시 원인 분석 |
| Verify | `superpowers:verification-before-completion` | 완료 선언 전 검증 |
| Code Review | `superpowers:requesting-code-review` | 주요 단계 완료 후 리뷰 |

> 독립적인 하위 작업이 2개 이상인 경우 `superpowers:dispatching-parallel-agents` 또는 `superpowers:subagent-driven-development`로 병렬 처리한다.

---

## 개발 원칙

1. **모듈화**: 수집 / 처리 / 시각화 / 레포트 레이어를 명확히 분리
2. **실시간성 우선**: 시각화 지연 최소화, 버퍼 관리 최적화
3. **신호 무결성**: 원시 데이터는 항상 저장, 전처리 파라미터는 설정 파일로 관리
4. **재현성**: 동일 데이터에 동일 분석 결과 보장
5. **사용성**: 연구자/임상가가 쉽게 사용할 수 있는 직관적 UI

---

## 코딩 규칙

- PEP 8 준수
- 함수/클래스에 docstring 작성 (새로 만드는 코드에 한함)
- 타입 힌트 사용 (새로 만드는 함수에 한함)
- 단위 테스트: 신호 처리 핵심 함수는 반드시 테스트 작성
- 커밋 메시지: `[모듈] 변경 내용 요약` 형식

---

## 주요 참고 사항

- fNIRS 채널: 8채널, 전전두엽(Fp1, Fp2, AF3, AF4 인근) 배치 예상
- 샘플링 레이트: 장치 스펙 확인 후 확정
- 집중도 지수: HbO 증가 + HbR 감소 패턴 기반, 추후 알고리즘 상세 정의
