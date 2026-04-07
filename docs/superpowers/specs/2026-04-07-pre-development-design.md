# 개발 시작 전 필수 설계 결정 사항

**날짜:** 2026-04-07
**주제:** 본격 개발 전 절대 놓쳐서는 안 되는 것들
**상태:** 승인 완료

---

## 1. 프로젝트 컨텍스트 요약

| 항목 | 내용 |
|------|------|
| 장치 | 8채널 커스텀 fNIRS (전전두엽 측정) |
| 하드웨어 담당 | 별도 인원 (소프트웨어와 분리) |
| 통신 방식 | 미확정 (USB/Serial 또는 BLE) |
| 광원 파장 | 미확정 → 하드웨어 팀 확인 필요 |
| 샘플링 레이트 | 미확정 → 하드웨어 팀 확인 필요 |
| 집중도 알고리즘 | 논문 기반, 구체적 논문 추후 확정 |
| 데이터 저장 방식 | 추후 결정 (HDF5 권장) |
| 대상 사용자 | 일반 사용자 (직관적 UI 필수) |
| 시뮬레이션 모드 | 필수 (하드웨어 없이 전체 개발 가능해야 함) |
| 플랫폼 | 클라이언트·HW 담당자 논의 후 확정 (데스크톱/모바일/웹) |

---

## 2. 선택한 개발 전략: Simulation-First + HAL 패턴

하드웨어 스펙이 미확정인 상태에서 소프트웨어 개발을 즉시 시작하기 위해 선택.

- HAL(Hardware Abstraction Layer) 인터페이스를 먼저 정의
- Real device와 Simulator가 동일한 인터페이스를 구현
- 하드웨어 스펙 확정 전까지 Simulator로 전체 파이프라인 개발 및 테스트
- 하드웨어 완성 시 Real device driver만 교체, 나머지 코드 무변경

---

## 3. 섹션 1: 하드웨어 인터페이스 계약 (최우선 과제)

**개발 시작 전 하드웨어 담당자와 문서로 확정해야 할 5가지:**

| # | 항목 | 이유 |
|---|------|------|
| 1 | **광원 파장 2개** (예: 735nm, 850nm) | mBLL 계산의 입력값. 파장이 바뀌면 신호처리 수식 전체가 달라짐 |
| 2 | **샘플링 레이트** (Hz) | 링 버퍼 크기, 필터 설계, 시각화 업데이트 주기 모두 종속 |
| 3 | **채널-소스/디텍터 매핑표** | 8채널이 어느 소스(S)·디텍터(D) 쌍인지 정의 없으면 토포그래피 불가 |
| 4 | **통신 프로토콜 + 패킷 포맷** | 헤더, 데이터 길이, 체크섬, 채널 순서 등 바이트 수준 명세 |
| 5 | **소스-디텍터 간격(SDS, mm)** | DPF(차분 경로 인수) 계산에 필요. 잘못되면 HbO/HbR 수치 자체가 틀림 |

> 이 5가지는 `config/settings.yaml`에서 관리하며, 코드에 하드코딩하지 않는다.

---

## 4. 섹션 2: 소프트웨어 아키텍처 기반 결정

### 4.1 시스템 레이어 구조

```
[ GUI Layer ]
  실시간 시각화 · 세션 컨트롤 · 레포트 뷰어
        ↕ Qt Signal/Slot (스레드 간 통신)
[ Processing Layer ]
  전처리 · mBLL 계산 · 집중도 지수 · 레포트 생성
        ↕ thread-safe 링 버퍼
[ Acquisition Layer ]
  데이터 수집 스레드 · 버퍼 관리 · 타임스탬프
        ↕ HAL 인터페이스
  ┌──────────────┐   ┌─────────────────┐
  │ Real Device  │   │   Simulator     │
  │ (추후 연결)  │   │ (즉시 개발 가능) │
  └──────────────┘   └─────────────────┘
```

### 4.2 HAL 인터페이스

```python
class FNIRSDevice(ABC):
    def connect(self) -> bool: ...
    def start_stream(self) -> None: ...
    def stop_stream(self) -> None: ...
    def read_packet(self) -> RawPacket: ...  # 채널별 intensity 값
    def disconnect(self) -> None: ...
```

모든 데이터 수집은 이 인터페이스를 통해서만 이루어진다. Real device와 Simulator가 동일하게 구현한다.

### 4.3 스레드 모델

```
[Acquisition Thread] → [Ring Buffer] → [Processing Thread] → [Qt Signal] → [GUI Thread]
```

- Acquisition Thread: 장치에서 패킷 읽어 버퍼에 씀
- Processing Thread: 버퍼에서 읽어 HbO/HbR 계산, 집중도 지수 계산
- GUI Thread: Qt Signal/Slot으로만 데이터 받음 (직접 접근 금지)

### 4.4 데이터 저장

**원시 데이터(raw intensity) + 처리 결과(HbO/HbR, 집중도 지수)를 HDF5로 저장**

- 원시 데이터 보존 → 파장/알고리즘 변경 시 재분석 가능
- 세션 파일명: `data/YYYYMMDD_HHMMSS_session.h5`

### 4.5 집중도 지수 플러그인 구조

```python
class ConcentrationIndex(ABC):
    def compute(self, hbo: np.ndarray, hbr: np.ndarray) -> float: ...

# 추후 논문 기반 알고리즘으로 교체 가능
class SimpleHbOIndex(ConcentrationIndex): ...
class ScholkmannSCI(ConcentrationIndex): ...
```

---

## 5. 섹션 3: 개발 환경 & 시뮬레이션 우선 워크플로

### 5.1 개발 환경 체크리스트

| 항목 | 내용 |
|------|------|
| Python 버전 | 3.10+ |
| 가상환경 | `venv` 또는 `conda` |
| 핵심 패키지 | `numpy`, `scipy`, `pyqt6` or `pyside6`, `pyqtgraph`, `h5py`, `mne-nirs` |
| 테스트 | `pytest` + `pytest-qt` |
| 코드 품질 | `ruff` (lint), `black` (formatter) |
| 버전 관리 | git + `.gitignore` |

### 5.2 시뮬레이션 우선 개발 순서

```
1단계: HAL 인터페이스 정의 + Simulator 구현
2단계: Acquisition Layer + Ring Buffer (Simulator 연결)
3단계: Processing Layer (전처리, mBLL, 집중도 지수 placeholder)
4단계: GUI Layer (실시간 그래프, 집중도 미터)
5단계: Report Layer (HDF5 → 레포트)
6단계: Real Device Driver 연결 (HAL 교체만으로 완성)
```

### 5.3 절대 나중으로 미루면 안 되는 것 3가지

1. **HAL 인터페이스 확정** — 모든 레이어의 기반
2. **스레드 모델 확정** — 나중에 추가 시 전체 구조 재작성
3. **설정 파일(config) 구조** — 파장, 샘플링 레이트, 채널 매핑은 절대 하드코딩 금지

---

## 6. 섹션 4: 플랫폼 독립 아키텍처

### 6.1 핵심 원칙

플랫폼(데스크톱/모바일/웹) 결정은 클라이언트·HW 담당자 논의 후 확정한다.
그러나 어떤 플랫폼이 선택되어도 **Core Library는 버려지지 않도록** 설계한다.

```
┌─────────────────────────────────────────┐
│           Core Library (Python)          │  ← 플랫폼 무관, 항상 유지
│  - HAL / 시뮬레이터                      │
│  - 신호처리 (mBLL, 필터, 집중도 지수)    │
│  - 데이터 저장/로드 (HDF5)              │
│  - 레포트 생성 엔진                      │
└──────────────────┬──────────────────────┘
                   │ 인터페이스(API)
       ┌───────────┼────────────┐
       ↓           ↓            ↓
  [데스크톱 UI]  [웹 UI]   [모바일 FFI]
  PySide6       FastAPI    Flutter/RN
  (프로토타입)  (추후)     (추후)
```

### 6.2 지금 결정할 것 / 미룰 것

| 항목 | 지금 확정 | 미룰 수 있음 |
|------|----------|-------------|
| Core Library 구조 | ✅ | |
| HAL 인터페이스 | ✅ | |
| 데이터 저장 포맷 (HDF5) | ✅ | |
| 집중도 지수 플러그인 구조 | ✅ | |
| UI 플랫폼 | | ✅ 논의 후 |
| 통신 방식 (USB/BLE) | | ✅ HW 담당자 논의 후 |
| 배포 방식 | | ✅ 플랫폼 확정 후 |

### 6.3 크로스플랫폼 개발 시 주의사항 (플랫폼 확정 후 적용)

- 파일 경로: `pathlib.Path` 사용, 문자열 경로 하드코딩 금지
- 시리얼 포트명: 자동 탐지 또는 드롭다운 선택
- 폰트/DPI: Qt `devicePixelRatio` 대응, 고정 픽셀 크기 금지
- 모바일 통신: BLE 지원 여부 HW 팀에 확인 필요

---

## 7. 즉시 시작 가능한 작업 범위

플랫폼 논의와 무관하게 지금 바로 시작할 수 있는 것:

1. **개발 환경 세팅** (Python, venv, 패키지, git)
2. **Core Library 골격** (HAL 인터페이스, 데이터 모델, 디렉토리 구조)
3. **Simulator** (가짜 fNIRS 신호 생성기)
4. **Processing Layer** (필터, mBLL placeholder, 집중도 플러그인 구조)
5. **PySide6 프로토타입 UI** (플랫폼 확정 전 기능 검증용)

---

## 8. 미결 항목 추적

| 항목 | 담당 | 상태 |
|------|------|------|
| 광원 파장 2개 확인 | HW 담당자 | ⏳ 대기 |
| 샘플링 레이트 확인 | HW 담당자 | ⏳ 대기 |
| 채널-소스/디텍터 매핑 | HW 담당자 | ⏳ 대기 |
| 통신 프로토콜 + 패킷 포맷 | HW 담당자 | ⏳ 대기 |
| SDS(소스-디텍터 간격) 확인 | HW 담당자 | ⏳ 대기 |
| UI 플랫폼 확정 | 클라이언트·HW 담당자 | ⏳ 논의 필요 |
| 집중도 지수 논문 선정 | 개발자 | ⏳ 추후 |
| 데이터 저장 방식 확정 | 개발자 | ⏳ 추후 (HDF5 권장) |
