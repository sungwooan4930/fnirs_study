# 개발 진행 로그

> 모든 진행 상황, 결정 사항, 변경 내역을 이 파일에 기록한다.
> 형식: `[YYYY-MM-DD] 제목` → 내용

---

## 2026-04-07 — 프로젝트 초기화

### 완료
- `PROJECT_GUIDELINES.md` 작성: 프로젝트 목표, 핵심 기능, 기술 스택, 디렉토리 구조, 개발 원칙 정의
- `run_logging.md` 작성: 진행 로그 파일 생성

### 프로젝트 목표 확정
- 8채널 커스텀 fNIRS로 전전두엽 신호 실시간 측정
- 실시간 HbO/HbR 시각화
- 측정 종료 후 집중도 분석 레포트 자동 생성

### 다음 단계 (결정 필요)
- [ ] 장치 통신 방식 확인 (시리얼 포트, USB HID, BLE 등)
- [ ] 샘플링 레이트 및 데이터 포맷 확인
- [ ] 채널 배치도 및 소스-디텍터 쌍 정의
- [ ] 집중도 지수(Concentration Index) 알고리즘 설계
- [ ] GUI 프레임워크 최종 선택 (PyQt5 vs PySide6)
- [ ] 개발 환경 세팅 (가상환경, 패키지 목록)

---

## 2026-04-07 — 개발 프로세스 확정

### 결정 사항
- 모든 기능 개발에 superpowers 스킬 기반 프로세스 의무화
- 프로세스: **Brainstorm → Plan → TDD → Implement → Debug → Verify → Code Review**
- 각 단계마다 대응하는 superpowers 스킬을 명시적으로 호출

### 업데이트 파일
- `PROJECT_GUIDELINES.md` — "개발 프로세스" 섹션 추가
- `run_logging.md` — 결정 사항 기록

---

## 2026-04-07 — 사전 설계 브레인스토밍 & Core Foundation 구현 계획 완료

### 완료
- 사전 설계 브레인스토밍 (Simulation-First + HAL 패턴 채택)
- 스펙 문서 작성: `docs/superpowers/specs/2026-04-07-pre-development-design.md`
- 구현 계획 작성: `docs/superpowers/plans/2026-04-07-core-foundation.md`

### 결정 사항
- **개발 전략**: Simulation-First + HAL 패턴 (하드웨어 없이 전체 파이프라인 개발)
- **Core Library 우선**: 플랫폼 무관하게 재사용 가능한 코어 구현 먼저
- **플랫폼**: 클라이언트·HW 담당자 논의 후 확정 (데스크톱/모바일/웹)
- **UI 프로토타입**: PySide6 사용 (플랫폼 확정 전 기능 검증용)
- **HDF5**: 데이터 저장 포맷으로 사전 채택

### 하드웨어 팀과 확정 필요 (미결)
- 광원 파장 2개, 샘플링 레이트, 채널-소스/디텍터 매핑, 통신 프로토콜+패킷 포맷, SDS(mm)

### 구현 계획 요약 (Task 0~12)
- Task 0: 개발 환경 세팅
- Task 1: 프로젝트 골격 & AppConfig
- Task 2: 핵심 데이터 모델 (RawPacket, ProcessedSample)
- Task 3: HAL 인터페이스 (FNIRSDevice ABC)
- Task 4: Thread-safe RingBuffer
- Task 5: FNIRSSimulator
- Task 6: 신호 필터 (bandpass, baseline)
- Task 7: 수정된 Beer-Lambert Law (mBLL)
- Task 8: ConcentrationIndex 플러그인
- Task 9: AcquisitionThread
- Task 10: ProcessingPipeline
- Task 11: SessionStore (HDF5)
- Task 12: PySide6 프로토타입 UI

---

## 2026-04-07 — 하드웨어 스펙 일부 확정

### 확정된 스펙
- **채널 수: 4채널** (기존 계획 8채널에서 변경)
- **측정 파장: 780nm, 850nm, 950nm (3파장)** (기존 계획 2파장에서 변경)

### 미확정 스펙 (계속 대기)
- 샘플링 레이트, 통신 프로토콜, 채널-소스/디텍터 매핑, SDS

### 코드/문서 반영
- `config/settings.yaml` — n_channels: 4, wavelengths_nm: [780, 850, 950], 흡광계수 3파장으로 업데이트
- `docs/superpowers/specs/2026-04-07-pre-development-design.md` — 미결 항목 업데이트
- `docs/superpowers/plans/2026-04-07-core-foundation.md` — 스펙 변경 공지 추가

### 후속 태스크 영향
- Task 2: RawPacket.channel_intensities 크기 = 4×3 = 12, n_wavelengths는 config에서 동적 계산
- Task 5: Simulator 3파장 신호 생성
- Task 7: mBLL 3×2 행렬 (pseudo-inverse로 과결정계 처리)
- Task 12: N_CHANNELS = 4

---

<!-- 아래에 날짜 순으로 로그 추가 -->

---

## 2026-04-08 — Core Foundation Task 7~12 구현 완료

### 완료된 태스크

| 태스크 | 내용 | 테스트 |
|--------|------|--------|
| Task 7 | mBLL (수정된 Beer-Lambert Law) — 3파장 pseudo-inverse (np.linalg.pinv), np.einsum 벡터화 | 18개 |
| Task 8 | ConcentrationIndex 플러그인 — ABC + SimpleHbOIndex | 7개 |
| Task 9 | AcquisitionThread — threading.Thread, daemon=True, try/finally 보장 | 4개 |
| Task 10 | ProcessingPipeline — 슬라이딩 윈도우, bandpass + mBLL + CI | 3개 |
| Task 11 | SessionStore — HDF5 저장/로드 (h5py), n_wavelengths 속성 저장 | 5개 |
| Task 12 | PySide6 MainWindow — 4채널 실시간 그래프, 집중도 미터, start/stop 공개 메서드 | 4개 |

**총 73개 테스트 — 전부 통과**

### 주요 버그 수정 및 결정 사항
- **PySide6 6.11.0 DLL 오류**: 6.7.3으로 다운그레이드 (requirements.txt 반영 필요)
- **bandpass order=6**: sosfiltfilt 이중 적용으로 실효 차수 12 (0.01-0.5 Hz fNIRS 대역)
- **ProcessingPipeline padlen 대응**: `filter_min=40` (order=6 + sosfiltfilt padlen=39 초과)
- **평균 오프셋 복원**: 필터 후 강도가 0 근처로 떨어져 mBLL NaN 발생 → 필터 전 평균 복원
- **mBLL 경고**: raw_intensity에 0 포함 시 UserWarning 발행
- **n_wavelengths 저장**: HDF5 /raw 그룹 속성으로 저장, 로드 시 복원

### 최종 코드 리뷰 — READY TO MERGE
다음 마일스톤에서 수정할 항목:
1. AcquisitionThread/ProcessingPipeline 오류 콜백 추가 (현재 daemon 스레드 오류 묵살)
2. main.py에서 `window._start_btn` 직접 접근 제거 → MainWindow.connect_start/stop() 또는 Signal로 교체
3. SessionStore `__enter__`/`__exit__` 컨텍스트 매니저 추가
4. main.py에 SessionStore 연동 (현재 실행 중 디스크 저장 미구현)

---

## 2026-04-09 — PySide6 GUI 개선 (UI 마일스톤)

### 완료
- **원클릭 실행**: `run.bat` 생성 — venv 자동 생성, 패키지 설치, PYTHONPATH 설정, 앱 실행
- **Cross-thread 버그 수정**: `Q_ARG(object, ...)` → `Signal(object)` 브리지 패턴으로 교체
- **실시간 그래프 개선**: 누적 표시 → 최근 5초 슬라이딩 윈도우, x축 상대 시간(초)
- **Y축 고정 스케일**: -5 ~ +5 μmol/L 고정
- **GUI 전면 재설계** (다크 테마 + 탭 구조):
  - 탭 1: Calibration — SNR 3파장 바 차트, 채널 상태 판정, "진행 ▶" 버튼
  - 탭 2: 3D Brain Map — RBF 보간, Blue↔White↔Red 컬러맵, HbO/HbR 토글
  - 탭 3: Time Series — 채널별 실시간 그래프
  - 실행 시 Calibration 탭 기본, 완료 후 나머지 탭 활성화
- **3D Brain Map 배경**: nilearn 대신 matplotlib+PIL로 실제 뇌 형태 이미지 생성
  - `src/ui/assets/generate_brain.py` — 다층 노이즈 기반 회백질 질감, 7종 해부학적 고랑
  - `src/ui/assets/brain_top.png` — 600×600 RGBA PNG

### 주요 결정 사항
- PySide6 버전: 6.7.0 유지 (6.11.0 DLL 로드 오류)
- `matplotlib`, `Pillow` requirements.txt에 추가
- 총 테스트: **78개 전부 통과**

---

## 2026-04-10 — 플랫폼 전환 결정: 웹 서비스

### 배경
고객 요청: 플랫폼(기기 종류) 무관하게 웹 브라우저로 접속 가능한 서비스

### 브레인스토밍 주요 결정 사항

| 항목 | 결정 |
|------|------|
| 플랫폼 | 웹 서비스 (브라우저 기반) |
| 하드웨어 연결 | BLE 고정 |
| 신호처리 위치 | JavaScript 재구현 (Web Worker) |
| 프론트엔드 | React + Vite |
| 빌드 도구 | Vite |
| 데이터 저장 | 클라이언트 전용 (CSV 다운로드) |
| 배포 | 로컬 개발 우선, 이후 정적 호스팅 |
| 프로토타입 타겟 | Chrome/Edge (Web Bluetooth API) |

### 기술적 제약 및 결정
- **Web Bluetooth API**: Chrome/Edge만 지원, Firefox/Safari 미지원
- **iOS Safari 미지원**: Bluefy 앱으로 우회 (단기), React Native 전환 (장기 고려)
- **순수 웹 + BLE 모순**: 브라우저 직접 BLE 연결은 Chromium 계열만 가능
  → 프로토타입은 Chrome 타겟으로 먼저 완성, 고객 데모 후 협의
- **Python 코드 자산**: 신호처리 로직(mbll.py, filters.py, pipeline.py)을 JS로 재구현

### PySide6 Qt 앱 상태
- Core 라이브러리(신호처리, HDF5 저장)는 웹 전환 후에도 참조용으로 유지
- UI 레이어(src/ui/)는 웹 버전으로 대체 예정

### 다음 단계
- [ ] 웹 서비스 스펙 문서 작성 (brainstorming 완료 후)
- [ ] Excalidraw로 화면 구성 와이어프레임
- [ ] v0.dev로 React 컴포넌트 초안 생성
- [ ] React 프로젝트 구조 생성 (web/ 디렉토리)
