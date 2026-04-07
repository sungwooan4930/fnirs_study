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

<!-- 아래에 날짜 순으로 로그 추가 -->
