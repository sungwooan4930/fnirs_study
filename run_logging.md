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
- [x] 웹 서비스 스펙 문서 작성 → `docs/superpowers/specs/2026-04-10-web-service-design.md`
- [x] 구현 계획 작성 → `docs/superpowers/plans/2026-04-10-web-service.md` (Task 0~13)
- [x] React+Vite 프로젝트 구조 생성 (web/ 디렉토리)

---

## 2026-04-10 — 웹 서비스 프로토타입 완성 (Task 0~13)

### 완료 항목

| Task | 내용 | 주요 파일 |
|------|------|-----------|
| 0 | Vite+React 초기화, 다크 테마, 3탭 구조 | `web/` 전체 |
| 1 | AppContext — 전역 상태 (bleStatus, sessionData 등) | `src/context/AppContext.jsx` |
| 2 | lib/mbll.js — Modified Beer-Lambert Law JS 포팅 | `src/lib/mbll.js` |
| 3 | lib/filters.js — Butterworth 밴드패스 (fili 라이브러리) | `src/lib/filters.js` |
| 4 | lib/rbf.js — RBF 보간 + BWR 컬러맵 | `src/lib/rbf.js` |
| 5 | lib/simulator.js — Python FNIRSSimulator JS 포팅 | `src/lib/simulator.js` |
| 6 | pipeline.worker.js + usePipeline 훅 — Web Worker 신호처리 | `src/workers/`, `src/hooks/usePipeline.js` |
| 7 | useBLE.js — Web Bluetooth API + 시뮬레이터 모드 | `src/hooks/useBLE.js` |
| 8 | TopBar — BLE 연결 버튼, 상태 표시, CI 게이지 | `src/components/TopBar.jsx` |
| 9 | CalibrationTab — SNR 바 차트, 채널 상태, 진행 버튼 | `src/components/CalibrationTab.jsx` |
| 10 | TimeSeriesTab — Canvas 실시간 HbO/HbR 그래프 | `src/components/TimeSeriesTab.jsx` |
| 11 | BrainMapTab — RBF 보간 BWR Canvas 렌더링 | `src/components/BrainMapTab.jsx` |
| 12 | CSV 내보내기 + sessionStorage 임시 저장 | TopBar, AppContext |
| 13 | 최종 단위 테스트 — 25 tests passed (4 test files) | `src/__tests__/` |

### 주요 기술 결정 사항
- `iir-filter` npm 패키지 미존재 → `fili` v2.0.3으로 대체
- `fili` BW 파라미터: Hz가 아닌 옥타브 단위 (`BW = Math.log2(highHz / lowHz)`)
- Web Worker는 ES module 방식으로 생성 (`{ type: 'module' }`)
- sessionStorage: 새로고침 후에도 세션 데이터 복원 가능

### 개발 실행
```bash
cd web && npm run dev
# http://localhost:5173/?simulate=true 접속 → 시뮬레이터 모드
```

### 다음 단계
- [ ] 실제 BLE 하드웨어 UUID 확정 후 useBLE.js에 characteristic 구독 추가
- [ ] 고객 데모 후 iOS 지원 방식 협의 (Bluefy vs React Native)
- [ ] 정적 호스팅 배포 환경 선택 (Netlify / Vercel / self-hosted)

---

## 2026-04-13 — UI 전면 재설계 + Report 탭 추가

### 배경
고객 요청: Dark+Lovable Blue 디자인 시스템 적용, 실제 3D 뇌 이미지 활용, 채널별 플롯 분리, 세션 종료 레포트 추가

### 완료 항목

| 항목 | 내용 |
|------|------|
| 디자인 토큰 | `App.css` :root 전면 교체 — `#0f1117` 배경, `#1f55f1` 액센트 (Lovable Blue) |
| 폰트 | Inter Variable (로컬 woff2, `web/public/fonts/InterVariable.woff2`) |
| 탭/버튼 | pill shape (`border-radius: 20px`) 전체 적용 |
| BrainMapTab | Canvas 방식 → `<img>`+SVG 오버레이 방식으로 교체. `refer/3d brain.png` → `web/public/brain.png`. 채널 마커 제거. 채널별 radialGradient 블롭(BWR 색상) |
| TimeSeriesTab | 단일 캔버스 → 2×2 채널별 ChannelCanvas 그리드. **Auto-scale** 적용 (채널별 min/max 동적 계산, 12% 패딩, Y축 레이블 표시) |
| ReportTab | 신규 생성. 마운트 시점 스냅샷 고정(실시간 업데이트 없음). conic-gradient 점수 원, SVG CI 시계열, 뇌 맵 스냅샷, **뇌 부위명** 표기 (Ch번호 아님) |

### 채널 → 뇌 부위 매핑
| 채널 | 부위 |
|------|------|
| Ch1 | 좌전방 PFC |
| Ch2 | 우전방 PFC |
| Ch3 | 좌후방 PFC |
| Ch4 | 우후방 PFC |

### 주요 기술 결정 사항
- BrainMapTab: canvas 기반 RBF 보간 제거 → SVG radialGradient 오버레이로 단순화 (의존성 없음)
- ReportTab 스냅샷: `const [snapshot] = useState(() => sessionData)` — 마운트 1회만 캡처
- TimeSeriesTab auto-scale: 매 RAF 프레임마다 `calcRange()` 호출, 0선은 범위 내에 있을 때만 표시

### 테스트
- 25 tests passed (4 test files) — 변경 후에도 전부 통과

### 커밋
- `feat(web): UI 전면 재설계 — Dark+Lovable Blue, Inter 폰트, Brain Map SVG 오버레이, TS auto-scale, Report 탭`

---

## 2026-04-13 — 프로필 입력 단계 추가

### 완료 항목

| 항목 | 내용 |
|------|------|
| ProfileStep 컴포넌트 | Calibration 전 이름·나이 입력 카드. 유효성 검사 통과 시 "시작하기 ▶" 활성화 |
| AppContext | `userProfile` 상태(`{ name, age }`) 추가, context value에 노출 |
| App.jsx | `TabContainer`에서 `userProfile` 없을 때 `ProfileStep` 렌더. **Rules of Hooks** 준수: `useState` 선언을 조건부 return 앞에 배치 |
| ReportTab | 세션 요약 info-grid에 이름·나이·측정날짜·측정시간 4열 표시 |

### 주요 기술 결정 사항
- Rules of Hooks 위반 디버깅: 조건부 return 뒤에 `useState` 배치 시 React 런타임 에러 발생 → `useState` 선언을 컴포넌트 최상단으로 이동

### 테스트
- 25 tests passed (4 test files) — 변경 후에도 전부 통과

### 커밋
- `feat(web): 프로필 입력 단계 추가 (이름·나이) — Report에 반영`

---

## 2026-08-18 — 프로젝트 전면 재정의 & 하네스 엔지니어링

### 배경
연구계획서 원본(`docs/plan/…2차수정본.pdf`, 11쪽)을 처음으로 코드베이스에 반영.
기존 구현이 계획서와 **다른 프로젝트**임이 확인되어 전면 재설계 결정.

| 항목 | 기존 구현 (폐기) | 연구계획서 |
|------|-----------------|-----------|
| 정체성 | 4ch 커스텀 fNIRS 집중도 측정기(제품) | 멀티모달 AI 인지상태 추정 연구과제 |
| fNIRS | 4ch, 3파장(780/850/950), BLE | 48ch, 전두·두정·운동·후두, 10.4 Hz |
| EEG | 없음 | 30ch, 10-5, 1,000 Hz |
| 동기화 / 표준 | 자체 타임스탬프 / 자체 HDF5 | LSL / BIDS |
| 출력 | 집중도 1차원 | 6차원 인지상태 벡터 |
| 모델 | SimpleHbOIndex(규칙식) | 융합 3전략 비교 + 크로스모달 어텐션 CNN-LSTM, LOSO-CV |
| XAI / 교수전략 | 없음 | SHAP·어텐션 / 규칙-데이터 하이브리드 추론 |
| 대시보드 사용자 | 측정 대상자 본인 | 교수자(다수 학습자 동시 모니터링) |

### 완료 — 하네스 구축
- **`CLAUDE.md` 신규 작성** — 계획서를 코딩 규약으로 번역한 최상위 프로젝트 문서.
  도메인 사전(6차원 벡터·4계층 라벨·모달 스펙), 파이프라인 규약(전처리·윈도우·융합 3전략·검증·XAI·교수전략 규칙),
  연구코드 절대규칙(누수 방지·재현성·개인정보·보고 정직성), 목표 디렉토리 구조, 미확정 6건, 절대금지 9건.
- **`.claude/skills/leakage-check`** — 데이터 누수 7항목 점검 스킬.
  5초창·1초스텝 = 80% 오버랩이라 누수 시 정확도가 허구로 치솟는 구조적 위험에 대응.
- **`.claude/skills/research-log`** — 본 로그 기록 규약. 실험 기록 시 CV 방식·chance level·최악 피험자 성능 필수화.
- **`.claude/settings.json`** — 안전 명령 allowlist + `data/raw`·원시신호 파일 read/write deny.
- **`.gitignore` 강화** — `.snirf/.edf/.fif/.eeg` 등 뇌신호 포맷, `results/`, 식별정보 패턴 차단.
- **메모리 교정** — 기존 메모리가 "4채널 fNIRS 집중도 측정기"를 기술해 매 세션 오염 중이었음. 전면 교체 + 계획서 참조 메모리 신규 추가.
- **문서 재배치** — 구 `PROJECT_GUIDELINES.md`·`docs/superpowers/` → `docs/legacy/`. 신규 `docs/{plan,specs,plans,protocol}/` 생성.

### 결정 사항
| 항목 | 결정 | 근거 | 대안 (기각 사유) |
|------|------|------|-----------------|
| 계획서 권위 | 계획서 > CLAUDE.md > 코드 | 연구과제는 계획서가 계약 | 코드 우선 (계획서 이탈 위험) |
| 구 코드 | `docs/legacy/`로 문서 아카이브, 코드는 처리 방침 미정 | 이력 보존 | 즉시 삭제 (되돌릴 수 없음) |
| Read deny 범위 | `data/raw/**` + 신호파일 확장자만 | BIDS sidecar·participants.tsv 조회는 정당 | `data/**` 전면 (메타데이터 작업 불가) |

### 계획서 연계
전 목표(1~5) 공통 기반. 아직 목표 1(지표 정의·프로토콜) 착수 전.

### 미해결 (CLAUDE.md §9와 동기화)
- [ ] **참가자 수 계획서 내부 불일치** — 요약·목표2는 100명, 추진전략§3·예산(30,000원×30명)은 30명
- [ ] 실제 보유 장비 모델명 (EEG 30ch / fNIRS 48ch) — LSL 드라이버·몽타주 결정에 필요
- [ ] 대시보드 스택 (React+FastAPI vs Streamlit/Dash)
- [ ] fNIRS 48ch 몽타주 (소스-디텍터 배치, SDS, 파장)
- [ ] "3수준 분류"의 정의 (인지부하 저/중/고인지, 6차원 각각인지)
- [ ] 구 4ch 코드 처리 방침 (`legacy_4ch/` 아카이브 vs 삭제)
- [ ] `.claude/launch.json`이 폐기된 web/·Qt 앱을 참조 중 — 대시보드 스택 확정 후 갱신

### 추가 결정 (사용자 확인)
| 항목 | 결정 | 영향 |
|------|------|------|
| 착수 지점 | **목표 3 — 전처리 + 모델 파이프라인** | 데이터 수집(목표1·2)을 기다리지 않고 병렬 진행 |
| 하드웨어 | **시뮬레이션 우선(Simulation-First)** | 합성 신호 + LSL 모킹으로 전 파이프라인 개발, 장비 확보 후 드라이버만 교체(HAL) |
| 구 4ch 코드 | `legacy_4ch/`로 아카이브 | `src/ web/ tests/ config/ refer/ run.bat requirements*.txt lovable-*.md` 이동 |
| 참가자 수 | **미확정 유지** | 하드코딩 금지. `config`의 `n_subjects`로만 취급, 합성 생성기는 임의 N 수용 |

### 아카이브 실행 내역
- `git mv`로 이력 보존하며 `legacy_4ch/` 이동. `legacy_4ch/README.md`에 폐기 사유·재사용 가능 범위(mBLL/필터 로직 참조만)·금지사항 명시.
- 루트 PDF 중복본은 `docs/plan/` 사본과 SHA256 동일 확인 후 삭제.
- `requirements.txt` 신규 작성 — MNE / MNE-NIRS / mne-bids / pyprep / pylsl / scikit-learn / XGBoost / PyTorch / SHAP.
- `docs/superpowers/` → `docs/legacy/superpowers_4ch/`. 신규 `docs/{plan,specs,plans,protocol}/` 생성.

### 다음 단계
목표 3 착수 — 개발 프로세스(§6)에 따라 `superpowers:brainstorming`부터 시작.

---

## 2026-08-18 — 목표 3 브레인스토밍 & 테스트베드 설계 스펙 완료

### 완료
- **목표 3 범위 분해** — 단일 스펙 범위를 넘어(M5–8, 보조원 5명) 5개 서브시스템으로 분해:
  A 합성 생성기 / B 전처리 / C 특징·데이터셋 / D 평가 프레임워크 / E 융합 3전략
- **첫 서브프로젝트 A+D 설계 완료** → `docs/specs/2026-08-18-simulation-testbed-design.md` (커밋 `fdf06c3`)
- 스펙 자체 검토 5건 수정 (C 범위 정정, effect_size 의미 명확화, dirty 트리 기록, T4 판정기준 계량화, 가드 비활성화 통제)

### 결정 사항
| 항목 | 결정 | 근거 | 대안 (기각 사유) |
|------|------|------|-----------------|
| 첫 서브프로젝트 | **A+D 묶음** | 합성 데이터는 효과크기를 우리가 심으므로 정답을 앎 → 누수 가드 작동을 증명할 수 있는 유일한 시점 | A만 (검증 수단 없음) / B·E 먼저 (오류 원인 분리 불가) |
| 신호 현실성 | 수준 2 **구조** + 단계적 구현 | 생성기를 두 번 만들지 않으면서 범위 억제 | 수준 1 (B에서 재설계) / 수준 2 전체 (범위 과대) |
| 행동 2차원 | **선행 예측 분리** (t 뇌신호 → t+Δ 행동) | 계획서 가설 2(오류 1.2초 전 선행 신호)를 구현하며 라벨 누수 회피 | 동시점 예측 (자명한 누수) / 4차원만 (계획서와 불일치) |
| 누수 방지 | **계약 강제형** — `iter_folds()`로만 접근, `TestView.fit()` 예외 | 5명×1년×수백 실험을 사람의 규율에 맡길 수 없음. 누수는 성능이 *올라가서* 조용히 실패 | 관례 기반(뚫림) / 런타임 감지(사후) |
| 실험 추적 | 자체 경량 러너 | 외부 전송 없음(IRB), 의존성 최소 | MLflow(학습·유지비) / Hydra(곡선) |

### 승인 기준 (스펙 §8)
- **T1 널 테스트** — `effect_size=0` → chance(33.3%) 포함. 파이프라인 전역 누수 검출, 가장 강력
- **T2 효과 회복** — 심은 효과를 하네스가 회수
- **T3 누수 검출** — 일부러 누수 시 +15%p 부풀려지고 가드가 차단. **이 서브프로젝트의 존재 이유**
- **T4 개인차 스윕** — τ² 증가 → LOSO 단조 하락, within-subject는 유지

### 계획서 연계
목표 3 착수. 아직 구현 코드 0줄 — 설계 단계까지만 완료.

### 미해결
- [ ] **사용자 스펙 리뷰 대기 중** ← 세션 재개 시 여기부터
- [ ] 승인 후 `superpowers:writing-plans`로 구현 계획 작성
- [ ] "3수준 분류" 정의 확인 (스펙은 인지부하 3수준=n-back 0/2/3으로 가정)
- [ ] 참가자 수 계획서 불일치 (100 vs 30) — 연구책임자 확인
- [ ] T2 정확도 범위 — 파일럿 관측 후 회귀 기준 고정
