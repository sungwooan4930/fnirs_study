# fNIRS 웹 서비스 전환 설계 스펙

**날짜**: 2026-04-10  
**상태**: 승인됨  
**대상**: PySide6 데스크톱 앱 → React 웹 서비스 전환

---

## 배경 및 목표

고객 요청: 기기 종류(휴대폰·태블릿·랩탑·데스크톱) 무관하게 웹 브라우저로 접속 가능한 fNIRS 모니터링 서비스.

**프로토타입 목표**: Chrome/Edge 타겟으로 동작하는 웹 앱을 먼저 완성 → 고객 데모 → iOS/Firefox 지원 범위 협의.

---

## 핵심 제약 사항

| 제약 | 내용 |
|------|------|
| 하드웨어 연결 | BLE 고정 (변경 불가) |
| Web Bluetooth 지원 | Chrome, Edge만 지원. Firefox·Safari 미지원 |
| iOS 대응 | Bluefy 앱 설치로 우회 (단기), React Native 전환 (장기 고려) |
| 서버 없음 | 순수 정적 웹 앱, 백엔드 서버 없음 |
| 데이터 저장 | 클라이언트 전용 (CSV 다운로드) |

---

## 섹션 1: 전체 구조

### 기술 스택

| 분류 | 선택 | 이유 |
|------|------|------|
| UI 프레임워크 | React | 실시간 상태 자동 갱신, 컴포넌트 재사용 |
| 빌드 도구 | Vite | 빠른 개발 서버, React 표준 |
| 언어 | JavaScript (ES Modules) | 프레임워크 없는 lib/ 레이어 |
| 테스트 | Vitest + React Testing Library | Vite 네이티브 통합 |

### 디렉토리 구조

```
web/
├── index.html
├── package.json
├── vite.config.js
└── src/
    ├── main.jsx                   — 앱 진입점
    ├── App.jsx                    — 탭 라우팅, AppContext 제공
    ├── context/
    │   └── AppContext.jsx         — 전역 상태 (BLE, 세션, 샘플)
    ├── hooks/
    │   ├── useBLE.js              — Web Bluetooth 연결/수신
    │   └── usePipeline.js         — Web Worker 연결 브리지
    ├── workers/
    │   └── pipeline.worker.js     — 신호처리 (UI 스레드 분리)
    ├── components/
    │   ├── TopBar.jsx             — BLE 버튼, 연결 상태, CI 게이지
    │   ├── CalibrationTab.jsx     — SNR 바 차트, 채널 상태, 진행 버튼
    │   ├── BrainMapTab.jsx        — 뇌 이미지 + RBF Canvas, HbO/HbR 토글
    │   └── TimeSeriesTab.jsx      — 채널별 실시간 그래프 Canvas
    └── lib/
        ├── mbll.js                — mBLL (Python mbll.py 포팅)
        ├── filters.js             — Butterworth 필터 (Python filters.py 포팅)
        ├── rbf.js                 — RBF 보간 (Python brain_map_widget.py 포팅)
        └── simulator.js           — 개발용 BLE 시뮬레이터
```

### 데이터 흐름 (전체)

```
fNIRS BLE
    ↓ (Web Bluetooth API)
useBLE.js  →  rawPacket
    ↓ (postMessage)
pipeline.worker.js
    ├── lib/mbll.js        — HbO, HbR 계산
    ├── lib/filters.js     — Butterworth 밴드패스
    └── lib/rbf.js         — 공간 보간 준비
    ↓ (postMessage back)
usePipeline.js  →  processedSample
    ↓ (React Context)
AppContext.processedSample
    ├── BrainMapTab   →  Canvas 렌더링
    └── TimeSeriesTab →  Canvas 렌더링
```

---

## 섹션 2: BLE 연결 + 신호처리

### BLE 연결 흐름

```javascript
// useBLE.js 핵심 흐름
navigator.bluetooth.requestDevice({ filters: [...] })
  → device.gatt.connect()
  → service.getCharacteristic(DATA_UUID)
  → characteristic.startNotifications()
  → characteristic.addEventListener('characteristicvaluechanged', onData)
```

- 연결 버튼 클릭 시 브라우저 BLE 기기 선택 팝업 표시
- 연결 끊김 시 자동 재연결 시도 (3회)
- 연결 상태: `idle | connecting | connected | error`

### Web Worker 신호처리

100Hz 데이터를 메인 스레드에서 직접 처리하면 UI가 버벅임. Worker로 분리.

```
메인 스레드                         Worker 스레드
─────────────────                  ──────────────────────
postMessage(rawPacket)  →→→→→→→   슬라이딩 윈도우 관리
                                   lib/filters.js (Butterworth)
                                   lib/mbll.js (mBLL)
postMessage(processedSample) ←←←  → HbO, HbR, CI
React 상태 갱신
Canvas 렌더링
```

### Python → JS 포팅 대상

| Python 파일 | JS 파일 | 핵심 로직 |
|-------------|---------|-----------|
| `mbll.py` | `lib/mbll.js` | pseudo-inverse 행렬 연산 |
| `filters.py` | `lib/filters.js` | Butterworth sos 필터 (`iir-filter` 패키지) |
| `pipeline.py` | `workers/pipeline.worker.js` | 슬라이딩 윈도우, 파이프라인 오케스트레이션 |
| `brain_map_widget._BrainCanvas._build_overlay()` | `lib/rbf.js` | RBF 보간, BWR 컬러맵 |

### iOS 대응

- 단기: 사용자 가이드에 Bluefy 앱 설치 안내 명시
- 장기: 사용자 수 확인 후 React Native 전환 여부 결정

---

## 섹션 3: UI 컴포넌트 구조

### 탭 구조 (Qt 앱과 1:1 대응)

| Qt 위젯 | React 컴포넌트 | 비고 |
|---------|--------------|------|
| `CalibrationWidget` | `CalibrationTab.jsx` | SNR 바 차트, 채널 상태 판정 |
| `BrainMapWidget` | `BrainMapTab.jsx` | 뇌 이미지 배경 + RBF Canvas |
| `TimeSeriesWidget` | `TimeSeriesTab.jsx` | 5초 슬라이딩 윈도우 |
| 상단 바 | `TopBar.jsx` | BLE 버튼, CI 게이지 |

### 전역 상태 (AppContext)

```javascript
{
  bleStatus: 'idle' | 'connecting' | 'connected' | 'error',
  calibrationDone: false,           // true 시 BrainMap/TimeSeries 탭 활성화
  processedSample: {                // 최신 1개 샘플
    timestamp, hbo, hbr, ci
  },
  sessionData: [...],              // 전체 시계열 (CSV 내보내기용)
}
```

### 반응형 레이아웃

- 데스크톱 (>1024px): 탭 상단, 콘텐츠 좌우 여백
- 태블릿 (768–1024px): 탭 상단, Canvas 전체 너비
- 모바일 (<768px): 탭 하단 고정 네비게이션, Canvas 전체 너비

### UI 상세 디자인

Excalidraw로 와이어프레임 작성 후 v0.dev로 React 컴포넌트 초안 생성. 스펙 범위 외.

---

## 섹션 4: 데이터 저장 & 내보내기

### 세션 중 (메모리)

```javascript
// AppContext.sessionData 구조
[
  {
    timestamp: 0.000,
    hbo: [ch1, ch2, ch3, ch4],   // μmol/L
    hbr: [ch1, ch2, ch3, ch4],   // μmol/L
    ci: 0.72                      // [0.0, 1.0]
  },
  // 100Hz × 측정 시간만큼 누적
]
```

### CSV 내보내기

측정 종료 후 "다운로드" 버튼 클릭 시:

```
timestamp,hbo_ch1,hbo_ch2,hbo_ch3,hbo_ch4,hbr_ch1,hbr_ch2,hbr_ch3,hbr_ch4,ci
0.000,0.12,-0.05,0.08,-0.03,-0.04,0.02,-0.03,0.01,0.72
0.010,...
```

파일명: `fnirs_session_YYYYMMDD_HHMMSS.csv`

### 메모리 상한

100Hz × 1시간 = 36만 샘플 ≈ 25MB. 브라우저 허용 범위 내. 별도 상한 처리 불필요.

### 브라우저 임시 저장

`sessionStorage` 활용 — 새로고침 시 세션 데이터 보존, 탭 닫으면 삭제.

---

## 섹션 5: 개발 환경 & 테스트

### 개발 시작

```bash
cd web
npm install
npm run dev      # localhost:5173
```

### 시뮬레이터 모드

BLE 기기 없이 개발 가능. `?simulate=true` 쿼리 파라미터로 활성화.

```javascript
// useBLE.js
if (simulate) return FNIRSSimulator.start(onData)
```

`lib/simulator.js`는 Python `FNIRSSimulator`와 동일한 신호 패턴 생성.

### 테스트 전략

| 대상 | 방법 | 기준 |
|------|------|------|
| `lib/mbll.js` | Vitest 단위 테스트 | Python 결과와 오차 < 1e-6 |
| `lib/filters.js` | Vitest 단위 테스트 | 주파수 응답 검증 |
| `lib/rbf.js` | Vitest 단위 테스트 | 보간 수치 검증 |
| React 컴포넌트 | React Testing Library | 탭 전환, 버튼 동작 |
| BLE 연결 | 수동 테스트 | 실제 기기 또는 BLE 시뮬레이터 |
| 실시간 성능 | 수동 측정 | 100Hz 입력 시 60fps 유지 |

---

## 미결 사항

- [ ] UI 상세 디자인 (Excalidraw 와이어프레임)
- [ ] BLE GATT 서비스/특성 UUID (하드웨어 팀 확인 필요)
- [ ] 배포 방식 확정 (GitHub Pages vs 자체 서버)
- [ ] iOS 지원 범위 고객 협의 (프로토타입 데모 후)
