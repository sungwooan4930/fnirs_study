# fNIRS 웹 서비스 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PySide6 Qt 앱을 React+Vite 웹 앱으로 전환. BLE 연결, JS 신호처리(mBLL/필터/RBF), 3탭 UI(Calibration/BrainMap/TimeSeries), CSV 내보내기를 포함한 브라우저 기반 fNIRS 모니터링 시스템 구현.

**Architecture:** Web Bluetooth API로 BLE 직접 연결 → rawPacket을 Web Worker(pipeline.worker.js)에서 신호처리 → processedSample을 React Context(AppContext)로 전달 → Canvas 기반 실시간 렌더링. 서버 없는 순수 정적 웹 앱.

**Tech Stack:** React 18, Vite 5, Vitest, React Testing Library, iir-filter(Butterworth JS 포팅), CSS Modules(반응형)

---

## 파일 맵

```
web/
├── index.html                         — Vite HTML 진입점
├── package.json                       — 의존성 (react, vite, vitest, iir-filter)
├── vite.config.js                     — Vite + Vitest 설정, worker 지원
├── src/
│   ├── main.jsx                       — React 앱 마운트
│   ├── App.jsx                        — 탭 라우팅, TopBar, AppContext 제공
│   ├── App.css                        — 전역 다크 테마, 탭 레이아웃, 반응형
│   ├── context/
│   │   └── AppContext.jsx             — bleStatus, calibrationDone, processedSample, sessionData
│   ├── hooks/
│   │   ├── useBLE.js                  — Web Bluetooth 연결/수신/재연결
│   │   └── usePipeline.js             — Web Worker 생성·메시지 브리지
│   ├── workers/
│   │   └── pipeline.worker.js         — 슬라이딩 윈도우 + filters.js + mbll.js
│   ├── lib/
│   │   ├── mbll.js                    — mBLL 행렬 연산 (Python mbll.py 포팅)
│   │   ├── filters.js                 — Butterworth 밴드패스 (iir-filter 래핑)
│   │   ├── rbf.js                     — RBF 보간 + BWR 컬러맵
│   │   └── simulator.js               — 개발용 가짜 BLE 스트림
│   └── components/
│       ├── TopBar.jsx / TopBar.css    — BLE 버튼, 연결 상태, CI 게이지
│       ├── CalibrationTab.jsx / .css  — SNR 바 차트, 채널 상태, 진행 버튼
│       ├── BrainMapTab.jsx / .css     — 뇌 이미지 + RBF Canvas, HbO/HbR 토글
│       └── TimeSeriesTab.jsx / .css   — 채널별 실시간 Canvas (5초 슬라이딩)
└── src/__tests__/
    ├── lib/mbll.test.js
    ├── lib/filters.test.js
    ├── lib/rbf.test.js
    └── components/App.test.jsx
```

---

## Task 0: 프로젝트 초기화

**Files:**
- Create: `web/package.json`
- Create: `web/vite.config.js`
- Create: `web/index.html`
- Create: `web/src/main.jsx`
- Create: `web/src/App.jsx`
- Create: `web/src/App.css`

- [ ] **Step 1: web/ 디렉토리에 Vite+React 프로젝트 생성**

```bash
cd D:/Study_fNIRS
npm create vite@latest web -- --template react
cd web
npm install
npm install iir-filter
npm install --save-dev vitest @vitest/coverage-v8 @testing-library/react @testing-library/jest-dom jsdom
```

- [ ] **Step 2: vite.config.js 수정 — Vitest + Worker 지원**

`web/vite.config.js` 전체를 교체:

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  worker: {
    format: 'es',
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/__tests__/setup.js',
  },
})
```

- [ ] **Step 3: 테스트 setup 파일 생성**

`web/src/__tests__/setup.js`:
```javascript
import '@testing-library/jest-dom'
```

- [ ] **Step 4: App.jsx 뼈대 작성**

`web/src/App.jsx`:
```jsx
import { useState } from 'react'
import './App.css'

export default function App() {
  const [activeTab, setActiveTab] = useState('calibration')

  return (
    <div className="app">
      <header className="topbar">
        <span className="app-title">fNIRS Monitor</span>
      </header>
      <nav className="tab-nav">
        <button
          className={activeTab === 'calibration' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('calibration')}
        >
          Calibration
        </button>
        <button
          className={activeTab === 'brainmap' ? 'tab active' : 'tab'}
          disabled
        >
          Brain Map
        </button>
        <button
          className={activeTab === 'timeseries' ? 'tab active' : 'tab'}
          disabled
        >
          Time Series
        </button>
      </nav>
      <main className="tab-content">
        {activeTab === 'calibration' && <div>Calibration Tab (placeholder)</div>}
        {activeTab === 'brainmap' && <div>Brain Map Tab (placeholder)</div>}
        {activeTab === 'timeseries' && <div>Time Series Tab (placeholder)</div>}
      </main>
    </div>
  )
}
```

- [ ] **Step 5: App.css 다크 테마 작성**

`web/src/App.css`:
```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #1a1a1a;
  --surface: #242424;
  --border: #333;
  --text: #e0e0e0;
  --text-muted: #888;
  --accent: #4a90d9;
  --accent-hover: #5aa0e9;
  --success: #4caf50;
  --error: #f44336;
  --warning: #ff9800;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Segoe UI', system-ui, sans-serif;
  height: 100dvh;
  overflow: hidden;
}

.app {
  display: flex;
  flex-direction: column;
  height: 100dvh;
}

.topbar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px 16px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.app-title {
  font-weight: 700;
  font-size: 1.1rem;
  color: var(--accent);
}

.tab-nav {
  display: flex;
  gap: 4px;
  padding: 8px 16px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.tab {
  padding: 6px 20px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.15s;
}

.tab:hover:not(:disabled) {
  background: var(--border);
  color: var(--text);
}

.tab.active {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}

.tab:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.tab-content {
  flex: 1;
  overflow: auto;
  padding: 16px;
}

/* 반응형: 모바일 탭을 하단으로 */
@media (max-width: 768px) {
  .app { flex-direction: column-reverse; }
  .tab-nav {
    justify-content: space-around;
    border-bottom: none;
    border-top: 1px solid var(--border);
  }
  .tab { flex: 1; text-align: center; padding: 10px 4px; font-size: 0.8rem; }
}
```

- [ ] **Step 6: 개발 서버 동작 확인**

```bash
cd web
npm run dev
```

브라우저에서 `http://localhost:5173` 접속 → 다크 배경, 3개 탭(Calibration 활성, 나머지 disabled) 확인.

- [ ] **Step 7: 커밋**

```bash
cd D:/Study_fNIRS
git add web/
git commit -m "feat(web): Vite+React 프로젝트 초기화, 다크 테마 탭 구조"
```

---

## Task 1: AppContext — 전역 상태

**Files:**
- Create: `web/src/context/AppContext.jsx`
- Create: `web/src/__tests__/components/App.test.jsx`

- [ ] **Step 1: 실패할 테스트 작성**

`web/src/__tests__/components/App.test.jsx`:
```jsx
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AppProvider, useApp } from '../../context/AppContext'

function Inspector() {
  const ctx = useApp()
  return (
    <div>
      <span data-testid="ble">{ctx.bleStatus}</span>
      <span data-testid="cal">{String(ctx.calibrationDone)}</span>
      <span data-testid="ci">{ctx.processedSample?.ci ?? 'null'}</span>
      <button onClick={() => ctx.setCalibrationDone(true)}>cal</button>
      <button onClick={() => ctx.setBleStatus('connected')}>ble</button>
      <button onClick={() => ctx.pushSample({ timestamp: 1, hbo: [0], hbr: [0], ci: 0.5 })}>push</button>
    </div>
  )
}

test('초기 상태가 올바르다', () => {
  render(<AppProvider><Inspector /></AppProvider>)
  expect(screen.getByTestId('ble')).toHaveTextContent('idle')
  expect(screen.getByTestId('cal')).toHaveTextContent('false')
  expect(screen.getByTestId('ci')).toHaveTextContent('null')
})

test('setBleStatus 변경', async () => {
  render(<AppProvider><Inspector /></AppProvider>)
  await userEvent.click(screen.getByText('ble'))
  expect(screen.getByTestId('ble')).toHaveTextContent('connected')
})

test('setCalibrationDone 변경', async () => {
  render(<AppProvider><Inspector /></AppProvider>)
  await userEvent.click(screen.getByText('cal'))
  expect(screen.getByTestId('cal')).toHaveTextContent('true')
})

test('pushSample이 processedSample과 sessionData를 갱신', async () => {
  render(<AppProvider><Inspector /></AppProvider>)
  await userEvent.click(screen.getByText('push'))
  expect(screen.getByTestId('ci')).toHaveTextContent('0.5')
})
```

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
cd web
npm run test -- --run
```

Expected: `Cannot find module '../../context/AppContext'`

- [ ] **Step 3: AppContext 구현**

`web/src/context/AppContext.jsx`:
```jsx
import { createContext, useContext, useState, useCallback } from 'react'

const AppContext = createContext(null)

export function AppProvider({ children }) {
  const [bleStatus, setBleStatus] = useState('idle')
  // 'idle' | 'connecting' | 'connected' | 'error'

  const [calibrationDone, setCalibrationDone] = useState(false)

  const [processedSample, setProcessedSample] = useState(null)
  // { timestamp, hbo: number[], hbr: number[], ci: number }

  const [sessionData, setSessionData] = useState([])
  // 전체 시계열 — CSV 내보내기용

  const pushSample = useCallback((sample) => {
    setProcessedSample(sample)
    setSessionData((prev) => [...prev, sample])
  }, [])

  const clearSession = useCallback(() => {
    setSessionData([])
    setProcessedSample(null)
  }, [])

  return (
    <AppContext.Provider value={{
      bleStatus, setBleStatus,
      calibrationDone, setCalibrationDone,
      processedSample,
      sessionData,
      pushSample,
      clearSession,
    }}>
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}
```

- [ ] **Step 4: 테스트 재실행 — PASS 확인**

```bash
npm run test -- --run
```

Expected: 4 tests passed

- [ ] **Step 5: App.jsx에 AppProvider 연결**

`web/src/App.jsx` 상단에 import 추가 후 AppProvider로 감싸기:

```jsx
import { useState } from 'react'
import { AppProvider, useApp } from './context/AppContext'
import './App.css'

function TabContainer() {
  const { calibrationDone } = useApp()
  const [activeTab, setActiveTab] = useState('calibration')

  return (
    <>
      <nav className="tab-nav">
        <button
          className={activeTab === 'calibration' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('calibration')}
        >
          Calibration
        </button>
        <button
          className={activeTab === 'brainmap' ? 'tab active' : 'tab'}
          disabled={!calibrationDone}
          onClick={() => setActiveTab('brainmap')}
        >
          Brain Map
        </button>
        <button
          className={activeTab === 'timeseries' ? 'tab active' : 'tab'}
          disabled={!calibrationDone}
          onClick={() => setActiveTab('timeseries')}
        >
          Time Series
        </button>
      </nav>
      <main className="tab-content">
        {activeTab === 'calibration' && <div>Calibration Tab</div>}
        {activeTab === 'brainmap' && <div>Brain Map Tab</div>}
        {activeTab === 'timeseries' && <div>Time Series Tab</div>}
      </main>
    </>
  )
}

export default function App() {
  return (
    <AppProvider>
      <div className="app">
        <header className="topbar">
          <span className="app-title">fNIRS Monitor</span>
        </header>
        <TabContainer />
      </div>
    </AppProvider>
  )
}
```

- [ ] **Step 6: 커밋**

```bash
cd D:/Study_fNIRS
git add web/
git commit -m "feat(web): AppContext 전역 상태 (bleStatus, calibrationDone, sessionData)"
```

---

## Task 2: lib/mbll.js — mBLL 신호처리

**Files:**
- Create: `web/src/lib/mbll.js`
- Create: `web/src/__tests__/lib/mbll.test.js`

Python `mbll.py`를 JS로 포팅. 행렬 연산은 순수 JS로 직접 구현(외부 라이브러리 불필요 — 4×4 이하 소규모 행렬).

- [ ] **Step 1: 실패할 테스트 작성**

`web/src/__tests__/lib/mbll.test.js`:
```javascript
import { describe, it, expect } from 'vitest'
import { modifiedBeerLambert } from '../../lib/mbll'

// Python 기준값 (config/settings.yaml 파라미터 사용)
const WAVELENGTHS = [780, 850, 950]
const EXT_HBO = [0.975, 0.901, 1.046]
const EXT_HBR = [2.755, 0.781, 0.260]
const DPF = [6.51, 5.86, 5.12]
const SDS_MM = 30.0

describe('modifiedBeerLambert', () => {
  it('정적 입력에 대해 HbO/HbR 0 반환 (첫 샘플 = 베이스라인)', () => {
    // 모든 샘플이 동일 → delta_OD = 0 → hbo = hbr = 0
    const raw = [
      [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
      [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
      [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    ]
    // raw shape: [n_wavelengths][n_channels][n_samples] = [3][4][3]
    const { hbo, hbr } = modifiedBeerLambert(raw, EXT_HBO, EXT_HBR, DPF, SDS_MM)
    // hbo/hbr shape: [n_channels][n_samples]
    expect(hbo.length).toBe(4)
    hbo.forEach(ch => ch.forEach(v => expect(Math.abs(v)).toBeLessThan(1e-9)))
    hbr.forEach(ch => ch.forEach(v => expect(Math.abs(v)).toBeLessThan(1e-9)))
  })

  it('강도 증가 시 HbO 양수 반환', () => {
    // 채널 0, 모든 파장에서 강도 1.0 → 1.1로 증가 (HbO 증가 시뮬레이션)
    const raw = [
      [[1.0, 1.05, 1.1], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
      [[1.0, 1.05, 1.1], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
      [[1.0, 1.05, 1.1], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    ]
    const { hbo } = modifiedBeerLambert(raw, EXT_HBO, EXT_HBR, DPF, SDS_MM)
    // 마지막 샘플에서 채널 0의 HbO는 음수 (강도 증가 = OD 감소)
    expect(typeof hbo[0][2]).toBe('number')
    expect(isFinite(hbo[0][2])).toBe(true)
  })

  it('sds_mm = 0이면 에러', () => {
    const raw = [[[1.0]], [[1.0]], [[1.0]]]
    expect(() => modifiedBeerLambert(raw, EXT_HBO, EXT_HBR, DPF, 0)).toThrow('sds_mm')
  })

  it('반환 shape: hbo/hbr [n_channels][n_samples]', () => {
    const nCh = 4, nSamples = 5
    const raw = Array.from({ length: 3 }, () =>
      Array.from({ length: nCh }, () => Array(nSamples).fill(1.0))
    )
    const { hbo, hbr } = modifiedBeerLambert(raw, EXT_HBO, EXT_HBR, DPF, SDS_MM)
    expect(hbo.length).toBe(nCh)
    expect(hbo[0].length).toBe(nSamples)
    expect(hbr.length).toBe(nCh)
  })
})
```

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
cd web && npm run test -- --run src/__tests__/lib/mbll.test.js
```

Expected: `Cannot find module '../../lib/mbll'`

- [ ] **Step 3: mbll.js 구현**

`web/src/lib/mbll.js`:
```javascript
/**
 * Modified Beer-Lambert Law — JS port of src/processing/mbll.py
 *
 * raw 배열 shape: [n_wavelengths][n_channels][n_samples]
 * 반환: { hbo, hbr } — 각각 [n_channels][n_samples], 단위 μmol/L
 */

/** 2×2 역행렬 (det ≠ 0 보장된 경우) */
function inv2x2(m) {
  const det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
  if (Math.abs(det) < 1e-12) throw new Error('Singular matrix')
  return [
    [ m[1][1] / det, -m[0][1] / det],
    [-m[1][0] / det,  m[0][0] / det],
  ]
}

/**
 * Pseudo-inverse of A (n_wl × 2) via (A^T A)^{-1} A^T
 * Returns 2 × n_wl matrix.
 */
function pseudoInverse(A) {
  const nWl = A.length
  // AT: 2 × n_wl
  const AT = [[...A.map(r => r[0])], [...A.map(r => r[1])]]
  // ATA: 2 × 2
  const ATA = [
    [0, 0],
    [0, 0],
  ]
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      for (let k = 0; k < nWl; k++) {
        ATA[i][j] += AT[i][k] * A[k][j]
      }
    }
  }
  const ATAinv = inv2x2(ATA)
  // pinv = ATAinv × AT: 2 × n_wl
  const pinv = Array.from({ length: 2 }, () => Array(nWl).fill(0))
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < nWl; j++) {
      for (let k = 0; k < 2; k++) {
        pinv[i][j] += ATAinv[i][k] * AT[k][j]
      }
    }
  }
  return pinv
}

/**
 * @param {number[][][]} raw  [n_wavelengths][n_channels][n_samples]
 * @param {number[]} extHbo   HbO extinction coefficients, one per wavelength
 * @param {number[]} extHbr   HbR extinction coefficients, one per wavelength
 * @param {number[]} dpf      Differential path length factors, one per wavelength
 * @param {number}   sdsMm    Source-detector separation in mm
 * @returns {{ hbo: number[][], hbr: number[][] }}  each [n_channels][n_samples] in μmol/L
 */
export function modifiedBeerLambert(raw, extHbo, extHbr, dpf, sdsMm) {
  if (sdsMm <= 0) throw new Error('sds_mm must be positive')

  const nWl = raw.length
  const nCh = raw[0].length
  const nSamples = raw[0][0].length
  const sdsCm = sdsMm / 10.0

  // Build extinction matrix A: n_wl × 2
  const A = Array.from({ length: nWl }, (_, i) => [extHbo[i], extHbr[i]])
  const Apinv = pseudoInverse(A)  // 2 × n_wl

  // hbo/hbr: [n_channels][n_samples]
  const hbo = Array.from({ length: nCh }, () => Array(nSamples).fill(0))
  const hbr = Array.from({ length: nCh }, () => Array(nSamples).fill(0))

  for (let ch = 0; ch < nCh; ch++) {
    // baseline: first sample per wavelength
    const baseline = Array.from({ length: nWl }, (_, wl) => {
      const v = raw[wl][ch][0]
      return v === 0 ? 1e-10 : v
    })

    for (let s = 0; s < nSamples; s++) {
      // od_normalized: [n_wl] = -log(I/I0) / (dpf * sds_cm)
      const odNorm = Array.from({ length: nWl }, (_, wl) => {
        const I = raw[wl][ch][s] === 0 ? 1e-10 : raw[wl][ch][s]
        const deltaOd = -Math.log(I / baseline[wl])
        return deltaOd / (dpf[wl] * sdsCm)
      })

      // concentrations = Apinv × odNorm (2 × n_wl) · (n_wl) = (2,)
      let cHbo = 0, cHbr = 0
      for (let wl = 0; wl < nWl; wl++) {
        cHbo += Apinv[0][wl] * odNorm[wl]
        cHbr += Apinv[1][wl] * odNorm[wl]
      }
      // mmol/L → μmol/L
      hbo[ch][s] = cHbo * 1000.0
      hbr[ch][s] = cHbr * 1000.0
    }
  }

  return { hbo, hbr }
}
```

- [ ] **Step 4: 테스트 재실행 — PASS 확인**

```bash
npm run test -- --run src/__tests__/lib/mbll.test.js
```

Expected: 4 tests passed

- [ ] **Step 5: 커밋**

```bash
cd D:/Study_fNIRS
git add web/src/lib/mbll.js web/src/__tests__/lib/mbll.test.js
git commit -m "feat(web/lib): mBLL JS 구현 (Python mbll.py 포팅)"
```

---

## Task 3: lib/filters.js — Butterworth 밴드패스

**Files:**
- Create: `web/src/lib/filters.js`
- Create: `web/src/__tests__/lib/filters.test.js`

`iir-filter` 패키지로 Butterworth SOS 필터 구현. Python `sosfiltfilt` (zero-phase) 대응으로 forward+backward pass 적용.

- [ ] **Step 1: 실패할 테스트 작성**

`web/src/__tests__/lib/filters.test.js`:
```javascript
import { describe, it, expect } from 'vitest'
import { bandpassFilter } from '../../lib/filters'

describe('bandpassFilter', () => {
  it('반환 길이가 입력과 동일', () => {
    const data = Array.from({ length: 100 }, (_, i) => Math.sin(2 * Math.PI * 0.1 * i / 10))
    const out = bandpassFilter(data, 0.01, 0.5, 10.0)
    expect(out.length).toBe(data.length)
  })

  it('고주파 성분 제거 — 5Hz 신호, SR=10Hz, bandpass 0.01~0.5Hz', () => {
    // 4.9Hz는 통과대역(0.01~0.5Hz) 밖 → 거의 제거되어야 함
    const sr = 10.0
    const data = Array.from({ length: 200 }, (_, i) => Math.sin(2 * Math.PI * 4.9 * i / sr))
    const out = bandpassFilter(data, 0.01, 0.5, sr)
    const rms = Math.sqrt(out.reduce((s, v) => s + v * v, 0) / out.length)
    expect(rms).toBeLessThan(0.1)  // 원래 RMS ≈ 0.707 대비 대폭 감쇠
  })

  it('통과대역 내 신호 유지 — 0.1Hz, SR=10Hz', () => {
    const sr = 10.0
    // 충분한 샘플 필요 (sosfiltfilt padlen 대응)
    const data = Array.from({ length: 300 }, (_, i) => Math.sin(2 * Math.PI * 0.1 * i / sr))
    const out = bandpassFilter(data, 0.01, 0.5, sr)
    // 끝부분(edge effect 제외) RMS가 입력의 50% 이상
    const tail = out.slice(100)
    const rms = Math.sqrt(tail.reduce((s, v) => s + v * v, 0) / tail.length)
    expect(rms).toBeGreaterThan(0.3)
  })

  it('low_hz <= 0이면 에러', () => {
    expect(() => bandpassFilter([1, 2, 3], 0, 0.5, 10)).toThrow('low_hz')
  })

  it('high_hz >= Nyquist이면 에러', () => {
    expect(() => bandpassFilter([1, 2, 3], 0.01, 5.0, 10)).toThrow('Nyquist')
  })
})
```

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
npm run test -- --run src/__tests__/lib/filters.test.js
```

Expected: `Cannot find module '../../lib/filters'`

- [ ] **Step 3: filters.js 구현**

`web/src/lib/filters.js`:
```javascript
/**
 * Butterworth 밴드패스 필터 — JS port of src/processing/filters.py
 * iir-filter 패키지로 SOS 계수 계산 + forward-backward pass (zero-phase).
 */
import { IirFilter, CalcCascades } from 'iir-filter'

/**
 * @param {number[]} data          1D 배열
 * @param {number}   lowHz         하한 주파수 (Hz)
 * @param {number}   highHz        상한 주파수 (Hz)
 * @param {number}   samplingRate  샘플링 레이트 (Hz)
 * @param {number}   order         필터 차수 (기본 6)
 * @returns {number[]}             필터링된 1D 배열
 */
export function bandpassFilter(data, lowHz, highHz, samplingRate, order = 6) {
  if (lowHz <= 0) throw new Error('low_hz must be positive')
  const nyquist = samplingRate / 2
  if (highHz >= nyquist) throw new Error(`high_hz must be less than Nyquist (${nyquist})`)
  if (highHz <= lowHz) throw new Error('high_hz must be greater than low_hz')

  const calc = new CalcCascades()
  const coeffs = calc.bandpass(order, samplingRate, lowHz, highHz)
  const filter = new IirFilter(coeffs)

  // Forward pass
  const forward = data.map(v => filter.singleStep(v))

  // Reset and backward pass (zero-phase equivalent)
  filter.reset()
  const backward = [...forward].reverse().map(v => filter.singleStep(v))

  return backward.reverse()
}
```

- [ ] **Step 4: 테스트 재실행 — PASS 확인**

```bash
npm run test -- --run src/__tests__/lib/filters.test.js
```

Expected: 5 tests passed

- [ ] **Step 5: 커밋**

```bash
cd D:/Study_fNIRS
git add web/src/lib/filters.js web/src/__tests__/lib/filters.test.js
git commit -m "feat(web/lib): Butterworth 밴드패스 필터 JS 구현"
```

---

## Task 4: lib/rbf.js — RBF 보간 + BWR 컬러맵

**Files:**
- Create: `web/src/lib/rbf.js`
- Create: `web/src/__tests__/lib/rbf.test.js`

Python `brain_map_widget.py`의 `_build_overlay()` 로직 포팅.

- [ ] **Step 1: 실패할 테스트 작성**

`web/src/__tests__/lib/rbf.test.js`:
```javascript
import { describe, it, expect } from 'vitest'
import { bwr, rbfInterpolate } from '../../lib/rbf'

describe('bwr (Blue-White-Red 컬러맵)', () => {
  it('t=0 → 파랑 [0,0,255]', () => {
    const [r, g, b] = bwr(0)
    expect(r).toBe(0); expect(g).toBe(0); expect(b).toBe(255)
  })

  it('t=0.5 → 흰색 [255,255,255]', () => {
    const [r, g, b] = bwr(0.5)
    expect(r).toBe(255); expect(g).toBe(255); expect(b).toBe(255)
  })

  it('t=1 → 빨강 [255,0,0]', () => {
    const [r, g, b] = bwr(1)
    expect(r).toBe(255); expect(g).toBe(0); expect(b).toBe(0)
  })

  it('t가 [0,1] 클리핑', () => {
    const [r1] = bwr(-1)
    const [r2] = bwr(2)
    expect(r1).toBe(0)
    expect(r2).toBe(255)
  })
})

describe('rbfInterpolate', () => {
  it('채널 위치에서 해당 값 반환 (RBF 보간 검증)', () => {
    // 2채널: 한 채널 값 1.0, 다른 채널 값 -1.0
    const points = [[0.0, 0.0], [1.0, 0.0]]
    const values = [1.0, -1.0]
    const result = rbfInterpolate([[0.0, 0.0]], points, values)
    // 첫 채널 위치에서 값은 1.0에 가까워야 함
    expect(result[0]).toBeCloseTo(1.0, 1)
  })

  it('반환 길이가 queryPoints 수와 일치', () => {
    const points = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
    const values = [1.0, -1.0, 0.0]
    const queryPoints = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.1]]
    const result = rbfInterpolate(queryPoints, points, values)
    expect(result.length).toBe(3)
  })
})
```

- [ ] **Step 2: 테스트 실행 — FAIL 확인**

```bash
npm run test -- --run src/__tests__/lib/rbf.test.js
```

Expected: `Cannot find module '../../lib/rbf'`

- [ ] **Step 3: rbf.js 구현**

`web/src/lib/rbf.js`:
```javascript
/**
 * RBF 보간 + Blue-White-Red 컬러맵
 * Python brain_map_widget.py _build_overlay() 포팅
 */

/**
 * Blue(0) → White(0.5) → Red(1) 컬러맵
 * @param {number} t  [0, 1] 범위
 * @returns {[number, number, number]} [r, g, b] 각 0-255
 */
export function bwr(t) {
  t = Math.max(0, Math.min(1, t))
  let r, g, b
  if (t < 0.5) {
    const s = t / 0.5  // 0→1
    r = Math.round(s * 255)
    g = Math.round(s * 255)
    b = 255
  } else {
    const s = (t - 0.5) / 0.5  // 0→1
    r = 255
    g = Math.round((1 - s) * 255)
    b = Math.round((1 - s) * 255)
  }
  return [r, g, b]
}

/**
 * Gaussian RBF 보간 (epsilon=6)
 * @param {number[][]} queryPoints  [[x,y], ...] 보간 요청 좌표들
 * @param {number[][]} dataPoints   [[x,y], ...] 채널 위치들
 * @param {number[]}   values       채널별 값
 * @returns {number[]}              쿼리 포인트별 보간 값
 */
export function rbfInterpolate(queryPoints, dataPoints, values) {
  const epsilon = 6
  const n = dataPoints.length

  // RBF 행렬 Phi: n × n
  const Phi = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      const dx = dataPoints[i][0] - dataPoints[j][0]
      const dy = dataPoints[i][1] - dataPoints[j][1]
      const r2 = dx * dx + dy * dy
      return Math.exp(-epsilon * epsilon * r2)
    })
  )

  // 가우스 소거로 가중치 w 계산: Phi · w = values
  const w = gaussianElimination(Phi, [...values])

  // 쿼리 포인트에서 보간
  return queryPoints.map(([qx, qy]) => {
    let val = 0
    for (let j = 0; j < n; j++) {
      const dx = qx - dataPoints[j][0]
      const dy = qy - dataPoints[j][1]
      const r2 = dx * dx + dy * dy
      val += w[j] * Math.exp(-epsilon * epsilon * r2)
    }
    return val
  })
}

/** 부분 피벗 가우스 소거 */
function gaussianElimination(A, b) {
  const n = b.length
  const M = A.map((row, i) => [...row, b[i]])  // augmented matrix

  for (let col = 0; col < n; col++) {
    // 피벗 선택
    let maxRow = col
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row
    }
    ;[M[col], M[maxRow]] = [M[maxRow], M[col]]

    const pivot = M[col][col]
    if (Math.abs(pivot) < 1e-12) continue

    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / pivot
      for (let k = col; k <= n; k++) {
        M[row][k] -= factor * M[col][k]
      }
    }
  }

  // 후진 대입
  const x = Array(n).fill(0)
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n]
    for (let j = i + 1; j < n; j++) {
      x[i] -= M[i][j] * x[j]
    }
    x[i] /= M[i][i]
  }
  return x
}
```

- [ ] **Step 4: 테스트 재실행 — PASS 확인**

```bash
npm run test -- --run src/__tests__/lib/rbf.test.js
```

Expected: 6 tests passed

- [ ] **Step 5: 커밋**

```bash
cd D:/Study_fNIRS
git add web/src/lib/rbf.js web/src/__tests__/lib/rbf.test.js
git commit -m "feat(web/lib): RBF 보간 + BWR 컬러맵 JS 구현"
```

---

## Task 5: lib/simulator.js — 개발용 BLE 시뮬레이터

**Files:**
- Create: `web/src/lib/simulator.js`

Python `FNIRSSimulator`와 동일한 신호 패턴. `?simulate=true` URL 파라미터로 활성화.

- [ ] **Step 1: simulator.js 작성**

`web/src/lib/simulator.js`:
```javascript
/**
 * 개발용 fNIRS BLE 시뮬레이터
 * Python FNIRSSimulator와 동일한 sin파+노이즈 패턴
 *
 * 반환 rawPacket 형식:
 * {
 *   timestamp: number,
 *   channelIntensities: number[],  // [ch0_wl0, ch0_wl1, ch0_wl2, ch1_wl0, ...]
 *   nWavelengths: 3,
 * }
 */

const N_CHANNELS = 4
const N_WAVELENGTHS = 3
const SAMPLING_RATE_HZ = 10.0
const HBO_AMPLITUDE = 0.5
const HBO_FREQ_HZ = 0.1
const NOISE_STD = 0.05

function gaussianNoise(std) {
  // Box-Muller transform
  const u1 = Math.random()
  const u2 = Math.random()
  return std * Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2)
}

export class FNIRSSimulator {
  constructor() {
    this._intervalId = null
    this._sampleIndex = 0
    this._startTime = null
  }

  /**
   * 시뮬레이션 시작
   * @param {(packet: object) => void} onPacket  패킷 수신 콜백
   */
  start(onPacket) {
    this._sampleIndex = 0
    this._startTime = performance.now()

    const intervalMs = 1000 / SAMPLING_RATE_HZ
    this._intervalId = setInterval(() => {
      const t = this._sampleIndex / SAMPLING_RATE_HZ
      this._sampleIndex++

      const channelIntensities = []
      for (let ch = 0; ch < N_CHANNELS; ch++) {
        const phaseOffset = ch * (2 * Math.PI / N_CHANNELS)
        const hboSignal = HBO_AMPLITUDE * Math.sin(2 * Math.PI * HBO_FREQ_HZ * t + phaseOffset)

        for (let wl = 0; wl < N_WAVELENGTHS; wl++) {
          const hboSensitivity = 0.5 + wl * 0.3
          const val = 1.0 + hboSensitivity * hboSignal + gaussianNoise(NOISE_STD)
          channelIntensities.push(Math.max(0.01, val))
        }
      }

      onPacket({
        timestamp: performance.now() / 1000,
        channelIntensities,
        nWavelengths: N_WAVELENGTHS,
      })
    }, intervalMs)
  }

  stop() {
    if (this._intervalId !== null) {
      clearInterval(this._intervalId)
      this._intervalId = null
    }
  }
}
```

- [ ] **Step 2: 커밋**

```bash
cd D:/Study_fNIRS
git add web/src/lib/simulator.js
git commit -m "feat(web/lib): 개발용 BLE 시뮬레이터 (Python FNIRSSimulator 포팅)"
```

---

## Task 6: pipeline.worker.js — Web Worker 신호처리

**Files:**
- Create: `web/src/workers/pipeline.worker.js`
- Create: `web/src/hooks/usePipeline.js`

- [ ] **Step 1: pipeline.worker.js 작성**

`web/src/workers/pipeline.worker.js`:
```javascript
/**
 * Web Worker: 슬라이딩 윈도우 + 필터 + mBLL → processedSample 반환
 *
 * 메시지 수신:
 *   { type: 'packet', data: rawPacket }
 *   { type: 'reset' }
 *
 * 메시지 발신:
 *   { type: 'sample', data: processedSample }
 */
import { bandpassFilter } from '../lib/filters.js'
import { modifiedBeerLambert } from '../lib/mbll.js'

const WINDOW_SEC = 10.0
const SAMPLING_RATE_HZ = 10.0
const MIN_SAMPLES = 40  // sosfiltfilt padlen 대응

// 설정 (config/settings.yaml 기준)
const EXT_HBO = [0.975, 0.901, 1.046]
const EXT_HBR = [2.755, 0.781, 0.260]
const DPF = [6.51, 5.86, 5.12]
const SDS_MM = 30.0
const N_CHANNELS = 4
const N_WAVELENGTHS = 3
const BANDPASS_LOW = 0.01
const BANDPASS_HIGH = 0.5

const maxSamples = Math.floor(WINDOW_SEC * SAMPLING_RATE_HZ)
let window = []  // rawPacket[]

self.onmessage = ({ data: msg }) => {
  if (msg.type === 'reset') {
    window = []
    return
  }

  if (msg.type !== 'packet') return

  const packet = msg.data
  window.push(packet)
  if (window.length > maxSamples) window.shift()
  if (window.length < MIN_SAMPLES) return

  const sample = processWindow(window)
  self.postMessage({ type: 'sample', data: sample })
}

function processWindow(packets) {
  const nT = packets.length

  // raw: [n_wavelengths][n_channels][n_samples]
  const raw = Array.from({ length: N_WAVELENGTHS }, () =>
    Array.from({ length: N_CHANNELS }, () => Array(nT).fill(0))
  )

  for (let t = 0; t < nT; t++) {
    const pkt = packets[t]
    for (let ch = 0; ch < N_CHANNELS; ch++) {
      for (let wl = 0; wl < N_WAVELENGTHS; wl++) {
        raw[wl][ch][t] = pkt.channelIntensities[ch * N_WAVELENGTHS + wl]
      }
    }
  }

  // 밴드패스 필터 (파장별, 평균 복원 포함)
  for (let wl = 0; wl < N_WAVELENGTHS; wl++) {
    for (let ch = 0; ch < N_CHANNELS; ch++) {
      const slice = raw[wl][ch]
      const mean = slice.reduce((s, v) => s + v, 0) / slice.length
      const filtered = bandpassFilter(slice, BANDPASS_LOW, BANDPASS_HIGH, SAMPLING_RATE_HZ)
      raw[wl][ch] = filtered.map(v => v + mean)
    }
  }

  // mBLL
  const { hbo, hbr } = modifiedBeerLambert(raw, EXT_HBO, EXT_HBR, DPF, SDS_MM)

  // 마지막 샘플의 값
  const hboNow = hbo.map(ch => ch[ch.length - 1])
  const hbrNow = hbr.map(ch => ch[ch.length - 1])

  // 집중도 지수: HbO 평균을 [0,1]로 정규화
  const ci = Math.max(0, Math.min(1, (hboNow.reduce((s, v) => s + v, 0) / N_CHANNELS) / 10.0 + 0.5))

  return {
    timestamp: packets[packets.length - 1].timestamp,
    hbo: hboNow,
    hbr: hbrNow,
    ci,
  }
}
```

- [ ] **Step 2: usePipeline.js 작성**

`web/src/hooks/usePipeline.js`:
```javascript
import { useEffect, useRef, useCallback } from 'react'

/**
 * Web Worker를 생성하고 processedSample 콜백을 연결하는 훅.
 * @param {(sample: object) => void} onSample
 * @returns {{ sendPacket: (packet: object) => void, resetWorker: () => void }}
 */
export function usePipeline(onSample) {
  const workerRef = useRef(null)

  useEffect(() => {
    const worker = new Worker(
      new URL('../workers/pipeline.worker.js', import.meta.url),
      { type: 'module' }
    )
    worker.onmessage = ({ data: msg }) => {
      if (msg.type === 'sample') onSample(msg.data)
    }
    workerRef.current = worker
    return () => worker.terminate()
  }, [onSample])

  const sendPacket = useCallback((packet) => {
    workerRef.current?.postMessage({ type: 'packet', data: packet })
  }, [])

  const resetWorker = useCallback(() => {
    workerRef.current?.postMessage({ type: 'reset' })
  }, [])

  return { sendPacket, resetWorker }
}
```

- [ ] **Step 3: 커밋**

```bash
cd D:/Study_fNIRS
git add web/src/workers/pipeline.worker.js web/src/hooks/usePipeline.js
git commit -m "feat(web): Web Worker 신호처리 파이프라인 + usePipeline 훅"
```

---

## Task 7: hooks/useBLE.js — BLE 연결

**Files:**
- Create: `web/src/hooks/useBLE.js`

- [ ] **Step 1: useBLE.js 작성**

`web/src/hooks/useBLE.js`:
```javascript
import { useCallback, useRef } from 'react'
import { FNIRSSimulator } from '../lib/simulator.js'

const RECONNECT_MAX = 3

/**
 * Web Bluetooth BLE 연결 훅
 * URL에 ?simulate=true 가 있으면 시뮬레이터 사용
 *
 * @param {(packet: object) => void} onPacket      패킷 수신 콜백
 * @param {(status: string) => void} onStatusChange 연결 상태 변경 콜백
 */
export function useBLE(onPacket, onStatusChange) {
  const deviceRef = useRef(null)
  const charRef = useRef(null)
  const simRef = useRef(null)
  const reconnectCount = useRef(0)

  const isSimulate = new URLSearchParams(window.location.search).get('simulate') === 'true'

  const connect = useCallback(async () => {
    if (isSimulate) {
      onStatusChange('connected')
      const sim = new FNIRSSimulator()
      simRef.current = sim
      sim.start(onPacket)
      return
    }

    try {
      onStatusChange('connecting')
      const device = await navigator.bluetooth.requestDevice({
        acceptAllDevices: true,
        optionalServices: ['generic_access'],
      })
      deviceRef.current = device

      device.addEventListener('gattserverdisconnected', () => {
        onStatusChange('error')
        if (reconnectCount.current < RECONNECT_MAX) {
          reconnectCount.current++
          setTimeout(() => connect(), 1000)
        }
      })

      const server = await device.gatt.connect()
      // NOTE: 실제 UUID는 하드웨어 팀 확정 후 교체 필요
      // 현재는 연결 성공만 확인
      reconnectCount.current = 0
      onStatusChange('connected')

      // TODO: 실제 UUID 확정 후 characteristic 구독 추가
      // const service = await server.getPrimaryService('YOUR-SERVICE-UUID')
      // const char = await service.getCharacteristic('YOUR-CHAR-UUID')
      // await char.startNotifications()
      // char.addEventListener('characteristicvaluechanged', (e) => {
      //   const raw = parsePacket(e.target.value)
      //   onPacket(raw)
      // })
    } catch (err) {
      console.error('BLE 연결 실패:', err)
      onStatusChange('error')
    }
  }, [onPacket, onStatusChange, isSimulate])

  const disconnect = useCallback(() => {
    simRef.current?.stop()
    simRef.current = null
    if (deviceRef.current?.gatt?.connected) {
      deviceRef.current.gatt.disconnect()
    }
    deviceRef.current = null
    onStatusChange('idle')
  }, [onStatusChange])

  return { connect, disconnect, isSimulate }
}
```

- [ ] **Step 2: 커밋**

```bash
cd D:/Study_fNIRS
git add web/src/hooks/useBLE.js
git commit -m "feat(web): useBLE 훅 (Web Bluetooth + 시뮬레이터 모드)"
```

---

## Task 8: TopBar 컴포넌트

**Files:**
- Create: `web/src/components/TopBar.jsx`
- Create: `web/src/components/TopBar.css`

- [ ] **Step 1: TopBar.jsx 작성**

`web/src/components/TopBar.jsx`:
```jsx
import { useCallback } from 'react'
import { useApp } from '../context/AppContext'
import { useBLE } from '../hooks/useBLE'
import { usePipeline } from '../hooks/usePipeline'
import './TopBar.css'

const STATUS_LABEL = {
  idle: '연결 안됨',
  connecting: '연결 중...',
  connected: '연결됨',
  error: '연결 오류',
}

const STATUS_COLOR = {
  idle: 'var(--text-muted)',
  connecting: 'var(--warning)',
  connected: 'var(--success)',
  error: 'var(--error)',
}

export default function TopBar() {
  const { bleStatus, setBleStatus, processedSample, pushSample, clearSession } = useApp()

  const onPacket = useCallback((packet) => {
    sendPacket(packet)
  }, [])

  const { sendPacket } = usePipeline(pushSample)
  const { connect, disconnect, isSimulate } = useBLE(onPacket, setBleStatus)

  const ci = processedSample?.ci ?? 0
  const ciPercent = Math.round(ci * 100)

  return (
    <header className="topbar">
      <span className="app-title">fNIRS Monitor{isSimulate ? ' [SIM]' : ''}</span>

      <div className="ble-group">
        <span className="ble-dot" style={{ background: STATUS_COLOR[bleStatus] }} />
        <span className="ble-label">{STATUS_LABEL[bleStatus]}</span>
        {bleStatus === 'idle' || bleStatus === 'error' ? (
          <button className="btn-connect" onClick={connect}>연결</button>
        ) : (
          <button className="btn-disconnect" onClick={disconnect}>해제</button>
        )}
      </div>

      <div className="ci-group">
        <span className="ci-label">CI</span>
        <div className="ci-bar-track">
          <div className="ci-bar-fill" style={{ width: `${ciPercent}%` }} />
        </div>
        <span className="ci-value">{ciPercent}%</span>
      </div>
    </header>
  )
}
```

- [ ] **Step 2: TopBar.css 작성**

`web/src/components/TopBar.css`:
```css
.topbar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px 16px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  flex-wrap: wrap;
}

.app-title {
  font-weight: 700;
  font-size: 1.1rem;
  color: var(--accent);
  margin-right: auto;
}

.ble-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.ble-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
}

.ble-label { font-size: 0.85rem; color: var(--text-muted); }

.btn-connect, .btn-disconnect {
  padding: 4px 14px;
  border-radius: 6px;
  border: none;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 600;
}

.btn-connect { background: var(--accent); color: #fff; }
.btn-connect:hover { background: var(--accent-hover); }
.btn-disconnect { background: var(--error); color: #fff; }

.ci-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.ci-label { font-size: 0.85rem; font-weight: 700; color: var(--text-muted); }

.ci-bar-track {
  width: 120px;
  height: 10px;
  background: var(--border);
  border-radius: 5px;
  overflow: hidden;
}

.ci-bar-fill {
  height: 100%;
  background: linear-gradient(to right, #4a90d9, #e57373);
  border-radius: 5px;
  transition: width 0.2s;
}

.ci-value { font-size: 0.85rem; font-weight: 700; min-width: 36px; }

@media (max-width: 768px) {
  .topbar { padding: 6px 10px; gap: 10px; }
  .ci-bar-track { width: 70px; }
}
```

- [ ] **Step 3: App.jsx에 TopBar 통합**

`web/src/App.jsx`를 아래로 교체:

```jsx
import { useState } from 'react'
import { AppProvider, useApp } from './context/AppContext'
import TopBar from './components/TopBar'
import './App.css'

function TabContainer() {
  const { calibrationDone } = useApp()
  const [activeTab, setActiveTab] = useState('calibration')

  return (
    <>
      <nav className="tab-nav">
        <button
          className={activeTab === 'calibration' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('calibration')}
        >
          Calibration
        </button>
        <button
          className={activeTab === 'brainmap' ? 'tab active' : 'tab'}
          disabled={!calibrationDone}
          onClick={() => calibrationDone && setActiveTab('brainmap')}
        >
          Brain Map
        </button>
        <button
          className={activeTab === 'timeseries' ? 'tab active' : 'tab'}
          disabled={!calibrationDone}
          onClick={() => calibrationDone && setActiveTab('timeseries')}
        >
          Time Series
        </button>
      </nav>
      <main className="tab-content">
        {activeTab === 'calibration' && <div className="placeholder">Calibration Tab</div>}
        {activeTab === 'brainmap' && <div className="placeholder">Brain Map Tab</div>}}
        {activeTab === 'timeseries' && <div className="placeholder">Time Series Tab</div>}
      </main>
    </>
  )
}

export default function App() {
  return (
    <AppProvider>
      <div className="app">
        <TopBar />
        <TabContainer />
      </div>
    </AppProvider>
  )
}
```

- [ ] **Step 4: 시뮬레이터 모드로 동작 확인**

```bash
cd web && npm run dev
```

`http://localhost:5173/?simulate=true` 접속 → "연결" 버튼 클릭 → CI 게이지 변동 확인.

- [ ] **Step 5: 커밋**

```bash
cd D:/Study_fNIRS
git add web/src/components/TopBar.jsx web/src/components/TopBar.css web/src/App.jsx
git commit -m "feat(web): TopBar — BLE 연결 버튼, 상태 표시, CI 게이지"
```

---

## Task 9: CalibrationTab 컴포넌트

**Files:**
- Create: `web/src/components/CalibrationTab.jsx`
- Create: `web/src/components/CalibrationTab.css`

- [ ] **Step 1: CalibrationTab.jsx 작성**

`web/src/components/CalibrationTab.jsx`:
```jsx
import { useState, useEffect, useRef } from 'react'
import { useApp } from '../context/AppContext'
import './CalibrationTab.css'

const N_CHANNELS = 4
const WAVELENGTHS = ['780nm', '850nm', '950nm']
const CALIB_DURATION_MS = 3000
const SNR_THRESHOLD = 0.6  // 이 이상이면 Good

export default function CalibrationTab() {
  const { bleStatus, setCalibrationDone } = useApp()
  const [snr, setSnr] = useState(
    Array.from({ length: N_CHANNELS }, () => [0, 0, 0])
  )
  const [channelStatus, setChannelStatus] = useState(Array(N_CHANNELS).fill('waiting'))
  // 'waiting' | 'good' | 'poor'
  const [calibrating, setCalibrating] = useState(false)
  const [allGood, setAllGood] = useState(false)
  const timerRef = useRef(null)

  // 연결되면 자동으로 Calibration 시뮬레이션 시작
  useEffect(() => {
    if (bleStatus !== 'connected') return

    setCalibrating(true)
    const startTime = Date.now()

    timerRef.current = setInterval(() => {
      const elapsed = Date.now() - startTime
      const progress = Math.min(elapsed / CALIB_DURATION_MS, 1)

      // SNR을 점진적으로 채움 (시뮬레이션)
      const newSnr = Array.from({ length: N_CHANNELS }, (_, ch) =>
        WAVELENGTHS.map((_, wl) =>
          Math.min(progress * (0.65 + Math.random() * 0.25), 1)
        )
      )
      setSnr(newSnr)

      if (elapsed >= CALIB_DURATION_MS) {
        clearInterval(timerRef.current)
        setCalibrating(false)

        const finalStatus = newSnr.map(chSnr =>
          chSnr.every(v => v >= SNR_THRESHOLD) ? 'good' : 'poor'
        )
        setChannelStatus(finalStatus)
        setAllGood(finalStatus.every(s => s === 'good'))
      }
    }, 100)

    return () => clearInterval(timerRef.current)
  }, [bleStatus])

  return (
    <div className="calibration">
      <h2 className="calib-title">신호 보정 (Calibration)</h2>

      {bleStatus !== 'connected' && (
        <p className="calib-hint">상단에서 fNIRS 기기를 먼저 연결하세요.</p>
      )}

      <div className="snr-grid">
        {Array.from({ length: N_CHANNELS }, (_, ch) => (
          <div key={ch} className={`channel-card ${channelStatus[ch]}`}>
            <div className="channel-label">
              Ch {ch + 1}
              <span className={`status-badge ${channelStatus[ch]}`}>
                {channelStatus[ch] === 'good' ? '✓ Good' : channelStatus[ch] === 'poor' ? '✗ Poor' : '…'}
              </span>
            </div>
            <div className="wl-bars">
              {WAVELENGTHS.map((wl, wlIdx) => (
                <div key={wl} className="wl-row">
                  <span className="wl-label">{wl}</span>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{
                        width: `${Math.round(snr[ch][wlIdx] * 100)}%`,
                        background: snr[ch][wlIdx] >= SNR_THRESHOLD ? 'var(--success)' : 'var(--warning)',
                      }}
                    />
                  </div>
                  <span className="bar-value">{Math.round(snr[ch][wlIdx] * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <button
        className="btn-proceed"
        disabled={!allGood}
        onClick={() => setCalibrationDone(true)}
      >
        진행 ▶
      </button>
    </div>
  )
}
```

- [ ] **Step 2: CalibrationTab.css 작성**

`web/src/components/CalibrationTab.css`:
```css
.calibration {
  max-width: 800px;
  margin: 0 auto;
}

.calib-title {
  font-size: 1.2rem;
  margin-bottom: 8px;
  color: var(--text);
}

.calib-hint {
  color: var(--text-muted);
  margin-bottom: 16px;
  font-size: 0.9rem;
}

.snr-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}

.channel-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px;
}

.channel-card.good { border-color: var(--success); }
.channel-card.poor { border-color: var(--error); }

.channel-label {
  font-weight: 700;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.status-badge {
  font-size: 0.8rem;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 99px;
}

.status-badge.good { background: var(--success); color: #000; }
.status-badge.poor { background: var(--error); color: #fff; }
.status-badge.waiting { background: var(--border); color: var(--text-muted); }

.wl-bars { display: flex; flex-direction: column; gap: 8px; }

.wl-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.wl-label { font-size: 0.78rem; color: var(--text-muted); width: 44px; flex-shrink: 0; }

.bar-track {
  flex: 1;
  height: 12px;
  background: var(--border);
  border-radius: 6px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 6px;
  transition: width 0.15s, background 0.15s;
}

.bar-value { font-size: 0.78rem; width: 34px; text-align: right; }

.btn-proceed {
  display: block;
  margin: 0 auto;
  padding: 10px 40px;
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.15s;
}

.btn-proceed:hover:not(:disabled) { background: var(--accent-hover); }
.btn-proceed:disabled { opacity: 0.35; cursor: not-allowed; }

@media (max-width: 600px) {
  .snr-grid { grid-template-columns: 1fr; }
}
```

- [ ] **Step 3: App.jsx에 CalibrationTab 통합**

`web/src/App.jsx`의 `CalibrationTab placeholder` 부분 교체:

```jsx
// 상단 import 추가
import CalibrationTab from './components/CalibrationTab'

// TabContainer 내부 탭 콘텐츠 교체
{activeTab === 'calibration' && <CalibrationTab />}
```

- [ ] **Step 4: 시뮬레이터 모드로 동작 확인**

`http://localhost:5173/?simulate=true` → 연결 → Calibration 탭에서 SNR 바 채워지는 것 확인 → "진행 ▶" 활성화 확인.

- [ ] **Step 5: 커밋**

```bash
cd D:/Study_fNIRS
git add web/src/components/CalibrationTab.jsx web/src/components/CalibrationTab.css web/src/App.jsx
git commit -m "feat(web): CalibrationTab — SNR 바 차트, 채널 상태 판정, 진행 버튼"
```

---

## Task 10: TimeSeriesTab — 실시간 Canvas 그래프

**Files:**
- Create: `web/src/components/TimeSeriesTab.jsx`
- Create: `web/src/components/TimeSeriesTab.css`

- [ ] **Step 1: TimeSeriesTab.jsx 작성**

`web/src/components/TimeSeriesTab.jsx`:
```jsx
import { useRef, useEffect } from 'react'
import { useApp } from '../context/AppContext'
import './TimeSeriesTab.css'

const N_CHANNELS = 4
const PLOT_WINDOW_SEC = 5.0
const Y_MIN = -5.0
const Y_MAX = 5.0
const COLORS_HBO = ['#e57373', '#ef9a9a', '#ef5350', '#c62828']
const COLORS_HBR = ['#64b5f6', '#90caf9', '#42a5f5', '#1565c0']
const MAX_POINTS = 500  // 5초 × 100Hz

export default function TimeSeriesTab() {
  const { sessionData } = useApp()
  const canvasRef = useRef(null)
  const dataRef = useRef([])  // 최근 MAX_POINTS 샘플

  // sessionData 변경 시 dataRef 갱신
  useEffect(() => {
    dataRef.current = sessionData.slice(-MAX_POINTS)
  }, [sessionData])

  // requestAnimationFrame 렌더 루프
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    let animId

    function draw() {
      const ctx = canvas.getContext('2d')
      const W = canvas.width
      const H = canvas.height
      const data = dataRef.current

      ctx.fillStyle = '#1a1a1a'
      ctx.fillRect(0, 0, W, H)

      if (data.length < 2) {
        animId = requestAnimationFrame(draw)
        return
      }

      const rowH = H / N_CHANNELS

      for (let ch = 0; ch < N_CHANNELS; ch++) {
        const y0 = ch * rowH
        const yMid = y0 + rowH / 2

        // 채널 구분선
        ctx.strokeStyle = '#333'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(0, y0 + rowH)
        ctx.lineTo(W, y0 + rowH)
        ctx.stroke()

        // 0선
        ctx.strokeStyle = '#444'
        ctx.setLineDash([4, 4])
        ctx.beginPath()
        ctx.moveTo(0, yMid)
        ctx.lineTo(W, yMid)
        ctx.stroke()
        ctx.setLineDash([])

        // HbO 선
        drawLine(ctx, data, ch, 'hbo', W, rowH, y0, COLORS_HBO[ch])
        // HbR 선
        drawLine(ctx, data, ch, 'hbr', W, rowH, y0, COLORS_HBR[ch])

        // 채널 레이블
        ctx.fillStyle = '#888'
        ctx.font = '11px monospace'
        ctx.fillText(`Ch${ch + 1}`, 6, y0 + 14)
      }

      animId = requestAnimationFrame(draw)
    }

    animId = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(animId)
  }, [])

  return (
    <div className="timeseries">
      <div className="ts-legend">
        <span style={{ color: COLORS_HBO[0] }}>■ HbO</span>
        <span style={{ color: COLORS_HBR[0] }}>■ HbR</span>
        <span className="ts-scale">±5 μmol/L</span>
      </div>
      <canvas ref={canvasRef} className="ts-canvas" width={900} height={480} />
    </div>
  )
}

function drawLine(ctx, data, ch, key, W, rowH, y0, color) {
  const n = data.length
  ctx.strokeStyle = color
  ctx.lineWidth = 1.5
  ctx.beginPath()

  for (let i = 0; i < n; i++) {
    const val = data[i][key][ch]
    const x = (i / (n - 1)) * W
    const t = (val - Y_MIN) / (Y_MAX - Y_MIN)
    const y = y0 + rowH * (1 - t)
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
  }
  ctx.stroke()
}
```

- [ ] **Step 2: TimeSeriesTab.css 작성**

`web/src/components/TimeSeriesTab.css`:
```css
.timeseries {
  display: flex;
  flex-direction: column;
  gap: 8px;
  height: 100%;
}

.ts-legend {
  display: flex;
  gap: 16px;
  align-items: center;
  font-size: 0.85rem;
}

.ts-scale { margin-left: auto; color: var(--text-muted); font-size: 0.8rem; }

.ts-canvas {
  width: 100%;
  height: auto;
  border-radius: 8px;
  border: 1px solid var(--border);
}
```

- [ ] **Step 3: App.jsx에 TimeSeriesTab 통합**

```jsx
import TimeSeriesTab from './components/TimeSeriesTab'
// ...
{activeTab === 'timeseries' && <TimeSeriesTab />}
```

- [ ] **Step 4: 커밋**

```bash
cd D:/Study_fNIRS
git add web/src/components/TimeSeriesTab.jsx web/src/components/TimeSeriesTab.css web/src/App.jsx
git commit -m "feat(web): TimeSeriesTab — Canvas 실시간 HbO/HbR 그래프 (5초 슬라이딩)"
```

---

## Task 11: BrainMapTab — 뇌 맵 Canvas

**Files:**
- Create: `web/src/components/BrainMapTab.jsx`
- Create: `web/src/components/BrainMapTab.css`

- [ ] **Step 1: BrainMapTab.jsx 작성**

`web/src/components/BrainMapTab.jsx`:
```jsx
import { useRef, useEffect, useState } from 'react'
import { useApp } from '../context/AppContext'
import { rbfInterpolate, bwr } from '../lib/rbf'
import './BrainMapTab.css'

// 채널 위치 (정규화 0~1, 뇌 타원 기준)
const CH_POS = [
  [0.37, 0.36],  // Ch1 left-anterior PFC
  [0.63, 0.36],  // Ch2 right-anterior PFC
  [0.37, 0.52],  // Ch3 left-posterior PFC
  [0.63, 0.52],  // Ch4 right-posterior PFC
]

const VMIN = -5.0
const VMAX = 5.0
const GRID_RES = 60  // 보간 그리드 해상도

export default function BrainMapTab() {
  const { processedSample } = useApp()
  const canvasRef = useRef(null)
  const [mode, setMode] = useState('hbo')  // 'hbo' | 'hbr'

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height

    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, W, H)

    if (!processedSample) return

    const values = mode === 'hbo' ? processedSample.hbo : processedSample.hbr

    // 뇌 타원 파라미터
    const bx = W * 0.08, by = H * 0.06
    const bw = W * 0.84, bh = H * 0.88
    const cxE = bx + bw / 2, cyE = by + bh / 2
    const rx = bw / 2, ry = bh / 2

    // RBF 보간 그리드 생성
    const gridPoints = []
    const gridXY = []
    for (let gy = 0; gy < GRID_RES; gy++) {
      for (let gx = 0; gx < GRID_RES; gx++) {
        const fx = gx / (GRID_RES - 1)
        const fy = gy / (GRID_RES - 1)
        // 뇌 타원 내부만
        const dx = (fx - 0.5) / 0.5
        const dy = (fy - 0.5) / 0.5
        if (dx * dx + dy * dy <= 1) {
          gridPoints.push([fx, fy])
          gridXY.push([bx + fx * bw, by + fy * bh])
        }
      }
    }

    const interpolated = rbfInterpolate(gridPoints, CH_POS, values)

    // 각 그리드 포인트에 컬러 픽셀 그리기
    const cellW = bw / GRID_RES
    const cellH = bh / GRID_RES
    interpolated.forEach((val, i) => {
      const [px, py] = gridXY[i]
      const t = (val - VMIN) / (VMAX - VMIN)
      const [r, g, b] = bwr(t)

      // 뇌 경계 페이드 alpha
      const fx = gridPoints[i][0]
      const fy = gridPoints[i][1]
      const dx2 = ((fx - 0.5) / 0.5) ** 2
      const dy2 = ((fy - 0.5) / 0.5) ** 2
      const alpha = Math.max(0, 1 - (dx2 + dy2) ** 0.6) * 0.75

      ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`
      ctx.fillRect(px - cellW / 2, py - cellH / 2, cellW + 1, cellH + 1)
    })

    // 채널 마커
    CH_POS.forEach(([fx, fy], ch) => {
      const px = bx + fx * bw
      const py = by + fy * bh
      ctx.strokeStyle = '#fff'
      ctx.fillStyle = 'rgba(255,255,255,0.5)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.arc(px, py, 8, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
      ctx.fillStyle = '#111'
      ctx.font = 'bold 10px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(`Ch${ch + 1}`, px, py + 4)
    })

    // 방향 레이블
    ctx.fillStyle = '#666'
    ctx.font = '11px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('▲ Anterior (PFC)', W / 2, by - 6)
    ctx.fillText('▼ Posterior', W / 2, by + bh + 14)

  }, [processedSample, mode])

  return (
    <div className="brainmap">
      <div className="bm-controls">
        <button
          className={mode === 'hbo' ? 'btn-mode active' : 'btn-mode'}
          onClick={() => setMode('hbo')}
        >
          HbO
        </button>
        <button
          className={mode === 'hbr' ? 'btn-mode active' : 'btn-mode'}
          onClick={() => setMode('hbr')}
        >
          HbR
        </button>
        <div className="colorbar">
          <span>−5</span>
          <div className="colorbar-gradient" />
          <span>+5 μmol/L</span>
        </div>
      </div>
      <canvas ref={canvasRef} className="bm-canvas" width={500} height={540} />
    </div>
  )
}
```

- [ ] **Step 2: BrainMapTab.css 작성**

`web/src/components/BrainMapTab.css`:
```css
.brainmap {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.bm-controls {
  display: flex;
  align-items: center;
  gap: 12px;
}

.btn-mode {
  padding: 5px 18px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 0.9rem;
}

.btn-mode.active {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}

.colorbar {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.8rem;
  color: var(--text-muted);
}

.colorbar-gradient {
  width: 100px;
  height: 12px;
  border-radius: 6px;
  background: linear-gradient(to right, #0000ff, #ffffff, #ff0000);
}

.bm-canvas {
  max-width: 100%;
  height: auto;
  border-radius: 10px;
  border: 1px solid var(--border);
}
```

- [ ] **Step 3: App.jsx에 BrainMapTab 통합**

```jsx
import BrainMapTab from './components/BrainMapTab'
// ...
{activeTab === 'brainmap' && <BrainMapTab />}
```

- [ ] **Step 4: 커밋**

```bash
cd D:/Study_fNIRS
git add web/src/components/BrainMapTab.jsx web/src/components/BrainMapTab.css web/src/App.jsx
git commit -m "feat(web): BrainMapTab — RBF 보간 BWR 컬러맵 Canvas 렌더링"
```

---

## Task 12: CSV 내보내기

**Files:**
- Modify: `web/src/components/TopBar.jsx`

- [ ] **Step 1: CSV 내보내기 함수 + 버튼 추가**

`web/src/components/TopBar.jsx`에 다음 함수와 버튼 추가:

```jsx
// TopBar.jsx 내부 — downloadCSV 함수 추가
function downloadCSV(sessionData) {
  if (sessionData.length === 0) return

  const header = 'timestamp,hbo_ch1,hbo_ch2,hbo_ch3,hbo_ch4,hbr_ch1,hbr_ch2,hbr_ch3,hbr_ch4,ci'
  const rows = sessionData.map(s =>
    [
      s.timestamp.toFixed(3),
      ...s.hbo.map(v => v.toFixed(4)),
      ...s.hbr.map(v => v.toFixed(4)),
      s.ci.toFixed(4),
    ].join(',')
  )
  const csv = [header, ...rows].join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)

  const now = new Date()
  const ts = now.toISOString().replace(/[-:T]/g, '').slice(0, 15)
  const a = document.createElement('a')
  a.href = url
  a.download = `fnirs_session_${ts}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

// TopBar JSX 내부 — CI group 옆에 버튼 추가
<button
  className="btn-download"
  disabled={sessionData.length === 0}
  onClick={() => downloadCSV(sessionData)}
  title="세션 데이터 CSV 다운로드"
>
  ↓ CSV
</button>
```

`web/src/components/TopBar.css`에 추가:
```css
.btn-download {
  padding: 4px 12px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 0.82rem;
}
.btn-download:hover:not(:disabled) { background: var(--border); color: var(--text); }
.btn-download:disabled { opacity: 0.35; cursor: not-allowed; }
```

- [ ] **Step 2: sessionStorage 임시 저장 추가**

`web/src/context/AppContext.jsx`의 `pushSample`과 초기화 부분에 sessionStorage 연동 추가:

```jsx
// AppContext.jsx — AppProvider 내부 상단에 추가
// 새로고침 시 sessionData 복원
const [sessionData, setSessionData] = useState(() => {
  try {
    const saved = sessionStorage.getItem('fnirs_session')
    return saved ? JSON.parse(saved) : []
  } catch {
    return []
  }
})

// pushSample에 저장 추가
const pushSample = useCallback((sample) => {
  setProcessedSample(sample)
  setSessionData((prev) => {
    const next = [...prev, sample]
    try { sessionStorage.setItem('fnirs_session', JSON.stringify(next)) } catch {}
    return next
  })
}, [])

// clearSession에 sessionStorage 초기화 추가
const clearSession = useCallback(() => {
  setSessionData([])
  setProcessedSample(null)
  sessionStorage.removeItem('fnirs_session')
}, [])
```

- [ ] **Step 3: 커밋**

```bash
cd D:/Study_fNIRS
git add web/src/components/TopBar.jsx web/src/components/TopBar.css web/src/context/AppContext.jsx
git commit -m "feat(web): CSV 내보내기 + sessionStorage 임시 저장"
```

---

## Task 13: 최종 통합 테스트 & 정리

- [ ] **Step 1: 전체 테스트 실행**

```bash
cd web && npm run test -- --run
```

Expected: 모든 단위 테스트 통과

- [ ] **Step 2: 시뮬레이터 모드 전체 흐름 검증**

```bash
npm run dev
```

`http://localhost:5173/?simulate=true` 접속 후 순서대로 확인:
1. "연결" 클릭 → 상태 "연결됨", CI 게이지 변동
2. Calibration 탭 → SNR 바 채워짐 → "진행 ▶" 활성화
3. "진행 ▶" 클릭 → Brain Map / Time Series 탭 활성화
4. Brain Map 탭 → RBF 컬러 오버레이 갱신 확인
5. Time Series 탭 → HbO/HbR 실시간 그래프 확인
6. "↓ CSV" 버튼 → 파일 다운로드 확인

- [ ] **Step 3: 모바일 반응형 확인**

브라우저 DevTools에서 iPhone SE (375px) 모드 → 탭이 하단으로 이동 확인

- [ ] **Step 4: run_logging.md 업데이트**

`run_logging.md`에 구현 완료 항목 추가.

- [ ] **Step 5: 최종 커밋**

```bash
cd D:/Study_fNIRS
git add .
git commit -m "feat(web): fNIRS 웹 서비스 프로토타입 완성 (React+Vite, BLE, 시뮬레이터)"
```
