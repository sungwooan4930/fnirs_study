# fNIRS 웹 UI 전면 재설계 스펙

**날짜:** 2026-04-13
**범위:** `web/src/` 전체 컴포넌트 CSS + App.css
**방식:** CSS 변수 교체 (아키텍처 변경 없음)

---

## 1. 디자인 시스템 기반

Lovable 디자인 시스템 (`lovable-DESIGN.md`, `lovable-design-system-compact.md`) 기반.
라이트 테마 그대로 적용하지 않고 **Dark + Lovable Blue 하이브리드**로 적용.

---

## 2. 색상 토큰 (CSS Variables)

`web/src/App.css`의 `:root` 블록 전체 교체.

| 변수 | 현재 | 신규 | 용도 |
|------|------|------|------|
| `--bg` | `#1a1a1a` | `#0f1117` | 전체 배경 (네이비 블랙) |
| `--surface` | `#242424` | `#1a1f35` | 카드/컨테이너 (블루 틴트 다크) |
| `--surface-2` | — | `#212840` | 중첩 서피스 (신규) |
| `--border` | `#333` | `#1e2130` | 기본 테두리 |
| `--border-accent` | — | `#1f55f1` | 강조 테두리 (신규, TopBar 하단 등) |
| `--text` | `#e0e0e0` | `#e8eaf0` | 기본 텍스트 |
| `--text-muted` | `#888` | `#6b7280` | 보조 텍스트 |
| `--accent` | `#4a90d9` | `#1f55f1` | Electric Blue (Lovable primary) |
| `--accent-2` | — | `#4e92ff` | 보조 블루 (그라데이션 끝, 신규) |
| `--accent-hover` | `#5aa0e9` | `#3b6ef0` | 액센트 hover |
| `--success` | `#4caf50` | `#22c55e` | Good 상태 |
| `--error` | `#f44336` | `#ef4444` | 에러/Poor 상태 |
| `--warning` | `#ff9800` | `#f59e0b` | 경고/측정 중 |

---

## 3. 타이포그래피

- **폰트**: Inter Variable (Google Fonts에서 사전 다운로드 → `web/public/fonts/` 저장)
- **@font-face**: `App.css` 상단에 선언, `font-display: swap`
- **font-family**: `'Inter', system-ui, sans-serif` (전역 적용)
- **다운로드 대상**: `Inter-Variable.woff2` (전체 weight 포함 variable font)

---

## 4. 공통 스타일 변경

### 버튼
- 기존: `border-radius: 6px`
- 신규: `border-radius: 20px` (pill 형태)
- 활성 버튼: `background: var(--accent)`, `color: #fff`
- 비활성 버튼: `border: 1px solid var(--border)`, `color: var(--text-muted)`

### 카드
- `border-radius: 10px` 유지
- `border: 1px solid var(--border)` → 신규 토큰 적용
- 활성 채널 카드(Good): `border-color: var(--success)` (기존과 동일, 색상만 교체)

### 탭 네비게이션
- 탭 버튼 → pill 형태 (`border-radius: 20px`)
- 활성 탭: `background: var(--accent)`
- 비활성 탭: `border: 1px solid var(--border)`

---

## 5. 컴포넌트별 변경

### 5.1 TopBar
- 배경: `var(--surface)` + 하단 border: `1px solid var(--border-accent)` (Electric Blue 라인)
- 연결 버튼: pill, `var(--accent)`
- 해제 버튼: pill, `#ef444420` 배경 + `#ef4444` 텍스트
- CI 게이지 fill: `linear-gradient(to right, var(--accent), var(--accent-2))`
- CSV 버튼: 투명 배경, `border: 1px solid var(--border)`

### 5.2 CalibrationTab
- 채널 카드: `var(--surface)` 배경
- SNR 바 fill: `linear-gradient(to right, var(--accent), var(--accent-2))`
- Good 뱃지: `background: #22c55e20`, `color: var(--success)`, `border: 1px solid #22c55e40`
- 측정 중 뱃지: `background: var(--surface-2)`, `color: var(--text-muted)`
- 진행 버튼: pill, `var(--accent)`

### 5.3 BrainMapTab (주요 변경)

**기존:** 타원 2D 평면에 RBF 컬러맵, 위에서 내려다보는 시점.

**신규:**
- 배경 이미지: `refer/3d brain.png` → `web/public/brain.png`로 복사해서 사용
- Canvas 오버레이를 이미지 위에 `position: absolute`로 배치
- **채널 마커 제거** — 색상 블롭만으로 신호 강도 표시
- 각 채널 위치에 `radialGradient` 블롭 (RBF 보간 결과를 Gaussian 블롭으로 렌더)
  - 양수(HbO↑) → Red(`#ef4444`)
  - 음수(HbO↓) → Blue(`#4e92ff`)
  - 0 근방 → White(`#ffffff`)
- 블롭 alpha: `0.65~0.75` (뇌 이미지가 투시되도록)
- HbO/HbR 토글 버튼: pill 형태
- 컬러바: `−5 → White → +5 μmol/L`

**채널 위치 (이미지 기준, 정규화 0~1):**
| 채널 | cx | cy | 설명 |
|------|----|----|------|
| Ch1 | 0.36 | 0.52 | 좌측 전방 PFC |
| Ch2 | 0.64 | 0.52 | 우측 전방 PFC |
| Ch3 | 0.38 | 0.65 | 좌측 후방 PFC |
| Ch4 | 0.62 | 0.65 | 우측 후방 PFC |

### 5.4 TimeSeriesTab (주요 변경)

**기존:** 4채널 모두 하나의 Canvas에 겹쳐서 표시 (행 분리).

**신규:** 2×2 그리드, 채널별 독립 Canvas 카드
- 레이아웃: `display: grid; grid-template-columns: 1fr 1fr; gap: 10px`
- 각 카드: `var(--surface)` 배경, `border-radius: 10px`
- 카드 상단: 채널명 (`Ch 1` ~ `Ch 4`), `color: var(--accent-2)`
- Canvas 배경: `var(--bg)` (`#0f1117`)
- 0선: `var(--border)` 점선
- HbO: `#ef4444`, HbR: `#4e92ff`
- 모바일 (max-width: 600px): `grid-template-columns: 1fr` (4행 1열)

---

## 6. 파일 변경 목록

| 파일 | 변경 내용 |
|------|-----------|
| `web/public/fonts/Inter-Variable.woff2` | 신규 생성 (다운로드) |
| `web/public/brain.png` | 신규 생성 (`refer/3d brain.png` 복사) |
| `web/src/App.css` | CSS 변수 전체 교체, @font-face 추가, 탭 pill 스타일 |
| `web/src/index.css` | font-family 업데이트 |
| `web/src/components/TopBar.css` | 버튼 pill, border-accent, CI 그라데이션 |
| `web/src/components/CalibrationTab.css` | 버튼 pill, 바 그라데이션, 뱃지 스타일 |
| `web/src/components/BrainMapTab.jsx` | Canvas 렌더 로직 — 블롭 방식으로 교체, 마커 제거 |
| `web/src/components/BrainMapTab.css` | 이미지+캔버스 레이아웃 |
| `web/src/components/TimeSeriesTab.jsx` | 단일 Canvas → 4개 Canvas (채널별) |
| `web/src/components/TimeSeriesTab.css` | 2×2 그리드 레이아웃 |

---

## 7. 구현 순서

1. Inter 폰트 다운로드 → `web/public/fonts/`
2. `brain.png` 복사 → `web/public/`
3. `App.css` CSS 변수 + @font-face + 탭 스타일 교체
4. `index.css` font-family 업데이트
5. `TopBar.css` 업데이트
6. `CalibrationTab.css` 업데이트
7. `BrainMapTab.jsx` + `BrainMapTab.css` 업데이트 (블롭 렌더, 마커 제거)
8. `TimeSeriesTab.jsx` + `TimeSeriesTab.css` 업데이트 (2×2 그리드)
9. 전체 시각 검증 (`?simulate=true`)
