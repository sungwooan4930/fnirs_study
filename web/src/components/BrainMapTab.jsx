import { useState } from 'react'
import { useApp } from '../context/AppContext'
import './BrainMapTab.css'

// 채널 위치 — SVG viewBox(340×290) 기준, 퍼센트(cx/cy)
const CH_GRAD = [
  { id: 'g1', cx: '36%', cy: '52%', r: '18%' },  // Ch1 좌측 전방 PFC
  { id: 'g2', cx: '64%', cy: '52%', r: '18%' },  // Ch2 우측 전방 PFC
  { id: 'g3', cx: '38%', cy: '65%', r: '15%' },  // Ch3 좌측 후방 PFC
  { id: 'g4', cx: '62%', cy: '65%', r: '15%' },  // Ch4 우측 후방 PFC
]

const VMIN = -5.0
const VMAX = 5.0

// 신호 값 → 색상 (Blue → White → Red, BWR)
function valToColor(val) {
  const t = Math.max(0, Math.min(1, (val - VMIN) / (VMAX - VMIN)))
  if (t < 0.5) {
    const f = t * 2
    const r = Math.round(78 + (255 - 78) * f)
    const g = Math.round(146 + (255 - 146) * f)
    return `rgb(${r},${g},255)`
  }
  const f = (t - 0.5) * 2
  const gb = Math.round(255 * (1 - f))
  return `rgb(255,${gb},${gb})`
}

// 신호 값 → opacity (절대값이 클수록 진하게)
function valToOpacity(val) {
  const t = Math.abs((val - (VMIN + VMAX) / 2) / ((VMAX - VMIN) / 2))
  return 0.35 + Math.min(t, 1) * 0.4
}

export default function BrainMapTab() {
  const { processedSample } = useApp()
  const [mode, setMode] = useState('hbo')

  const values = processedSample
    ? (mode === 'hbo' ? processedSample.hbo : processedSample.hbr)
    : [0, 0, 0, 0]

  return (
    <div className="brainmap">
      <div className="bm-controls">
        <button
          className={mode === 'hbo' ? 'btn-mode active' : 'btn-mode'}
          onClick={() => setMode('hbo')}
        >HbO</button>
        <button
          className={mode === 'hbr' ? 'btn-mode active' : 'btn-mode'}
          onClick={() => setMode('hbr')}
        >HbR</button>
        <div className="colorbar">
          <span>−5</span>
          <div className="colorbar-gradient" />
          <span>+5 μmol/L</span>
        </div>
      </div>

      <div className="bm-wrap">
        <img
          src="/brain.png"
          className="bm-brain-img"
          alt="3D brain top-down view"
          draggable={false}
        />
        <svg
          className="bm-overlay"
          viewBox="0 0 340 290"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            {CH_GRAD.map(({ id, cx, cy, r }, ch) => {
              const color = valToColor(values[ch])
              const opacity = valToOpacity(values[ch])
              return (
                <radialGradient key={id} id={id} cx={cx} cy={cy} r={r}>
                  <stop offset="0%"   stopColor={color} stopOpacity={opacity} />
                  <stop offset="100%" stopColor={color} stopOpacity={0} />
                </radialGradient>
              )
            })}
          </defs>
          {CH_GRAD.map(({ id }) => (
            <rect key={id} width="340" height="290" fill={`url(#${id})`} />
          ))}
          {/* Reset 버튼 UI 가리기 */}
          <rect x="100" y="256" width="140" height="34" fill="#000" />
        </svg>
      </div>

      <p className="bm-caption">
        전전두엽(PFC) {mode === 'hbo' ? 'HbO' : 'HbR'} 신호 강도 — Blue(음) → White(0) → Red(양)
      </p>
    </div>
  )
}
