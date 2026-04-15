import { useState, useEffect, useRef } from 'react'
import { useApp } from '../context/AppContext'
import BrainModelPNG from './BrainModel/BrainModelPNG'
import { BRAIN_FRONT } from './BrainModel/brainViews'
import './BrainMapTab.css'

const SAMPLING_HZ  = 10
const SMOOTH_SEC   = 5
const SMOOTH_N     = SAMPLING_HZ * SMOOTH_SEC  // 50 샘플 = 5초

export default function BrainMapTab() {
  const { processedSample, baseline } = useApp()
  const [mode, setMode] = useState('hbo')  // 'hbo' | 'hbr'

  // ── 5초 슬라이딩 버퍼 (집중도 스무딩) ──────────────────────────
  const deltaBufferRef = useRef([])   // 최근 SMOOTH_N개의 meanDelta 보관
  const [focusLevel, setFocusLevel] = useState('보통')

  useEffect(() => {
    if (!processedSample) return

    // 채널 평균 HbO delta (현재 샘플)
    const delta = processedSample.hbo.reduce(
      (s, v, i) => s + (v - baseline.hbo[i]), 0
    ) / processedSample.hbo.length

    // 버퍼에 추가 (최대 SMOOTH_N개 유지)
    const buf = deltaBufferRef.current
    deltaBufferRef.current = buf.length >= SMOOTH_N
      ? [...buf.slice(1), delta]
      : [...buf, delta]

    // 버퍼 평균으로 집중도 판정
    const smoothed = deltaBufferRef.current.reduce((s, v) => s + v, 0)
                   / deltaBufferRef.current.length
    const level = smoothed > 0.3 ? '높음' : smoothed > -0.1 ? '보통' : '낮음'
    setFocusLevel(level)
  }, [processedSample, baseline])
  // ────────────────────────────────────────────────────────────────

  const rawValues = processedSample
    ? (mode === 'hbo' ? processedSample.hbo : processedSample.hbr)
    : [0, 0, 0, 0]
  const baselineVals = mode === 'hbo' ? baseline.hbo : baseline.hbr
  const values = rawValues.map((v, i) => v - baselineVals[i])

  return (
    <div className="brainmap">
      <div className="bm-controls">
        <button className={mode === 'hbo' ? 'btn-mode active' : 'btn-mode'} onClick={() => setMode('hbo')}>HbO</button>
        <button className={mode === 'hbr' ? 'btn-mode active' : 'btn-mode'} onClick={() => setMode('hbr')}>HbR</button>

        <div className="bm-divider" />

        <div className="colorbar">
          <span>−5</span>
          <div className="colorbar-gradient" />
          <span>+5</span>
        </div>
      </div>

      <div className="bm-wrap">
        <BrainModelPNG view={BRAIN_FRONT} values={values} focusLabel={focusLevel} />
      </div>

      <p className="bm-caption">
        전전두엽(PFC) {mode === 'hbo' ? 'HbO' : 'HbR'} — baseline 대비: Blue(↓낮음) → White(기준) → Red(↑높음)
      </p>
    </div>
  )
}
