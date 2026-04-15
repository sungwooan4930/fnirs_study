import { useState } from 'react'
import { useApp } from '../context/AppContext'
import BrainModelPNG from './BrainModel/BrainModelPNG'
import { BRAIN_FRONT } from './BrainModel/brainViews'
import './BrainMapTab.css'

export default function BrainMapTab() {
  const { processedSample, baseline } = useApp()
  const [mode, setMode] = useState('hbo')  // 'hbo' | 'hbr'

  const rawValues = processedSample
    ? (mode === 'hbo' ? processedSample.hbo : processedSample.hbr)
    : [0, 0, 0, 0]
  const baselineVals = mode === 'hbo' ? baseline.hbo : baseline.hbr
  const values = rawValues.map((v, i) => v - baselineVals[i])

  // 집중도: 전전두엽 HbO 델타 평균으로 산출 (HbO 증가 = 인지 부하 상승)
  const hboDeltas = processedSample
    ? processedSample.hbo.map((v, i) => v - baseline.hbo[i])
    : [0, 0, 0, 0]
  const meanDelta = hboDeltas.reduce((s, v) => s + v, 0) / hboDeltas.length
  const focusLevel = meanDelta > 0.3 ? '높음' : meanDelta > -0.1 ? '보통' : '낮음'

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
