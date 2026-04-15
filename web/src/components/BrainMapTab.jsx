import { useState, lazy, Suspense } from 'react'
import { useApp } from '../context/AppContext'
import BrainModelPNG from './BrainModel/BrainModelPNG'
import { BRAIN_FRONT } from './BrainModel/brainViews'
import './BrainMapTab.css'

// Three.js는 무겁기 때문에 lazy load
const BrainModel3D = lazy(() => import('./BrainModel/BrainModel3D'))

export default function BrainMapTab() {
  const { processedSample } = useApp()
  const [mode, setMode] = useState('hbo')      // 'hbo' | 'hbr'
  const [render, setRender] = useState('png')  // 'png' | '3d'
  const [view3d, setView3d] = useState('front') // 'front' | 'top'

  const values = processedSample
    ? (mode === 'hbo' ? processedSample.hbo : processedSample.hbr)
    : [0, 0, 0, 0]

  return (
    <div className="brainmap">
      <div className="bm-controls">
        {/* HbO / HbR */}
        <button className={mode === 'hbo' ? 'btn-mode active' : 'btn-mode'} onClick={() => setMode('hbo')}>HbO</button>
        <button className={mode === 'hbr' ? 'btn-mode active' : 'btn-mode'} onClick={() => setMode('hbr')}>HbR</button>

        <div className="bm-divider" />

        {/* PNG / 3D */}
        <button className={render === 'png' ? 'btn-mode active' : 'btn-mode'} onClick={() => setRender('png')}>2D</button>
        <button className={render === '3d'  ? 'btn-mode active' : 'btn-mode'} onClick={() => setRender('3d')}>3D</button>

        {/* 3D 뷰 프리셋 */}
        {render === '3d' && (
          <>
            <div className="bm-divider" />
            <button className={view3d === 'front' ? 'btn-mode active' : 'btn-mode'} onClick={() => setView3d('front')}>Front</button>
            <button className={view3d === 'top'   ? 'btn-mode active' : 'btn-mode'} onClick={() => setView3d('top')}>Top</button>
          </>
        )}

        <div className="colorbar">
          <span>−5</span>
          <div className="colorbar-gradient" />
          <span>+5 μmol/L</span>
        </div>
      </div>

      <div className="bm-wrap">
        {render === 'png' ? (
          <BrainModelPNG view={BRAIN_FRONT} values={values} />
        ) : (
          <Suspense fallback={<div className="bm-3d-loading">3D 모델 로딩 중…</div>}>
            <BrainModel3D values={values} view={view3d} />
          </Suspense>
        )}
      </div>

      <p className="bm-caption">
        전전두엽(PFC) {mode === 'hbo' ? 'HbO' : 'HbR'} 신호 강도 — Blue(음) → White(0) → Red(양)
        {render === '3d' && ' · 마우스로 회전/줌 가능'}
      </p>
    </div>
  )
}
