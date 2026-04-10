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
