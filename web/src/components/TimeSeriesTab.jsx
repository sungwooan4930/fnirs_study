import { useRef, useEffect } from 'react'
import { useApp } from '../context/AppContext'
import './TimeSeriesTab.css'

const N_CHANNELS = 4
const COLOR_HBO = '#ef4444'
const COLOR_HBR = '#4e92ff'
const MAX_POINTS = 500  // 5초 × 100Hz
const PADDING = 0.12    // 상하 여백 비율

// 채널 ch의 HbO+HbR 전체에서 min/max 계산, 데이터 없으면 ±1 기본
function calcRange(data, ch) {
  if (data.length < 2) return { yMin: -1, yMax: 1 }
  let lo = Infinity, hi = -Infinity
  for (const d of data) {
    if (d.hbo[ch] < lo) lo = d.hbo[ch]
    if (d.hbo[ch] > hi) hi = d.hbo[ch]
    if (d.hbr[ch] < lo) lo = d.hbr[ch]
    if (d.hbr[ch] > hi) hi = d.hbr[ch]
  }
  const span = hi - lo || 1
  return { yMin: lo - span * PADDING, yMax: hi + span * PADDING }
}

function drawLine(ctx, data, ch, key, W, H, color, yMin, yMax) {
  const n = data.length
  if (n < 2) return
  const range = yMax - yMin
  ctx.strokeStyle = color
  ctx.lineWidth = 1.5
  ctx.beginPath()
  for (let i = 0; i < n; i++) {
    const val = data[i][key][ch]
    const x = (i / (n - 1)) * W
    const y = H * (1 - (val - yMin) / range)
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
  }
  ctx.stroke()
}

function ChannelCanvas({ ch, dataRef }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    let animId

    function draw() {
      const ctx = canvas.getContext('2d')
      const W = canvas.width
      const H = canvas.height
      const data = dataRef.current

      ctx.fillStyle = '#0f1117'
      ctx.fillRect(0, 0, W, H)

      const { yMin, yMax } = calcRange(data, ch)
      const range = yMax - yMin

      // 0선 (범위 내에 있을 때만)
      if (yMin < 0 && yMax > 0) {
        const yZero = H * (1 - (0 - yMin) / range)
        ctx.strokeStyle = '#1e2130'
        ctx.setLineDash([4, 4])
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(0, yZero)
        ctx.lineTo(W, yZero)
        ctx.stroke()
        ctx.setLineDash([])
      }

      drawLine(ctx, data, ch, 'hbo', W, H, COLOR_HBO, yMin, yMax)
      drawLine(ctx, data, ch, 'hbr', W, H, COLOR_HBR, yMin, yMax)

      // Y축 범위 레이블
      if (data.length > 1) {
        ctx.fillStyle = '#4b5563'
        ctx.font = '9px Inter, monospace'
        ctx.textAlign = 'right'
        ctx.fillText(yMax.toFixed(1), W - 3, 10)
        ctx.fillText(yMin.toFixed(1), W - 3, H - 3)
      }

      animId = requestAnimationFrame(draw)
    }

    animId = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(animId)
  }, [ch, dataRef])

  return (
    <div className="ts-card">
      <div className="ts-card-label">Ch {ch + 1}</div>
      <canvas ref={canvasRef} className="ts-canvas" width={440} height={180} />
    </div>
  )
}

export default function TimeSeriesTab() {
  const { sessionData } = useApp()
  const dataRef = useRef([])

  useEffect(() => {
    dataRef.current = sessionData.slice(-MAX_POINTS)
  }, [sessionData])

  return (
    <div className="timeseries">
      <div className="ts-legend">
        <span style={{ color: COLOR_HBO }}>■ HbO</span>
        <span style={{ color: COLOR_HBR }}>■ HbR</span>
        <span className="ts-scale">auto scale · μmol/L</span>
      </div>
      <div className="ts-grid">
        {Array.from({ length: N_CHANNELS }, (_, ch) => (
          <ChannelCanvas key={ch} ch={ch} dataRef={dataRef} />
        ))}
      </div>
    </div>
  )
}
