import { useRef, useEffect } from 'react'
import { useApp } from '../context/AppContext'
import './TimeSeriesTab.css'

const N_CHANNELS = 4
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
