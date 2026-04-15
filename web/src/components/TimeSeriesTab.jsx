import { useRef, useEffect } from 'react'
import { useApp } from '../context/AppContext'
import './TimeSeriesTab.css'

const N_CHANNELS  = 4
const COLOR_HBO   = '#ef4444'
const COLOR_HBR   = '#4e7aff'
const MAX_POINTS  = 300        // 표시할 최대 샘플 수 (10Hz × 30초)
const SAMPLING_HZ = 10
const PADDING_RATIO = 0.15     // 상하 여백 비율

// 축 여백 (pixel, 논리 좌표계 기준)
const ML = 52   // left  — Y축 레이블
const MR = 12   // right
const MT = 12   // top
const MB = 32   // bottom — X축 레이블

const CH_LABEL = ['좌전방 PFC', '우전방 PFC', '좌후방 PFC', '우후방 PFC']

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
  return { yMin: lo - span * PADDING_RATIO, yMax: hi + span * PADDING_RATIO }
}

// 보기 좋은 눈금 간격 계산
function niceStep(range, targetTicks = 4) {
  const raw = range / targetTicks
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const norm = raw / mag
  const nice = norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10
  return nice * mag
}

function ChannelCanvas({ ch, dataRef }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    let animId

    function draw() {
      const dpr = window.devicePixelRatio || 1
      const displayW = canvas.clientWidth
      const displayH = canvas.clientHeight

      // 캔버스 내부 해상도를 DPR에 맞게 조정
      if (canvas.width !== displayW * dpr || canvas.height !== displayH * dpr) {
        canvas.width  = displayW * dpr
        canvas.height = displayH * dpr
      }

      const ctx = canvas.getContext('2d')
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

      const W = displayW
      const H = displayH

      // 플롯 영역
      const pw = W - ML - MR
      const ph = H - MT - MB

      // 배경
      ctx.fillStyle = '#f8fafc'
      ctx.fillRect(0, 0, W, H)

      // 플롯 배경
      ctx.fillStyle = '#ffffff'
      ctx.fillRect(ML, MT, pw, ph)

      const data = dataRef.current
      const { yMin, yMax } = calcRange(data, ch)
      const range = yMax - yMin

      // ── Y축 그리드 + 레이블 ──────────────────────────────
      const yStep = niceStep(range, 4)
      const yStart = Math.ceil(yMin / yStep) * yStep

      ctx.font = `${11 / dpr + 11}px Inter, system-ui, monospace`
      ctx.font = '11px Inter, system-ui, monospace'
      ctx.textAlign = 'right'
      ctx.textBaseline = 'middle'

      for (let yv = yStart; yv <= yMax + 1e-9; yv += yStep) {
        const py = MT + ph * (1 - (yv - yMin) / range)
        if (py < MT - 1 || py > MT + ph + 1) continue

        // 그리드 라인
        ctx.strokeStyle = '#e2e8f0'
        ctx.lineWidth = 1
        ctx.setLineDash([])
        ctx.beginPath()
        ctx.moveTo(ML, py)
        ctx.lineTo(ML + pw, py)
        ctx.stroke()

        // 레이블
        ctx.fillStyle = '#64748b'
        ctx.fillText(yv.toFixed(2), ML - 6, py)
      }

      // 0선 강조
      if (yMin < 0 && yMax > 0) {
        const py = MT + ph * (1 - (0 - yMin) / range)
        ctx.strokeStyle = '#cbd5e1'
        ctx.lineWidth = 1.5
        ctx.setLineDash([5, 3])
        ctx.beginPath()
        ctx.moveTo(ML, py)
        ctx.lineTo(ML + pw, py)
        ctx.stroke()
        ctx.setLineDash([])
      }

      // ── X축 그리드 + 레이블 ──────────────────────────────
      const totalSec = MAX_POINTS / SAMPLING_HZ          // 30초
      const xTickSec = totalSec <= 30 ? 5 : 10           // 5초 또는 10초 간격
      const nTicks = Math.floor(totalSec / xTickSec)

      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'

      for (let i = 0; i <= nTicks; i++) {
        const sec = i * xTickSec
        const px = ML + (sec / totalSec) * pw
        const label = i === nTicks ? '0s' : `-${totalSec - sec}s`

        // 그리드 라인
        ctx.strokeStyle = '#e2e8f0'
        ctx.lineWidth = 1
        ctx.setLineDash([])
        ctx.beginPath()
        ctx.moveTo(px, MT)
        ctx.lineTo(px, MT + ph)
        ctx.stroke()

        // 레이블
        ctx.fillStyle = '#64748b'
        ctx.fillText(label, px, MT + ph + 6)
      }

      // ── 축 테두리 ────────────────────────────────────────
      ctx.strokeStyle = '#cbd5e1'
      ctx.lineWidth = 1.5
      ctx.setLineDash([])
      ctx.strokeRect(ML, MT, pw, ph)

      // ── 데이터 라인 ──────────────────────────────────────
      const drawLine = (key, color) => {
        if (data.length < 2) return
        ctx.strokeStyle = color
        ctx.lineWidth = 1.8
        ctx.beginPath()
        for (let i = 0; i < data.length; i++) {
          const val = data[i][key][ch]
          const x = ML + (i / (MAX_POINTS - 1)) * pw
          const y = MT + ph * (1 - (val - yMin) / range)
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
        }
        ctx.stroke()
      }

      drawLine('hbo', COLOR_HBO)
      drawLine('hbr', COLOR_HBR)

      // ── Y축 단위 레이블 ──────────────────────────────────
      ctx.save()
      ctx.translate(12, MT + ph / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillStyle = '#94a3b8'
      ctx.font = '10px Inter, system-ui'
      ctx.fillText('μmol/L', 0, 0)
      ctx.restore()

      animId = requestAnimationFrame(draw)
    }

    animId = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(animId)
  }, [ch, dataRef])

  return (
    <div className="ts-card">
      <div className="ts-card-label">Ch {ch + 1} &nbsp;<span className="ts-ch-region">{CH_LABEL[ch]}</span></div>
      <canvas ref={canvasRef} className="ts-canvas" />
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
        <span style={{ color: COLOR_HBO }}>■ HbO (산화헤모글로빈)</span>
        <span style={{ color: COLOR_HBR }}>■ HbR (환원헤모글로빈)</span>
        <span className="ts-scale">최근 30초 · auto scale</span>
      </div>
      <div className="ts-grid">
        {Array.from({ length: N_CHANNELS }, (_, ch) => (
          <ChannelCanvas key={ch} ch={ch} dataRef={dataRef} />
        ))}
      </div>
    </div>
  )
}
