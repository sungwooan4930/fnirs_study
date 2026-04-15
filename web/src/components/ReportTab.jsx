import { useState, useMemo, useRef, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { useApp } from '../context/AppContext'
import BrainModelPNG from './BrainModel/BrainModelPNG'
import { BRAIN_FRONT } from './BrainModel/brainViews'
import ReportPDFTemplate from './ReportPDFTemplate'
import { generatePDF } from '../lib/generatePDF'
import './ReportTab.css'

// 채널 → 뇌 부위 매핑
const REGION = ['좌전방 PFC', '우전방 PFC', '좌후방 PFC', '우후방 PFC']
const REGION_SHORT = ['좌전방', '우전방', '좌후방', '우후방']

function sessionDuration(data) {
  if (data.length < 2) return 0
  return data[data.length - 1].timestamp - data[0].timestamp
}

function calcScore(data) {
  if (data.length === 0) return 0
  const avg = data.reduce((s, d) => s + d.ci, 0) / data.length
  return Math.round(Math.max(0, Math.min(1, avg)) * 100)
}

function channelAverages(data) {
  if (data.length === 0) return Array.from({ length: 4 }, () => ({ hbo: 0, hbr: 0 }))
  return Array.from({ length: 4 }, (_, ch) => ({
    hbo: data.reduce((s, d) => s + d.hbo[ch], 0) / data.length,
    hbr: data.reduce((s, d) => s + d.hbr[ch], 0) / data.length,
  }))
}

// 채널 평균에서 뇌 부위별 분석 텍스트 생성 (일반인 대상, 수치 미표기)
function buildAnalysis(avgs) {
  const leftAvg  = (avgs[0].hbo + avgs[2].hbo) / 2   // 좌전방 + 좌후방
  const rightAvg = (avgs[1].hbo + avgs[3].hbo) / 2   // 우전방 + 우후방
  const diff = leftAvg - rightAvg
  const thr = 0.3

  const leftText = Math.abs(diff) < thr
    ? '좌측 전전두엽(좌전방·좌후방 PFC)이 우측과 비슷한 수준으로 고르게 활성화되었습니다.'
    : diff > 0
      ? '좌측 전전두엽(좌전방·좌후방 PFC)이 우측보다 더 활발하게 활성화되었습니다. 좌측 전전두엽은 언어 처리 및 논리적 사고와 관련된 영역으로, 분석적 집중 또는 언어 기반 작업에 몰입한 상태로 볼 수 있습니다.'
      : '좌측 전전두엽(좌전방·좌후방 PFC)이 우측에 비해 상대적으로 낮은 활성도를 보였습니다.'

  const rightText = Math.abs(diff) < thr
    ? '우측 전전두엽(우전방·우후방 PFC)이 좌측과 고른 활성화를 보였습니다.'
    : diff > 0
      ? '우측 전전두엽(우전방·우후방 PFC)이 좌측에 비해 상대적으로 낮은 활성도를 보였습니다. 우측 전전두엽은 공간 인식 및 창의적 사고와 연관된 영역으로, 이번 세션에서는 좌측 우세 패턴이 나타났습니다.'
      : '우측 전전두엽(우전방·우후방 PFC)이 좌측보다 더 활발하게 활성화되었습니다. 우측 전전두엽은 공간 인식 및 창의적 사고와 연관된 영역으로, 이번 세션에서는 우측 우세 패턴이 나타났습니다.'

  const dominance = Math.abs(diff) < thr
    ? '좌우 전전두엽이 균형 있게 활성화된 패턴으로, 전반적으로 안정된 집중 상태를 유지한 것으로 판단됩니다.'
    : diff > 0
      ? '좌측 전전두엽이 더 활발하게 반응한 패턴으로, 논리적·언어적 과제에 집중하는 상태에서 자주 나타나는 경향입니다.'
      : '우측 전전두엽이 더 활발하게 반응한 패턴으로, 공간적·직관적 사고가 활발하게 이루어진 상태로 볼 수 있습니다.'

  return { leftText, rightText, dominance }
}

// 채널별 개별 분석 텍스트 (뇌 부위 박스용)
const REGION_DETAIL = [
  { name: '좌측 전두엽 전방', role: '언어 처리·논리적 사고' },
  { name: '우측 전두엽 전방', role: '직관·창의적 사고' },
  { name: '좌측 전두엽 후방', role: '작업 기억·계획 수립' },
  { name: '우측 전두엽 후방', role: '공간 인식·정서 조절' },
]

function buildChannelAnalysis(avgs) {
  const hboVals = avgs.map(a => a.hbo)
  const mean = hboVals.reduce((s, v) => s + v, 0) / hboVals.length
  const maxDev = Math.max(...hboVals.map(v => Math.abs(v - mean))) || 1

  return avgs.map((avg, ch) => {
    const rel = (avg.hbo - mean) / maxDev
    const level = rel > 0.25 ? 'high' : rel < -0.25 ? 'low' : 'normal'
    const detail = REGION_DETAIL[ch]
    const text =
      level === 'high'
        ? `${detail.role}와 관련된 이 영역이 세션 중 활발하게 반응하였습니다. 다른 부위보다 혈류량이 더 많이 증가한 것으로 나타났습니다.`
        : level === 'low'
        ? `이 영역의 활성도는 상대적으로 낮았습니다. ${detail.role} 관련 부하가 다른 부위보다 적었던 것으로 볼 수 있습니다.`
        : `${detail.role}와 관련된 이 영역이 안정적인 수준으로 고르게 활성화되었습니다.`
    return { ...detail, level, text }
  })
}

function formatDuration(sec) {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}분 ${s}초`
}

// CI SVG 시계열 차트
function CIChart({ data }) {
  if (data.length < 2) return <div className="ci-chart-empty">데이터 없음</div>

  const n = data.length
  const W = 640, H = 100
  const pts = data.map((d, i) => {
    const x = (i / (n - 1)) * W
    const y = H * (1 - Math.max(0, Math.min(1, d.ci)))
    return `${x},${y}`
  }).join(' ')
  const polyPts = `0,${H} ` + pts + ` ${W},${H}`
  const avgCI = Math.round(data.reduce((s, d) => s + d.ci, 0) / n * 100)

  return (
    <div className="ci-chart-wrap">
      <div className="ci-chart-header">
        <span className="ci-chart-title">시간에 따른 집중도 변화 양상</span>
        <span className="ci-avg">평균 CI: {avgCI}%</span>
      </div>
      <svg width="100%" height="100" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <rect width={W} height={H} fill="#eef2ff" rx="6" />
        {[25, 50, 75].map(y => (
          <line key={y} x1="0" y1={y} x2={W} y2={y} stroke="#e2e8f0" strokeWidth="1" strokeDasharray="4,4" />
        ))}
        <defs>
          <linearGradient id="ci-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#4f46e5" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#4f46e5" stopOpacity="0" />
          </linearGradient>
        </defs>
        <polygon points={polyPts} fill="url(#ci-grad)" />
        <polyline points={pts} fill="none" stroke="#4f46e5" strokeWidth="2.5" />
        <text x="6" y="22" fontSize="8" fill="#6b7280" fontFamily="system-ui">100%</text>
        <text x="6" y="47" fontSize="8" fill="#6b7280" fontFamily="system-ui">50%</text>
        <text x="6" y="97" fontSize="8" fill="#6b7280" fontFamily="system-ui">0%</text>
      </svg>
      <div className="ci-chart-axis">
        <span>0:00</span>
        <span>{formatDuration(data[Math.floor(n / 3)].timestamp - data[0].timestamp)}</span>
        <span>{formatDuration(data[Math.floor(2 * n / 3)].timestamp - data[0].timestamp)}</span>
        <span>{formatDuration(data[n - 1].timestamp - data[0].timestamp)}</span>
      </div>
    </div>
  )
}

function RegionBox({ info, side }) {
  const levelColor = { high: '#ef4444', low: '#4e92ff', normal: '#22c55e' }[info.level]
  const levelText = { high: '활성도 높음', low: '활성도 낮음', normal: '안정적' }[info.level]
  return (
    <div className={`rba-box rba-box-${side}`}>
      <div className="rba-box-header">
        <span className="rba-region">{info.name}</span>
        <span className="rba-level" style={{ color: levelColor }}>{levelText}</span>
      </div>
      <p className="rba-text">{info.text}</p>
      <div className="rba-role">{info.role}</div>
    </div>
  )
}

function BrainRegionPanel({ avgs, baselineHbo }) {
  const channels = buildChannelAnalysis(avgs)
  // baseline 대비 델타 값으로 뇌 맵 색상 표현 (live 화면과 동일 기준)
  const rawValues = avgs.map(a => a.hbo)
  const values = baselineHbo
    ? rawValues.map((v, i) => v - baselineHbo[i])
    : (() => {
        // baseline이 없으면 채널 간 상대 차이로 정규화
        const mean = rawValues.reduce((s, v) => s + v, 0) / rawValues.length
        const maxDev = Math.max(...rawValues.map(v => Math.abs(v - mean)), 0.001)
        return rawValues.map(v => ((v - mean) / maxDev) * 3)
      })()
  return (
    <div className="rba-wrap">
      <div className="rba-col">
        <RegionBox info={channels[0]} side="left" />
        <RegionBox info={channels[2]} side="left" />
      </div>
      <div className="rba-center">
        <BrainModelPNG view={BRAIN_FRONT} values={values} />
        <div className="rba-colorbar">
          <div className="rba-colorbar-grad" />
          <span>낮음 → 높음</span>
        </div>
      </div>
      <div className="rba-col">
        <RegionBox info={channels[1]} side="right" />
        <RegionBox info={channels[3]} side="right" />
      </div>
    </div>
  )
}

export default function ReportTab() {
  const { sessionData, userProfile, baseline } = useApp()

  // 마운트 시점 스냅샷 고정 — 이후 실시간 변화 무시
  const [snapshot] = useState(() => sessionData)
  const [pdfLoading, setPdfLoading] = useState(false)
  const pdfRef = useRef(null)

  const score    = useMemo(() => calcScore(snapshot), [snapshot])
  const duration = useMemo(() => sessionDuration(snapshot), [snapshot])
  const avgs     = useMemo(() => channelAverages(snapshot), [snapshot])
  const analysis = useMemo(() => buildAnalysis(avgs), [avgs])
  const channels = useMemo(() => buildChannelAnalysis(avgs), [avgs])

  const handleDownloadPDF = useCallback(async () => {
    if (!pdfRef.current || pdfLoading) return
    setPdfLoading(true)
    try {
      const now = new Date()
      const dateStr = `${now.getFullYear()}${String(now.getMonth()+1).padStart(2,'0')}${String(now.getDate()).padStart(2,'0')}`
      const name = (userProfile?.name ?? 'report').replace(/\s/g, '_')
      await generatePDF(pdfRef.current, `fnirs_${name}_${dateStr}.pdf`)
    } finally {
      setPdfLoading(false)
    }
  }, [pdfLoading, userProfile])

  const now = new Date()
  const dateStr = `${now.getFullYear()}.${String(now.getMonth() + 1).padStart(2, '0')}.${String(now.getDate()).padStart(2, '0')}`

  if (snapshot.length === 0) {
    return (
      <div className="report">
        <div className="rpt-header">
          <div>
            <div className="rpt-title">집중도 분석 레포트</div>
            <div className="rpt-subtitle">fNIRS Monitor · 세션 종료 후 자동 생성</div>
          </div>
        </div>
        <div className="rpt-body">
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            세션 데이터가 없습니다. Calibration 후 측정을 진행하세요.
          </p>
        </div>
      </div>
    )
  }

  const scoreLabel = score >= 80 ? '집중도 지수 — 우수' : score >= 60 ? '집중도 지수 — 양호' : '집중도 지수 — 보통'
  const scoreComment = score >= 80
    ? '전전두엽 HbO 신호가 측정 전반에 걸쳐 안정적으로 상승하였습니다. 높은 집중 수준이 유지되었습니다.'
    : score >= 60
      ? '전전두엽 HbO 신호가 측정 전반에 걸쳐 안정적으로 상승하였습니다. 양호한 집중 수준이 유지되었습니다.'
      : '집중도 편차가 다소 높게 나타났습니다. 더 안정적인 측정 환경을 권장합니다.'

  return (
    <>
    <div className="report">
      <div className="rpt-header">
        <div>
          <div className="rpt-title">집중도 분석 레포트</div>
          <div className="rpt-subtitle">fNIRS Monitor · 세션 종료 후 자동 생성</div>
        </div>
        <button className="btn-pdf" onClick={handleDownloadPDF} disabled={pdfLoading}>
          {pdfLoading ? '생성 중…' : '↓ PDF 내보내기'}
        </button>
      </div>

      <div className="rpt-body">
        {/* 세션 요약 */}
        <div>
          <div className="section-label">세션 요약</div>
          <div className="info-grid">
            <div className="info-card">
              <div className="info-card-label">이름</div>
              <div className="info-card-value">{userProfile?.name ?? '—'}</div>
            </div>
            <div className="info-card">
              <div className="info-card-label">나이</div>
              <div className="info-card-value">{userProfile?.age != null ? `${userProfile.age}세` : '—'}</div>
            </div>
            <div className="info-card">
              <div className="info-card-label">측정 날짜</div>
              <div className="info-card-value">{dateStr}</div>
            </div>
            <div className="info-card">
              <div className="info-card-label">측정 시간</div>
              <div className="info-card-value">{formatDuration(duration)}</div>
            </div>
          </div>
        </div>

        {/* 전체 집중도 점수 */}
        <div>
          <div className="section-label">전체 집중도 평가</div>
          <div className="score-row">
            <div
              className="score-circle"
              style={{ background: `conic-gradient(#4f46e5 0% ${score}%, #e2e8f0 ${score}% 100%)` }}
            >
              <div className="score-inner">
                <span className="score-num">{score}</span>
                <span className="score-denom">/100</span>
              </div>
            </div>
            <div className="score-desc">
              <div className="score-title">{scoreLabel}</div>
              <div className="score-bar-track">
                <div className="score-bar-fill" style={{ width: `${score}%` }} />
              </div>
              <div className="score-comment">{scoreComment}</div>
            </div>
          </div>
        </div>

        {/* CI 시계열 */}
        <div>
          <div className="section-label">집중도 지수(CI)</div>
          <CIChart data={snapshot} />
        </div>

        {/* 전전두엽 부위별 분석 */}
        <div>
          <div className="section-label">전전두엽 부위별 분석</div>
          <BrainRegionPanel avgs={avgs} baselineHbo={baseline.hbo} />
          <div className="analysis-note">
            본 분석은 fNIRS 신호 기반 참고 정보이며, 임상적 진단에 활용할 수 없습니다.
          </div>
        </div>
      </div>
    </div>

    {/* PDF 캡처용 숨겨진 A4 템플릿 — 항상 DOM에 있어야 ref가 유효 */}
    {createPortal(
      <div ref={pdfRef} style={{ position: 'fixed', top: -9999, left: 0, zIndex: -1, visibility: 'hidden' }}>
        <ReportPDFTemplate
          snapshot={snapshot}
          userProfile={userProfile}
          baseline={baseline}
          score={score}
          duration={duration}
          avgs={avgs}
          analysis={analysis}
          channels={channels}
        />
      </div>,
      document.body
    )}
    </>
  )
}
