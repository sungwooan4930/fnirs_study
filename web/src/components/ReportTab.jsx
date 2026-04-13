import { useState, useMemo } from 'react'
import { useApp } from '../context/AppContext'
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

// 채널 평균에서 뇌 부위별 분석 텍스트 생성
function buildAnalysis(avgs) {
  const leftAvg  = (avgs[0].hbo + avgs[2].hbo) / 2   // 좌전방 + 좌후방
  const rightAvg = (avgs[1].hbo + avgs[3].hbo) / 2   // 우전방 + 우후방
  const diff = leftAvg - rightAvg
  const thr = 0.3

  const fmt = (v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}μmol/L`

  const leftText = Math.abs(diff) < thr
    ? `좌측 전전두엽(좌전방·좌후방 PFC)이 ${fmt(leftAvg)} 수준으로 우측과 균형 있게 활성화되었습니다.`
    : diff > 0
      ? `좌측 전전두엽(좌전방·좌후방 PFC)의 HbO가 ${fmt(leftAvg)} 수준으로 두드러지게 상승하였습니다. 좌측 전전두엽은 언어 처리 및 논리적 사고와 관련된 영역으로, 분석적 집중 또는 언어 기반 작업에 더 몰입하는 모습을 보입니다.`
      : `좌측 전전두엽(좌전방·좌후방 PFC)의 HbO가 ${fmt(leftAvg)} 수준으로 우측 대비 상대적으로 낮은 활성도를 보였습니다.`

  const rightText = Math.abs(diff) < thr
    ? `우측 전전두엽(우전방·우후방 PFC)이 ${fmt(rightAvg)} 수준으로 좌측과 고른 활성화를 보였습니다.`
    : diff > 0
      ? `우측 전전두엽(우전방·우후방 PFC)의 HbO가 ${fmt(rightAvg)} 수준으로 좌측 대비 상대적으로 낮은 활성도를 보였습니다. 우측 전전두엽은 공간 인식 및 창의적 사고와 연관되며, 이번 세션에서는 좌측 우세 패턴이 나타났습니다.`
      : `우측 전전두엽(우전방·우후방 PFC)의 HbO가 ${fmt(rightAvg)} 수준으로 두드러지게 상승하였습니다. 우측 전전두엽은 공간 인식 및 창의적 사고와 연관되며, 이번 세션에서는 우측 우세 패턴이 나타났습니다.`

  const dominance = Math.abs(diff) < thr
    ? '좌우 전전두엽이 균형 있게 활성화된 패턴으로, 전반적인 집중 상태를 유지하는 것으로 판단됩니다.'
    : diff > 0
      ? '좌측 전전두엽 우세 활성화 패턴으로, 논리적·언어적 과제에 집중하는 상태로 판단됩니다.'
      : '우측 전전두엽 우세 활성화 패턴으로, 공간적·직관적 과제에 집중하는 상태로 판단됩니다.'

  return { leftText, rightText, dominance }
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
        <rect width={W} height={H} fill="#0f1117" rx="6" />
        {[25, 50, 75].map(y => (
          <line key={y} x1="0" y1={y} x2={W} y2={y} stroke="#1e2130" strokeWidth="1" strokeDasharray="4,4" />
        ))}
        <defs>
          <linearGradient id="ci-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#1f55f1" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#1f55f1" stopOpacity="0" />
          </linearGradient>
        </defs>
        <polygon points={polyPts} fill="url(#ci-grad)" />
        <polyline points={pts} fill="none" stroke="#1f55f1" strokeWidth="2.5" />
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

// 뇌 맵 스냅샷 SVG
function BrainSnapshot({ avgs }) {
  const CH_SNAP = [
    { id: 'rs1', cx: '40%', cy: '50%', r: '25%' },
    { id: 'rs2', cx: '60%', cy: '50%', r: '22%' },
    { id: 'rs3', cx: '40%', cy: '65%', r: '20%' },
    { id: 'rs4', cx: '60%', cy: '65%', r: '18%' },
  ]

  function valToColor(val) {
    const t = Math.max(0, Math.min(1, (val + 5) / 10))
    if (t < 0.5) {
      const f = t * 2
      return `rgb(${Math.round(78 + 177 * f)},${Math.round(146 + 109 * f)},255)`
    }
    const f = (t - 0.5) * 2
    const gb = Math.round(255 * (1 - f))
    return `rgb(255,${gb},${gb})`
  }

  function valToOpacity(val) {
    return 0.35 + Math.min(Math.abs(val) / 5, 1) * 0.35
  }

  return (
    <div className="brain-snap-svg-wrap">
      <svg viewBox="0 0 240 200" width="100%" style={{ display: 'block' }}>
        <ellipse cx="120" cy="95" rx="100" ry="85" fill="#1e2130" stroke="#1f55f1" strokeWidth="1" opacity="0.6" />
        <defs>
          {CH_SNAP.map(({ id, cx, cy, r }, ch) => {
            const color = valToColor(avgs[ch].hbo)
            const opacity = valToOpacity(avgs[ch].hbo)
            return (
              <radialGradient key={id} id={id} cx={cx} cy={cy} r={r}>
                <stop offset="0%"   stopColor={color} stopOpacity={opacity} />
                <stop offset="100%" stopColor={color} stopOpacity={0} />
              </radialGradient>
            )
          })}
        </defs>
        {CH_SNAP.map(({ id }) => (
          <rect key={id} width="240" height="200" fill={`url(#${id})`} />
        ))}
        <text x="120" y="188" textAnchor="middle" fontSize="9" fill="#6b7280" fontFamily="system-ui">PFC · HbO</text>
      </svg>
    </div>
  )
}

// 부위별 평균 바 (채널명 대신 뇌 부위명)
function RegionAverages({ avgs }) {
  function toWidth(v) { return Math.max(0, Math.min(100, (v / 5 + 1) * 50)) }

  return (
    <div className="ch-avg-list">
      {avgs.map((avg, ch) => (
        <div key={ch} className="ch-avg-row">
          <span className="ch-avg-label">{REGION_SHORT[ch]}</span>
          <div className="ch-avg-bars">
            <div className="ch-avg-item">
              <span className="ch-type hbo">HbO</span>
              <div className="ch-track">
                <div className="ch-fill hbo" style={{ width: `${toWidth(avg.hbo)}%` }} />
              </div>
              <span className="ch-val">{avg.hbo >= 0 ? '+' : ''}{avg.hbo.toFixed(1)}</span>
            </div>
            <div className="ch-avg-item">
              <span className="ch-type hbr">HbR</span>
              <div className="ch-track">
                <div className="ch-fill hbr" style={{ width: `${toWidth(avg.hbr)}%` }} />
              </div>
              <span className="ch-val">{avg.hbr >= 0 ? '+' : ''}{avg.hbr.toFixed(1)}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

export default function ReportTab() {
  const { sessionData, userProfile } = useApp()

  // 마운트 시점 스냅샷 고정 — 이후 실시간 변화 무시
  const [snapshot] = useState(() => sessionData)

  const score    = useMemo(() => calcScore(snapshot), [snapshot])
  const duration = useMemo(() => sessionDuration(snapshot), [snapshot])
  const avgs     = useMemo(() => channelAverages(snapshot), [snapshot])
  const analysis = useMemo(() => buildAnalysis(avgs), [avgs])

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
    <div className="report">
      <div className="rpt-header">
        <div>
          <div className="rpt-title">집중도 분석 레포트</div>
          <div className="rpt-subtitle">fNIRS Monitor · 세션 종료 후 자동 생성</div>
        </div>
        <button className="btn-pdf" onClick={() => window.print()}>↓ PDF 내보내기</button>
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
              style={{ background: `conic-gradient(#1f55f1 0% ${score}%, #1e2130 ${score}% 100%)` }}
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

        {/* 뇌 맵 + 부위별 평균 */}
        <div>
          <div className="section-label">측정 결과 요약</div>
          <div className="bottom-row">
            <div className="brain-snap">
              <div className="snap-title">뇌 맵 스냅샷</div>
              <div className="snap-sub">전전두엽(PFC) 부위별 활성화 분포</div>
              <BrainSnapshot avgs={avgs} />
              <div className="snap-colorbar">
                <div className="snap-colorbar-grad" />
                <span>−5 → 0 → +5 μmol/L</span>
              </div>
            </div>
            <div className="brain-snap">
              <div className="snap-title">부위별 평균</div>
              <div className="snap-sub">세션 전체 HbO/HbR 평균값</div>
              <RegionAverages avgs={avgs} />
            </div>
          </div>
        </div>

        {/* 부위별 신호 분석 */}
        <div>
          <div className="section-label">전전두엽 부위별 분석</div>
          <div className="analysis-card">
            <div className="analysis-intro">
              측정된 신호를 기반으로 전전두엽(PFC) 영역별 활성화 패턴을 분석한 결과입니다.
              HbO 증가(↑)는 해당 영역의 혈류 증가 및 신경 활성화를, HbR 감소(↓)는 산소 소비 증가를 의미합니다.
            </div>
            <div className="analysis-items">
              <div className="analysis-item">
                <span className="analysis-tag tag-left">좌측 PFC</span>
                <div className="analysis-text">{analysis.leftText}</div>
              </div>
              <div className="analysis-item">
                <span className="analysis-tag tag-right">우측 PFC</span>
                <div className="analysis-text">{analysis.rightText}</div>
              </div>
              <div className="analysis-item">
                <span className="analysis-tag tag-overall">종합</span>
                <div className="analysis-text">{analysis.dominance}</div>
              </div>
              <div className="analysis-item">
                <span className="analysis-tag tag-note">참고</span>
                <div className="analysis-text note">
                  본 분석은 fNIRS 신호 기반 참고 정보이며, 임상적 진단에 활용할 수 없습니다.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
