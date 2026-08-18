/**
 * ReportPDFTemplate — A4(794px) 고정폭 PDF 전용 레이아웃
 * 화면에는 보이지 않고, generatePDF()가 캡처 시에만 사용.
 *
 * props: { snapshot, userProfile, baseline, score, duration, avgs, analysis, channels }
 */
import BrainModelPNG from './BrainModel/BrainModelPNG'
import { BRAIN_FRONT } from './BrainModel/brainViews'

const INDIGO  = '#4f46e5'
const INDIGO2 = '#6366f1'
const SLATE   = '#1e293b'
const MUTED   = '#64748b'
const BORDER  = '#e2e8f0'
const BG      = '#f8fafc'
const WHITE   = '#ffffff'

/* ── 공통 인라인 스타일 helpers ── */
const s = {
  page: {
    width: 794,
    background: WHITE,
    fontFamily: '"Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", system-ui, sans-serif',
    color: SLATE,
    fontSize: 12,
    boxSizing: 'border-box',
  },
  header: {
    background: INDIGO,
    padding: '28px 40px 24px',
    display: 'flex',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
  },
  section: {
    padding: '24px 40px 0',
  },
  sectionLabel: {
    fontSize: 9,
    fontWeight: 700,
    letterSpacing: 1.5,
    textTransform: 'uppercase',
    color: INDIGO2,
    marginBottom: 10,
  },
  card: {
    background: BG,
    border: `1px solid ${BORDER}`,
    borderRadius: 10,
    padding: '14px 18px',
  },
}

function Label({ children }) {
  return <div style={s.sectionLabel}>{children}</div>
}

function InfoGrid({ profile, dateStr, duration }) {
  const items = [
    { label: '이름', value: profile?.name ?? '—' },
    { label: '나이', value: profile?.age != null ? `${profile.age}세` : '—' },
    { label: '측정 날짜', value: dateStr },
    { label: '측정 시간', value: duration },
  ]
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10 }}>
      {items.map(({ label, value }) => (
        <div key={label} style={{ ...s.card }}>
          <div style={{ fontSize: 9, color: MUTED, marginBottom: 4 }}>{label}</div>
          <div style={{ fontSize: 13, fontWeight: 700 }}>{value}</div>
        </div>
      ))}
    </div>
  )
}

function ScoreRow({ score }) {
  const label = score >= 80 ? '우수' : score >= 60 ? '양호' : '보통'
  const comment = score >= 80
    ? '전전두엽 HbO 신호가 측정 전반에 걸쳐 안정적으로 상승하였으며, 높은 집중 수준이 유지되었습니다.'
    : score >= 60
    ? '전전두엽 HbO 신호가 양호한 수준으로 상승하였습니다. 집중 상태가 안정적으로 유지되었습니다.'
    : '집중도 편차가 다소 높게 나타났습니다. 더 안정적인 측정 환경을 권장합니다.'
  const pct = score + '%'

  return (
    <div style={{ ...s.card, display: 'flex', gap: 20, alignItems: 'center', borderLeft: `4px solid ${INDIGO}` }}>
      {/* 링 */}
      <div style={{
        width: 72, height: 72, borderRadius: '50%', flexShrink: 0,
        background: `conic-gradient(${INDIGO} 0% ${pct}, ${BORDER} ${pct} 100%)`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <div style={{
          width: 54, height: 54, borderRadius: '50%', background: BG,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        }}>
          <span style={{ fontSize: 18, fontWeight: 800, color: INDIGO, lineHeight: 1 }}>{score}</span>
          <span style={{ fontSize: 9, color: MUTED }}>/100</span>
        </div>
      </div>
      {/* 텍스트 */}
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 6 }}>집중도 지수 — {label}</div>
        <div style={{ height: 8, background: BORDER, borderRadius: 4, marginBottom: 6, overflow: 'hidden' }}>
          <div style={{ width: pct, height: '100%', background: `linear-gradient(to right,${INDIGO},${INDIGO2})`, borderRadius: 4 }} />
        </div>
        <div style={{ fontSize: 10, color: MUTED, lineHeight: 1.6 }}>{comment}</div>
      </div>
    </div>
  )
}

function CIChartPDF({ data }) {
  if (data.length < 2) return null
  const n = data.length
  const W = 714, H = 80
  const pts = data.map((d, i) => {
    const x = (i / (n - 1)) * W
    const y = H * (1 - Math.max(0, Math.min(1, d.ci)))
    return `${x},${y}`
  }).join(' ')
  const polyPts = `0,${H} ${pts} ${W},${H}`
  const avgCI = Math.round(data.reduce((s, d) => s + d.ci, 0) / n * 100)
  const dur = data[n - 1].timestamp - data[0].timestamp
  const fmt = s => `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`

  return (
    <div style={{ ...s.card }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{ fontSize: 11, fontWeight: 600 }}>시간에 따른 집중도 변화 양상</span>
        <span style={{ fontSize: 10, fontWeight: 700, color: INDIGO2 }}>평균 CI: {avgCI}%</span>
      </div>
      <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ display: 'block' }}>
        <rect width={W} height={H} fill="#eef2ff" rx="6" />
        {[20, 40, 60].map(y => (
          <line key={y} x1="0" y1={y} x2={W} y2={y} stroke={BORDER} strokeWidth="1" strokeDasharray="4,4" />
        ))}
        <defs>
          <linearGradient id="pdf-ci-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={INDIGO} stopOpacity="0.3" />
            <stop offset="100%" stopColor={INDIGO} stopOpacity="0" />
          </linearGradient>
        </defs>
        <polygon points={polyPts} fill="url(#pdf-ci-grad)" />
        <polyline points={pts} fill="none" stroke={INDIGO} strokeWidth="2" />
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8, color: MUTED, marginTop: 4 }}>
        <span>0:00</span>
        <span>{fmt(dur * 0.33)}</span>
        <span>{fmt(dur * 0.66)}</span>
        <span>{fmt(dur)}</span>
      </div>
    </div>
  )
}

function RegionBoxPDF({ ch, side }) {
  const levelColor = { high: '#dc2626', low: '#4f46e5', normal: '#059669' }[ch.level]
  const levelText  = { high: '활성도 높음', low: '활성도 낮음', normal: '안정적' }[ch.level]
  const accent = side === 'left' ? INDIGO : INDIGO2
  return (
    <div style={{ background: WHITE, border: `1px solid ${BORDER}`, borderLeft: `3px solid ${accent}`, borderRadius: 8, padding: '10px 12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
        <span style={{ fontSize: 10, fontWeight: 700 }}>{ch.name}</span>
        <span style={{ fontSize: 9, fontWeight: 700, color: levelColor }}>{levelText}</span>
      </div>
      <p style={{ fontSize: 9, color: MUTED, lineHeight: 1.6, margin: 0 }}>{ch.text}</p>
      <div style={{ fontSize: 8, color: INDIGO2, fontWeight: 600, marginTop: 4 }}>{ch.role}</div>
    </div>
  )
}

export default function ReportPDFTemplate({ snapshot, userProfile, baseline, score, duration, avgs, analysis, channels }) {
  const now = new Date()
  const dateStr = `${now.getFullYear()}.${String(now.getMonth()+1).padStart(2,'0')}.${String(now.getDate()).padStart(2,'0')}`
  const durStr = `${Math.floor(duration/60)}분 ${Math.floor(duration%60)}초`

  // 뇌 맵용 baseline 델타 값
  const brainValues = baseline
    ? avgs.map((a, i) => a.hbo - baseline.hbo[i])
    : (() => {
        const vals = avgs.map(a => a.hbo)
        const mean = vals.reduce((s,v)=>s+v,0)/vals.length
        const mx = Math.max(...vals.map(v=>Math.abs(v-mean)),0.001)
        return vals.map(v=>((v-mean)/mx)*3)
      })()

  return (
    <div style={s.page}>
      {/* ── 헤더 ── */}
      <div style={s.header}>
        <div>
          <div style={{ fontSize: 20, fontWeight: 800, color: WHITE, letterSpacing: -0.5 }}>fNIRS 집중도 분석 레포트</div>
          <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.7)', marginTop: 4 }}>
            fNIRS Monitor · 세션 종료 후 자동 생성
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.7)' }}>{dateStr}</div>
          <div style={{ fontSize: 13, fontWeight: 700, color: WHITE, marginTop: 4 }}>
            {userProfile?.name ?? '—'} 님
          </div>
        </div>
      </div>

      {/* ── 세션 요약 ── */}
      <div style={s.section}>
        <Label>세션 요약</Label>
        <InfoGrid profile={userProfile} dateStr={dateStr} duration={durStr} />
      </div>

      {/* ── 전체 집중도 ── */}
      <div style={{ ...s.section, paddingTop: 20 }}>
        <Label>전체 집중도 평가</Label>
        <ScoreRow score={score} />
      </div>

      {/* ── CI 차트 ── */}
      <div style={{ ...s.section, paddingTop: 20 }}>
        <Label>집중도 지수(CI) 시계열</Label>
        <CIChartPDF data={snapshot} />
      </div>

      {/* ── 뇌 부위별 분석 ── */}
      <div style={{ ...s.section, paddingTop: 20, paddingBottom: 32 }}>
        <Label>전전두엽 부위별 분석</Label>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 220px 1fr', gap: 16, alignItems: 'start' }}>
          {/* 좌측 채널 */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <RegionBoxPDF ch={channels[0]} side="left" />
            <RegionBoxPDF ch={channels[2]} side="left" />
          </div>

          {/* 뇌 맵 */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
            <BrainModelPNG view={BRAIN_FRONT} values={brainValues} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 8, color: MUTED }}>
              <span>낮음</span>
              <div style={{ width: 50, height: 5, borderRadius: 3, background: 'linear-gradient(to right,#4e7aff,#fff,#ef4444)' }} />
              <span>높음</span>
            </div>
          </div>

          {/* 우측 채널 */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <RegionBoxPDF ch={channels[1]} side="right" />
            <RegionBoxPDF ch={channels[3]} side="right" />
          </div>
        </div>

        {/* 종합 소견 */}
        <div style={{ marginTop: 16, padding: '12px 16px', background: '#eef2ff', borderRadius: 8, border: `1px solid #c7d2fe` }}>
          <div style={{ fontSize: 9, fontWeight: 700, color: INDIGO, marginBottom: 5 }}>종합 소견</div>
          <p style={{ fontSize: 10, color: SLATE, lineHeight: 1.7, margin: 0 }}>{analysis.dominance}</p>
        </div>

        {/* 면책 */}
        <div style={{ marginTop: 20, paddingTop: 12, borderTop: `1px solid ${BORDER}`, fontSize: 8.5, color: MUTED, textAlign: 'center', lineHeight: 1.6 }}>
          본 분석은 fNIRS 신호 기반 참고 정보이며, 임상적 진단에 활용할 수 없습니다.
        </div>
      </div>
    </div>
  )
}
