/**
 * BrainModelPNG — 정적 이미지 + SVG radialGradient 블롭 오버레이
 *
 * props:
 *   view     {object}  brainViews.js 의 뷰 정의 (image + channels)
 *   values   {number[]}  채널별 신호 값 (hbo 또는 hbr), 4개
 */
import { BRAIN_FRONT } from './brainViews.js'

const VMIN = -5.0
const VMAX = 5.0

function valToColor(val) {
  const t = Math.max(0, Math.min(1, (val - VMIN) / (VMAX - VMIN)))
  if (t < 0.5) {
    const f = t * 2
    return `rgb(${Math.round(78 + 177 * f)},${Math.round(146 + 109 * f)},255)`
  }
  const f = (t - 0.5) * 2
  const gb = Math.round(255 * (1 - f))
  return `rgb(255,${gb},${gb})`
}

function valToOpacity(val) {
  const t = Math.abs((val - (VMIN + VMAX) / 2) / ((VMAX - VMIN) / 2))
  return 0.35 + Math.min(t, 1) * 0.4
}

export default function BrainModelPNG({ view = BRAIN_FRONT, values = [0, 0, 0, 0] }) {
  return (
    <div style={{ position: 'relative', display: 'inline-block', width: '100%', maxWidth: 960 }}>
      <img
        src={view.image}
        alt="brain front view"
        draggable={false}
        style={{ width: '100%', height: 'auto', borderRadius: 12, display: 'block' }}
      />
      <svg
        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', borderRadius: 12 }}
        viewBox="0 0 340 290"
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          {view.channels.map(({ id, x, y, r }, ch) => {
            const color = valToColor(values[ch])
            const opacity = valToOpacity(values[ch])
            const gid = `png-grad-${id}`
            return (
              <radialGradient key={gid} id={gid} cx={`${x}%`} cy={`${y}%`} r={`${r}%`}>
                <stop offset="0%"   stopColor={color} stopOpacity={opacity} />
                <stop offset="100%" stopColor={color} stopOpacity={0} />
              </radialGradient>
            )
          })}
        </defs>
        {view.channels.map(({ id }) => (
          <rect key={id} width="340" height="290" fill={`url(#png-grad-${id})`} />
        ))}
        {/* Reset 버튼 UI 가리기 */}
        <rect x="100" y="256" width="140" height="34" fill="white" fillOpacity="0.85" />
      </svg>
    </div>
  )
}
