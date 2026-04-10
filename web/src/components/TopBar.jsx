import { useCallback } from 'react'
import { useApp } from '../context/AppContext'
import { useBLE } from '../hooks/useBLE'
import { usePipeline } from '../hooks/usePipeline'
import './TopBar.css'

const STATUS_LABEL = {
  idle: '연결 안됨',
  connecting: '연결 중...',
  connected: '연결됨',
  error: '연결 오류',
}

function downloadCSV(sessionData) {
  if (sessionData.length === 0) return

  const header = 'timestamp,hbo_ch1,hbo_ch2,hbo_ch3,hbo_ch4,hbr_ch1,hbr_ch2,hbr_ch3,hbr_ch4,ci'
  const rows = sessionData.map(s =>
    [
      s.timestamp.toFixed(3),
      ...s.hbo.map(v => v.toFixed(4)),
      ...s.hbr.map(v => v.toFixed(4)),
      s.ci.toFixed(4),
    ].join(',')
  )
  const csv = [header, ...rows].join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)

  const now = new Date()
  const ts = now.toISOString().replace(/[-:T]/g, '').slice(0, 15)
  const a = document.createElement('a')
  a.href = url
  a.download = `fnirs_session_${ts}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

const STATUS_COLOR = {
  idle: 'var(--text-muted)',
  connecting: 'var(--warning)',
  connected: 'var(--success)',
  error: 'var(--error)',
}

export default function TopBar() {
  const { bleStatus, setBleStatus, processedSample, pushSample, sessionData } = useApp()

  const { sendPacket } = usePipeline(pushSample)

  const onPacket = useCallback((packet) => {
    sendPacket(packet)
  }, [sendPacket])

  const { connect, disconnect, isSimulate } = useBLE(onPacket, setBleStatus)

  const ci = processedSample?.ci ?? 0
  const ciPercent = Math.round(ci * 100)

  return (
    <header className="topbar">
      <span className="app-title">fNIRS Monitor{isSimulate ? ' [SIM]' : ''}</span>

      <div className="ble-group">
        <span className="ble-dot" style={{ background: STATUS_COLOR[bleStatus] }} />
        <span className="ble-label">{STATUS_LABEL[bleStatus]}</span>
        {bleStatus === 'idle' || bleStatus === 'error' ? (
          <button className="btn-connect" onClick={connect}>연결</button>
        ) : (
          <button className="btn-disconnect" onClick={disconnect}>해제</button>
        )}
      </div>

      <div className="ci-group">
        <span className="ci-label">CI</span>
        <div className="ci-bar-track">
          <div className="ci-bar-fill" style={{ width: `${ciPercent}%` }} />
        </div>
        <span className="ci-value">{ciPercent}%</span>
      </div>

      <button
        className="btn-download"
        disabled={sessionData.length === 0}
        onClick={() => downloadCSV(sessionData)}
        title="세션 데이터 CSV 다운로드"
      >
        ↓ CSV
      </button>
    </header>
  )
}
