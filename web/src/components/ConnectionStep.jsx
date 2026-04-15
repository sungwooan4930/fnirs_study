import { useApp } from '../context/AppContext'
import './ConnectionStep.css'

export default function ConnectionStep() {
  const { bleStatus, setAppPhase, bleConnect, bleDisconnect } = useApp()
  const connected = bleStatus === 'connected'

  return (
    <div className="conn-step">
      <div className="conn-card">
        <div className="conn-icon">{connected ? '✓' : '📡'}</div>
        <h2 className="conn-title">기기 연결</h2>
        <p className="conn-desc">
          {connected
            ? 'fNIRS 기기가 성공적으로 연결되었습니다.'
            : 'fNIRS 기기를 블루투스로 연결하세요.\n시뮬레이션 모드는 바로 연결됩니다.'}
        </p>

        <div className={`conn-status-dot ${bleStatus}`} />
        <div className="conn-status-label">
          {{ idle: '연결 안됨', connecting: '연결 중…', connected: '연결됨', error: '연결 오류' }[bleStatus]}
        </div>

        {!connected ? (
          <button
            className="conn-btn"
            onClick={bleConnect}
            disabled={bleStatus === 'connecting'}
          >
            {bleStatus === 'connecting' ? '연결 중…' : '연결하기'}
          </button>
        ) : (
          <button className="conn-btn-proceed" onClick={() => setAppPhase('calibration')}>
            신호 보정 시작 ▶
          </button>
        )}

        {bleStatus === 'error' && (
          <p className="conn-error">연결에 실패했습니다. 다시 시도하세요.</p>
        )}
      </div>
    </div>
  )
}
