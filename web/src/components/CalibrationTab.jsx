import { useState, useEffect, useRef } from 'react'
import { useApp } from '../context/AppContext'
import './CalibrationTab.css'

const N_CHANNELS = 4
const WAVELENGTHS = ['780nm', '850nm', '950nm']
const CALIB_DURATION_MS = 3000
const SNR_THRESHOLD = 0.6  // 이 이상이면 Good

export default function CalibrationTab() {
  const { bleStatus, setCalibrationDone, setAppPhase } = useApp()
  const [snr, setSnr] = useState(
    Array.from({ length: N_CHANNELS }, () => [0, 0, 0])
  )
  const [channelStatus, setChannelStatus] = useState(Array(N_CHANNELS).fill('waiting'))
  // 'waiting' | 'good' | 'poor'
  const [calibrating, setCalibrating] = useState(false)
  const [allGood, setAllGood] = useState(false)
  const timerRef = useRef(null)

  // 연결되면 자동으로 Calibration 시뮬레이션 시작
  useEffect(() => {
    if (bleStatus !== 'connected') return

    setCalibrating(true)
    const startTime = Date.now()

    timerRef.current = setInterval(() => {
      const elapsed = Date.now() - startTime
      const progress = Math.min(elapsed / CALIB_DURATION_MS, 1)

      // SNR을 점진적으로 채움 (시뮬레이션)
      const newSnr = Array.from({ length: N_CHANNELS }, (_, ch) =>
        WAVELENGTHS.map((_, wl) =>
          Math.min(progress * (0.65 + Math.random() * 0.25), 1)
        )
      )
      setSnr(newSnr)

      if (elapsed >= CALIB_DURATION_MS) {
        clearInterval(timerRef.current)
        setCalibrating(false)

        const finalStatus = newSnr.map(chSnr =>
          chSnr.every(v => v >= SNR_THRESHOLD) ? 'good' : 'poor'
        )
        setChannelStatus(finalStatus)
        setAllGood(finalStatus.every(s => s === 'good'))
      }
    }, 100)

    return () => clearInterval(timerRef.current)
  }, [bleStatus])

  return (
    <div className="calibration">
      <h2 className="calib-title">신호 보정 (Calibration)</h2>

      {bleStatus !== 'connected' && (
        <p className="calib-hint">상단에서 fNIRS 기기를 먼저 연결하세요.</p>
      )}

      <div className="snr-grid">
        {Array.from({ length: N_CHANNELS }, (_, ch) => (
          <div key={ch} className={`channel-card ${channelStatus[ch]}`}>
            <div className="channel-label">
              Ch {ch + 1}
              <span className={`status-badge ${channelStatus[ch]}`}>
                {channelStatus[ch] === 'good' ? '✓ Good' : channelStatus[ch] === 'poor' ? '✗ Poor' : '…'}
              </span>
            </div>
            <div className="wl-bars">
              {WAVELENGTHS.map((wl, wlIdx) => (
                <div key={wl} className="wl-row">
                  <span className="wl-label">{wl}</span>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{
                        width: `${Math.round(snr[ch][wlIdx] * 100)}%`,
                        background: snr[ch][wlIdx] >= SNR_THRESHOLD
                          ? 'linear-gradient(to right, var(--accent), var(--accent-2))'
                          : 'var(--warning)',
                      }}
                    />
                  </div>
                  <span className="bar-value">{Math.round(snr[ch][wlIdx] * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <button
        className="btn-proceed"
        disabled={!allGood}
        onClick={() => { setCalibrationDone(true); setAppPhase('measuring') }}
      >
        측정 시작 ▶
      </button>
    </div>
  )
}
