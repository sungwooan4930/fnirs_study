/**
 * BaselineStep — 2분간 안정 상태 신호 수집 후 baseline 산출
 *
 * 흐름:
 *   1. 마운트 시 세션 데이터 초기화 (calibration 잔류 데이터 제거)
 *   2. 2분 카운트다운 진행 (그동안 pipeline에서 계속 sample 수집)
 *   3. 완료 시 sessionData 평균 → baseline 저장
 *   4. 세션 데이터 재초기화 후 'measuring' 단계로 전환
 */
import { useState, useEffect, useRef } from 'react'
import { useApp } from '../context/AppContext'
import './BaselineStep.css'

const BASELINE_SEC = 120  // 2분

export default function BaselineStep() {
  const { sessionData, clearSession, setBaseline, setAppPhase } = useApp()
  const [elapsed, setElapsed] = useState(0)
  const [done, setDone] = useState(false)
  const computedRef = useRef(null)

  // 마운트 시 calibration 잔류 데이터 제거
  useEffect(() => {
    clearSession()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // 1초 간격 카운트다운
  useEffect(() => {
    if (done) return
    const id = setInterval(() => {
      setElapsed(prev => {
        const next = prev + 1
        if (next >= BASELINE_SEC) {
          clearInterval(id)
          setDone(true)
        }
        return next
      })
    }, 1000)
    return () => clearInterval(id)
  }, [done])

  // 완료 시 baseline 산출
  useEffect(() => {
    if (!done) return
    const n = sessionData.length
    if (n === 0) {
      computedRef.current = { hbo: [0, 0, 0, 0], hbr: [0, 0, 0, 0] }
      return
    }
    const hbo = [0, 1, 2, 3].map(ch =>
      sessionData.reduce((s, d) => s + d.hbo[ch], 0) / n
    )
    const hbr = [0, 1, 2, 3].map(ch =>
      sessionData.reduce((s, d) => s + d.hbr[ch], 0) / n
    )
    computedRef.current = { hbo, hbr }
  }, [done, sessionData])

  const handleProceed = () => {
    if (computedRef.current) setBaseline(computedRef.current)
    clearSession()          // 측정 데이터는 새로 시작
    setAppPhase('measuring')
  }

  const remaining = BASELINE_SEC - elapsed
  const mm = String(Math.floor(remaining / 60)).padStart(2, '0')
  const ss = String(remaining % 60).padStart(2, '0')
  const progress = elapsed / BASELINE_SEC

  return (
    <div className="baseline-step">
      <div className="baseline-card">
        <div className="baseline-title">기준선 측정 (Baseline)</div>
        <p className="baseline-desc">
          편안한 상태로 가만히 앉아 계세요.<br />
          2분간 안정 상태의 뇌 혈류를 측정하여 기준선을 설정합니다.
        </p>

        {!done ? (
          <>
            <div className="baseline-timer">{mm}:{ss}</div>
            <div className="baseline-progress-track">
              <div
                className="baseline-progress-fill"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
            <div className="baseline-hint">움직이거나 말하지 마세요</div>
            <div className="baseline-sample-count">
              수집된 샘플: {sessionData.length}
            </div>
          </>
        ) : (
          <>
            <div className="baseline-done-icon">✓</div>
            <div className="baseline-done-text">기준선 측정 완료</div>
            <p className="baseline-done-sub">
              {sessionData.length}개 샘플로 채널별 기준선이 산출되었습니다.
            </p>
            <button className="baseline-proceed-btn" onClick={handleProceed}>
              신호 측정 시작 ▶
            </button>
          </>
        )}
      </div>
    </div>
  )
}
