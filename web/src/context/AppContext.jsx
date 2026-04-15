import { createContext, useContext, useState, useCallback, useRef } from 'react'

const AppContext = createContext(null)

export function AppProvider({ children }) {
  const [bleStatus, setBleStatus] = useState('idle')

  const [userProfile, setUserProfile] = useState(null)  // { name, age }

  // 앱 단계: 'connection' | 'calibration' | 'measuring' | 'report'
  const [appPhase, setAppPhase] = useState('connection')

  const [calibrationDone, setCalibrationDone] = useState(false)

  const [processedSample, setProcessedSample] = useState(null)

  const [sessionData, setSessionData] = useState(() => {
    try {
      const saved = sessionStorage.getItem('fnirs_session')
      return saved ? JSON.parse(saved) : []
    } catch {
      return []
    }
  })

  // BLE connect/disconnect를 ConnectionStep에서도 호출할 수 있도록 ref로 노출
  const bleActionsRef = useRef({ connect: () => {}, disconnect: () => {} })
  const bleConnect = useCallback(() => bleActionsRef.current.connect(), [])
  const bleDisconnect = useCallback(() => bleActionsRef.current.disconnect(), [])

  const pushSample = useCallback((sample) => {
    setProcessedSample(sample)
    setSessionData((prev) => {
      const next = [...prev, sample]
      try { sessionStorage.setItem('fnirs_session', JSON.stringify(next)) } catch {}
      return next
    })
  }, [])

  const clearSession = useCallback(() => {
    setSessionData([])
    setProcessedSample(null)
    sessionStorage.removeItem('fnirs_session')
  }, [])

  return (
    <AppContext.Provider value={{
      bleStatus, setBleStatus,
      userProfile, setUserProfile,
      appPhase, setAppPhase,
      calibrationDone, setCalibrationDone,
      processedSample,
      sessionData,
      pushSample,
      clearSession,
      bleActionsRef, bleConnect, bleDisconnect,
    }}>
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}
