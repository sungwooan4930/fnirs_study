import { createContext, useContext, useState, useCallback } from 'react'

const AppContext = createContext(null)

export function AppProvider({ children }) {
  const [bleStatus, setBleStatus] = useState('idle')

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
      calibrationDone, setCalibrationDone,
      processedSample,
      sessionData,
      pushSample,
      clearSession,
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
