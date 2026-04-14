import { useState, useEffect } from 'react'
import { AppProvider, useApp } from './context/AppContext'
import TopBar from './components/TopBar'
import CalibrationTab from './components/CalibrationTab'
import TimeSeriesTab from './components/TimeSeriesTab'
import BrainMapTab from './components/BrainMapTab'
import ReportTab from './components/ReportTab'
import ProfileStep from './components/ProfileStep'
import './App.css'

function useElapsed() {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const start = Date.now()
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000)
    return () => clearInterval(id)
  }, [])
  const m = String(Math.floor(elapsed / 60)).padStart(2, '0')
  const s = String(elapsed % 60).padStart(2, '0')
  return `${m}:${s}`
}

function MeasuringView() {
  const { setAppPhase } = useApp()
  const [activeTab, setActiveTab] = useState('brainmap')
  const elapsed = useElapsed()

  return (
    <div className="measuring-view">
      <div className="measuring-toolbar">
        <nav className="measuring-tabs">
          <button
            className={activeTab === 'brainmap' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('brainmap')}
          >
            3D Brain
          </button>
          <button
            className={activeTab === 'timeseries' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('timeseries')}
          >
            Time Series
          </button>
        </nav>
        <span className="measuring-elapsed">⏱ {elapsed}</span>
        <button className="btn-stop" onClick={() => setAppPhase('report')}>
          ■ 측정 종료
        </button>
      </div>
      <div className="measuring-content">
        {activeTab === 'brainmap' && <BrainMapTab />}
        {activeTab === 'timeseries' && <TimeSeriesTab />}
      </div>
    </div>
  )
}

function AppFlow() {
  const { userProfile, appPhase } = useApp()

  if (!userProfile) return <ProfileStep />
  if (appPhase === 'calibration') return <CalibrationTab />
  if (appPhase === 'measuring') return <MeasuringView />
  if (appPhase === 'report') return (
    <div className="report-scroll-wrap">
      <ReportTab />
    </div>
  )
}

export default function App() {
  return (
    <AppProvider>
      <div className="app">
        <TopBar />
        <AppFlow />
      </div>
    </AppProvider>
  )
}
