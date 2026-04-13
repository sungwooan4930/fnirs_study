import { useState } from 'react'
import { AppProvider, useApp } from './context/AppContext'
import TopBar from './components/TopBar'
import CalibrationTab from './components/CalibrationTab'
import TimeSeriesTab from './components/TimeSeriesTab'
import BrainMapTab from './components/BrainMapTab'
import ReportTab from './components/ReportTab'
import ProfileStep from './components/ProfileStep'
import './App.css'

function TabContainer() {
  const { calibrationDone, userProfile } = useApp()
  const [activeTab, setActiveTab] = useState('calibration')

  if (!userProfile) return <ProfileStep />

  return (
    <>
      <nav className="tab-nav">
        <button
          className={activeTab === 'calibration' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('calibration')}
        >
          Calibration
        </button>
        <button
          className={activeTab === 'brainmap' ? 'tab active' : 'tab'}
          disabled={!calibrationDone}
          onClick={() => calibrationDone && setActiveTab('brainmap')}
          title={!calibrationDone ? 'Calibration 완료 후 활성화됩니다' : undefined}
        >
          Brain Map
        </button>
        <button
          className={activeTab === 'timeseries' ? 'tab active' : 'tab'}
          disabled={!calibrationDone}
          onClick={() => calibrationDone && setActiveTab('timeseries')}
          title={!calibrationDone ? 'Calibration 완료 후 활성화됩니다' : undefined}
        >
          Time Series
        </button>
        <button
          className={activeTab === 'report' ? 'tab active' : 'tab'}
          disabled={!calibrationDone}
          onClick={() => calibrationDone && setActiveTab('report')}
          title={!calibrationDone ? 'Calibration 완료 후 활성화됩니다' : undefined}
        >
          Report
        </button>
      </nav>
      <main className="tab-content">
        {activeTab === 'calibration' && <CalibrationTab />}
        {activeTab === 'brainmap' && <BrainMapTab />}
        {activeTab === 'timeseries' && <TimeSeriesTab />}
        {activeTab === 'report' && <ReportTab />}
      </main>
    </>
  )
}

export default function App() {
  return (
    <AppProvider>
      <div className="app">
        <TopBar />
        <TabContainer />
      </div>
    </AppProvider>
  )
}
