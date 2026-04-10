import { useState } from 'react'
import { AppProvider, useApp } from './context/AppContext'
import './App.css'

function TabContainer() {
  const { calibrationDone } = useApp()
  const [activeTab, setActiveTab] = useState('calibration')

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
      </nav>
      <main className="tab-content">
        {activeTab === 'calibration' && <div className="placeholder">Calibration Tab</div>}
        {activeTab === 'brainmap' && <div className="placeholder">Brain Map Tab</div>}
        {activeTab === 'timeseries' && <div className="placeholder">Time Series Tab</div>}
      </main>
    </>
  )
}

export default function App() {
  return (
    <AppProvider>
      <div className="app">
        <header className="topbar">
          <span className="app-title">fNIRS Monitor</span>
        </header>
        <TabContainer />
      </div>
    </AppProvider>
  )
}
