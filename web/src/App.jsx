import { useState } from 'react'
import './App.css'

export default function App() {
  const [activeTab, setActiveTab] = useState('calibration')

  return (
    <div className="app">
      <header className="topbar">
        <span className="app-title">fNIRS Monitor</span>
      </header>
      <nav className="tab-nav">
        <button
          className={activeTab === 'calibration' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('calibration')}
        >
          Calibration
        </button>
        <button
          className={activeTab === 'brainmap' ? 'tab active' : 'tab'}
          disabled
        >
          Brain Map
        </button>
        <button
          className={activeTab === 'timeseries' ? 'tab active' : 'tab'}
          disabled
        >
          Time Series
        </button>
      </nav>
      <main className="tab-content">
        {activeTab === 'calibration' && <div>Calibration Tab (placeholder)</div>}
        {activeTab === 'brainmap' && <div>Brain Map Tab (placeholder)</div>}
        {activeTab === 'timeseries' && <div>Time Series Tab (placeholder)</div>}
      </main>
    </div>
  )
}
