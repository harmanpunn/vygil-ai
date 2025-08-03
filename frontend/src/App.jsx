import { useState } from 'react'
import ActivityTracker from './components/ActivityTracker'
import Header from './components/Header'
import './App.css'

function App() {
  const [monitoringStatus, setMonitoringStatus] = useState('stopped')
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-ghost via-primary-whisper to-primary-neutral">
      {/* Fixed Header */}
      <Header status={monitoringStatus} />
      
      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 md:px-6 lg:px-8 pt-16 pb-16">
        <ActivityTracker onStatusChange={setMonitoringStatus} />
      </div>
    </div>
  )
}

export default App
