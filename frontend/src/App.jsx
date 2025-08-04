import { useState } from 'react'
import ActivityTracker from './components/ActivityTracker'
import AgentSelector from './components/AgentSelector'
import SystemDashboard from './components/SystemDashboard'
import Header from './components/Header'
import './App.css'

function App() {
  const [monitoringStatus, setMonitoringStatus] = useState('stopped')
  const [currentAgent, setCurrentAgent] = useState('vygil-focus-assistant')
  
  const handleAgentChange = (agentId) => {
    setCurrentAgent(agentId)
    console.log('Agent switched to:', agentId)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-ghost via-primary-whisper to-primary-neutral">
      {/* Fixed Header */}
      <Header status={monitoringStatus} />
      
      {/* Main Content */}
      <div className="mx-auto px-4 md:px-6 lg:px-8 pt-20 pb-16">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Sidebar - Agent & System Info */}
          <div className="lg:col-span-1 space-y-6">
            <AgentSelector 
              currentAgent={currentAgent}
              onAgentChange={handleAgentChange}
              isActive={monitoringStatus === 'monitoring'}
            />
            <SystemDashboard />
          </div>
          
          {/* Main Activity Tracker */}
          <div className="lg:col-span-3">
            <ActivityTracker 
              onStatusChange={setMonitoringStatus}
              currentAgent={currentAgent}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
