import { useState, useEffect } from 'react'

const AgentSelector = ({ currentAgent, onAgentChange, isActive }) => {
  const [availableAgents, setAvailableAgents] = useState([])
  const [selectedAgent, setSelectedAgent] = useState(currentAgent || 'vygil-activity-tracker')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // Fetch available agents on component mount
  useEffect(() => {
    fetchAvailableAgents()
  }, [])

  // Update selected agent when currentAgent prop changes
  useEffect(() => {
    if (currentAgent) {
      setSelectedAgent(currentAgent)
    }
  }, [currentAgent])

  const fetchAvailableAgents = async () => {
    try {
      console.log('Fetching agents from /api/agents...')
      const response = await fetch('/api/agents')
      console.log('Response status:', response.status)
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const data = await response.json()
      console.log('Received agent data:', data)
      
      setAvailableAgents(data.agents)
      setError(null) // Clear any previous errors
    } catch (err) {
      console.error('Error fetching agents:', err)
      setError(`Failed to load agents: ${err.message}`)
      // Fallback to default agents
      setAvailableAgents([
        {
          id: 'vygil-activity-tracker',
          name: 'Activity Tracker',
          description: 'Monitors and logs your screen activity',
          features: ['Activity Logging', 'Confidence Scoring', 'Real-time Monitoring']
        },
        {
          id: 'vygil-focus-assistant',
          name: 'Focus Assistant',
          description: 'Helps maintain focus by detecting distractions',
          features: ['Focus Tracking', 'Distraction Alerts', 'Productivity Analysis']
        }
      ])
    }
  }

  const handleAgentChange = async (agentId) => {
    if (agentId === selectedAgent || isActive) {
      return // Don't switch if same agent or currently monitoring
    }

    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch('/api/agents/select', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ agent_id: agentId })
      })

      if (!response.ok) {
        throw new Error('Failed to switch agent')
      }

      setSelectedAgent(agentId)
      onAgentChange && onAgentChange(agentId)
    } catch (err) {
      console.error('Error switching agent:', err)
      setError('Failed to switch agent')
    } finally {
      setIsLoading(false)
    }
  }

  const getAgentById = (agentId) => {
    return availableAgents.find(agent => 
      agent.id === agentId || agent === agentId
    ) || { 
      id: agentId, 
      name: agentId, 
      description: 'Unknown agent',
      features: []
    }
  }

  const currentAgentInfo = getAgentById(selectedAgent)

  return (
    <div className="bg-primary-ghost/30 backdrop-blur-sm rounded-2xl p-6 border border-primary-whisper/20">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-primary-dark flex items-center gap-2">
          <div className="w-2 h-2 bg-accents-sage rounded-full"></div>
          Agent Selection
        </h3>
        {isActive && (
          <div className="px-3 py-1 bg-accents-sage/20 text-accents-sage rounded-full text-sm font-medium">
            Active
          </div>
        )}
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Agent Dropdown */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-primary-dark mb-2">
          Current Agent
        </label>
        <select
          value={selectedAgent}
          onChange={(e) => handleAgentChange(e.target.value)}
          disabled={isLoading || isActive}
          className={`w-full px-4 py-3 border border-primary-whisper rounded-xl bg-white/80 backdrop-blur-sm 
            focus:outline-none focus:ring-2 focus:ring-accents-sage focus:border-transparent
            disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200
            ${isLoading ? 'animate-pulse' : ''}`}
        >
          {availableAgents.map((agent) => (
            <option key={agent.id || agent} value={agent.id || agent}>
              {agent.name || agent}
            </option>
          ))}
        </select>
      </div>

      {/* Current Agent Info */}
      <div className="space-y-3">
        <div>
          <h4 className="font-medium text-primary-dark mb-1">
            {currentAgentInfo.name}
          </h4>
          <p className="text-sm text-primary-muted">
            {currentAgentInfo.description}
          </p>
        </div>

        {/* Agent Features */}
        {currentAgentInfo.features && currentAgentInfo.features.length > 0 && (
          <div>
            <p className="text-sm font-medium text-primary-dark mb-2">Features:</p>
            <div className="flex flex-wrap gap-2">
              {currentAgentInfo.features.map((feature, index) => (
                <span
                  key={index}
                  className="px-2 py-1 bg-accents-sage/10 text-accents-sage rounded-md text-xs font-medium"
                >
                  {feature}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Agent Status */}
        <div className="flex items-center justify-between pt-3 border-t border-primary-whisper/30">
          <span className="text-sm text-primary-muted">Status:</span>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              isActive ? 'bg-green-500' : 'bg-gray-400'
            }`}></div>
            <span className="text-sm font-medium text-primary-dark">
              {isActive ? 'Monitoring' : 'Ready'}
            </span>
          </div>
        </div>
      </div>

      {isActive && (
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-sm text-yellow-800">
            <strong>Note:</strong> Stop monitoring to switch agents
          </p>
        </div>
      )}
    </div>
  )
}

export default AgentSelector