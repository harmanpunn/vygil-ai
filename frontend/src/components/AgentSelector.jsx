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

      // Add Video Intelligence agent to the list (not in backend YAML configs)
      const videoIntelligenceAgent = {
        id: 'vygil-video-intelligence',
        name: 'Video Intelligence',
        description: 'AI-powered product detection in videos',
        features: ['Product Detection', 'Timeline Analysis', 'E-commerce Links']
      }

      setAvailableAgents([...data.agents, videoIntelligenceAgent])
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
        },
        {
          id: 'vygil-video-intelligence',
          name: 'Video Intelligence',
          description: 'AI-powered product detection in videos',
          features: ['Product Detection', 'Timeline Analysis', 'E-commerce Links']
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
      // Video Intelligence is frontend-only, no backend agent to select
      if (agentId === 'vygil-video-intelligence') {
        setSelectedAgent(agentId)
        onAgentChange && onAgentChange(agentId)
        setIsLoading(false)
        return
      }

      // For other agents, call backend API
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
    <div className="bg-primary-ghost border border-primary-neutral/30 rounded-xl p-6 transition-all duration-200 hover:transform hover:-translate-y-0.5 hover:shadow-lg shadow-primary-obsidian/8">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-medium text-primary-obsidian flex items-center gap-3">
          <div className="w-2 h-2 bg-accents-sage rounded-full animate-pulse"></div>
          Agent Selection
        </h3>
        {isActive && (
          <div className="px-3 py-1.5 bg-accents-sage/10 text-accents-sage rounded-full text-sm font-medium border border-accents-sage/20">
            Active
          </div>
        )}
      </div>

      {error && (
        <div className="mb-6 p-4 bg-accents-coral/5 border border-accents-coral/20 rounded-lg text-accents-coral text-sm">
          <div className="font-medium mb-1">Connection Error</div>
          <div className="opacity-80">{error}</div>
        </div>
      )}

      {/* Agent Dropdown */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-primary-charcoal mb-3">
          Current Agent
        </label>
        <select
          value={selectedAgent}
          onChange={(e) => handleAgentChange(e.target.value)}
          disabled={isLoading || isActive}
          className={`w-full px-4 py-3 border border-primary-neutral/40 rounded-lg bg-primary-whisper backdrop-blur-sm 
            focus:outline-none focus:ring-2 focus:ring-accents-sage/30 focus:border-accents-sage/50
            disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200
            text-primary-obsidian font-regular
            ${isLoading ? 'animate-pulse' : ''}
            hover:border-accents-sage/40`}
        >
          {availableAgents.map((agent) => (
            <option key={agent.id || agent} value={agent.id || agent}>
              {agent.name || agent}
            </option>
          ))}
        </select>
      </div>

      {/* Current Agent Info */}
      <div className="space-y-4">
        <div className="p-4 bg-primary-whisper/50 rounded-lg border border-primary-neutral/20">
          <h4 className="font-medium text-primary-obsidian mb-2">
            {currentAgentInfo.name}
          </h4>
          <p className="text-sm text-primary-charcoal leading-relaxed">
            {currentAgentInfo.description}
          </p>
        </div>

        {/* Agent Features */}
        {currentAgentInfo.features && currentAgentInfo.features.length > 0 && (
          <div>
            <p className="text-sm font-medium text-primary-charcoal mb-3">Capabilities:</p>
            <div className="flex flex-wrap gap-2">
              {currentAgentInfo.features.map((feature, index) => (
                <span
                  key={index}
                  className="px-3 py-1.5 bg-accents-sage/8 text-accents-sage rounded-md text-xs font-medium border border-accents-sage/15 transition-colors hover:bg-accents-sage/12"
                >
                  {feature}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Agent Status */}
        <div className="flex items-center justify-between pt-4 border-t border-primary-neutral/20">
          <span className="text-sm text-primary-charcoal">Status:</span>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full transition-colors ${
              isActive ? 'bg-accents-sage animate-pulse' : 'bg-semantic-inactive'
            }`}></div>
            <span className="text-sm font-medium text-primary-obsidian">
              {isActive ? 'Monitoring' : 'Ready'}
            </span>
          </div>
        </div>
      </div>

      {isActive && (
        <div className="mt-6 p-4 bg-accents-amber/5 border border-accents-amber/20 rounded-lg">
          <p className="text-sm text-amber-800 leading-relaxed">
            <span className="font-medium">Note:</span> Stop monitoring to switch agents
          </p>
        </div>
      )}
    </div>
  )
}

export default AgentSelector