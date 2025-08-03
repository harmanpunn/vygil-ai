// Summary dashboard showing overall multi-agent system status
import { useState, useEffect } from 'react'

const SystemDashboard = () => {
  const [systemStats, setSystemStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const [healthResponse, agentsResponse, statsResponse] = await Promise.all([
          fetch('/api/health'),
          fetch('/api/agents'),
          fetch('/api/stats')
        ])

        const health = await healthResponse.json()
        const agents = await agentsResponse.json()
        const stats = await statsResponse.json()

        setSystemStats({
          health,
          agents: agents.agents,
          stats
        })
      } catch (error) {
        console.error('Failed to fetch system stats:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchStats()
    const interval = setInterval(fetchStats, 30000) // Update every 30 seconds
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-primary-whisper/20">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-2">
            <div className="h-3 bg-gray-200 rounded"></div>
            <div className="h-3 bg-gray-200 rounded w-5/6"></div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 border border-primary-whisper/20">
      <h3 className="text-lg font-semibold text-primary-dark mb-4 flex items-center gap-2">
        <div className="w-2 h-2 bg-accents-sage rounded-full"></div>
        System Status
      </h3>

      <div className="space-y-4">
        {/* System Health */}
        <div className="flex items-center justify-between p-3 bg-primary-ghost/30 rounded-lg">
          <span className="text-sm font-medium text-primary-dark">Backend Status</span>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              systemStats?.health?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
            }`}></div>
            <span className="text-sm text-primary-muted capitalize">
              {systemStats?.health?.status || 'Unknown'}
            </span>
          </div>
        </div>

        {/* Available Agents */}
        <div>
          <div className="text-sm font-medium text-primary-dark mb-2">
            Available Agents ({systemStats?.agents?.length || 0})
          </div>
          <div className="space-y-2">
            {systemStats?.agents?.map((agent) => (
              <div key={agent.id} className="flex items-center justify-between p-2 bg-primary-ghost/20 rounded-lg">
                <div className="flex-1">
                  <div className="text-sm font-medium text-primary-dark">{agent.name}</div>
                  <div className="text-xs text-primary-muted">{agent.description}</div>
                </div>
                <div className="flex flex-wrap gap-1">
                  {agent.features?.slice(0, 2).map((feature, index) => (
                    <span key={index} className="text-xs bg-accents-sage/10 text-accents-sage px-2 py-1 rounded">
                      {feature}
                    </span>
                  ))}
                  {agent.features?.length > 2 && (
                    <span className="text-xs text-primary-muted">+{agent.features.length - 2}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Activity Stats */}
        {systemStats?.stats && (
          <div className="grid grid-cols-2 gap-3">
            <div className="text-center p-3 bg-blue-50 rounded-lg">
              <div className="text-lg font-bold text-blue-700">
                {systemStats.stats.total_activities || 0}
              </div>
              <div className="text-xs text-blue-600">Total Activities</div>
            </div>
            <div className="text-center p-3 bg-green-50 rounded-lg">
              <div className="text-lg font-bold text-green-700">
                {systemStats.stats.average_confidence?.toFixed(1) || '0.0'}%
              </div>
              <div className="text-xs text-green-600">Avg Confidence</div>
            </div>
          </div>
        )}

        {/* Quick Actions */}
        <div className="pt-3 border-t border-primary-whisper/30">
          <div className="text-sm font-medium text-primary-dark mb-2">Quick Actions</div>
          <div className="grid grid-cols-2 gap-2">
            <button 
              onClick={() => window.location.reload()}
              className="px-3 py-2 bg-primary-whisper text-primary-dark rounded-lg text-sm hover:bg-primary-whisper/80 transition-colors"
            >
              Refresh
            </button>
            <button 
              onClick={() => fetch('/api/activities', { method: 'DELETE' })}
              className="px-3 py-2 bg-red-100 text-red-700 rounded-lg text-sm hover:bg-red-200 transition-colors"
            >
              Clear Logs
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SystemDashboard
