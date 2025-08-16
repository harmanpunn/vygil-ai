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
    <div className="bg-primary-ghost border border-primary-neutral/30 rounded-xl p-6 transition-all duration-200 hover:transform hover:-translate-y-0.5 hover:shadow-lg shadow-primary-obsidian/8">
      <h3 className="text-lg font-medium text-primary-obsidian mb-6 flex items-center gap-3">
        <div className="w-2 h-2 bg-accents-frost rounded-full animate-pulse"></div>
        System Status
      </h3>

      <div className="space-y-6">
        {/* System Health */}
        <div className="flex items-center justify-between p-4 bg-primary-whisper/50 rounded-lg border border-primary-neutral/20">
          <span className="text-sm font-medium text-primary-charcoal">Backend Status</span>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full transition-colors ${
              systemStats?.health?.status === 'healthy' ? 'bg-accents-sage animate-pulse' : 'bg-accents-coral'
            }`}></div>
            <span className="text-sm font-medium text-primary-obsidian capitalize">
              {systemStats?.health?.status || 'Unknown'}
            </span>
          </div>
        </div>

        {/* Available Agents Count */}
        <div className="flex items-center justify-between p-4 bg-primary-whisper/50 rounded-lg border border-primary-neutral/20">
          <span className="text-sm font-medium text-primary-charcoal">Available Agents</span>
          <span className="text-lg font-light font-mono text-primary-obsidian">
            {systemStats?.agents?.length || 0}
          </span>
        </div>

        {/* Activity Stats */}
        {systemStats?.stats && (
          <div className="grid grid-cols-2 gap-3">
            <div className="text-center p-4 bg-accents-frost/5 rounded-lg border border-accents-frost/15 transition-all hover:transform hover:scale-102">
              <div className="text-2xl font-light font-mono text-primary-obsidian mb-1">
                {systemStats.stats.total_activities || 0}
              </div>
              <div className="text-xs text-primary-charcoal font-medium">Activities Captured</div>
            </div>
            <div className="text-center p-4 bg-accents-sage/5 rounded-lg border border-accents-sage/15 transition-all hover:transform hover:scale-102">
              <div className="text-2xl font-light font-mono text-primary-obsidian mb-1">
                {systemStats.stats.average_confidence?.toFixed(1) || '0.0'}%
              </div>
              <div className="text-xs text-primary-charcoal font-medium">Avg Confidence</div>
            </div>
          </div>
        )}

        {/* Quick Actions */}
        <div className="pt-6 border-t border-primary-neutral/20">
          <div className="text-sm font-medium text-primary-charcoal mb-3">Quick Actions</div>
          <div className="grid grid-cols-2 gap-3">
            <button 
              onClick={() => window.location.reload()}
              className="px-4 py-2.5 bg-primary-neutral/20 text-primary-charcoal rounded-lg text-sm hover:bg-primary-neutral/30 transition-all duration-200 hover:transform hover:-translate-y-0.5 font-medium"
            >
              Refresh
            </button>
            <button 
              onClick={async () => {
                try {
                  await fetch('/api/activities', { method: 'DELETE' })
                  // Broadcast a custom event so ActivityTracker can reset UI immediately
                  window.dispatchEvent(new CustomEvent('vygil:clear-logs'))
                } catch (e) {
                  console.error('Failed to clear logs', e)
                }
              }}
              className="px-4 py-2.5 bg-accents-coral/10 text-accents-coral rounded-lg text-sm hover:bg-accents-coral/15 transition-all duration-200 hover:transform hover:-translate-y-0.5 font-medium border border-accents-coral/20"
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
