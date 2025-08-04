const ActivityLog = ({ activities, isEmpty }) => {
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toISOString().slice(0, 19).replace('T', ' ');
  };

  const getLogLevel = (confidence) => {
    if (confidence > 0.7) return { level: 'INFO', color: 'text-green-400' };
    if (confidence > 0.4) return { level: 'WARN', color: 'text-yellow-400' };
    return { level: 'DEBUG', color: 'text-blue-400' };
  };

  return (
    <div className="mt-8">
      {/* Terminal Header */}
      <div className="mb-4">
        <div className="flex items-center space-x-2 mb-2">
          <span className="text-green-400 font-mono text-sm">$</span>
          <span className="text-gray-300 font-mono text-sm">tail -f /var/log/vygil/activity.log</span>
        </div>
        <div className="h-px bg-gray-600"></div>
      </div>

      {/* Terminal Window */}
      <div className="bg-black border border-gray-700 rounded-lg overflow-hidden">
        {/* Terminal Top Bar */}
        <div className="bg-gray-800 px-4 py-2 flex items-center space-x-2">
          <div className="flex space-x-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          </div>
          <span className="text-gray-400 text-xs font-mono ml-4">vygil-activity-monitor</span>
        </div>

        {/* Terminal Content */}
        <div className="bg-black p-4 font-mono text-sm min-h-[300px] max-h-96 overflow-y-auto">
          {isEmpty ? (
            <div className="text-gray-500 text-center py-8">
              <div className="animate-pulse">
                <span className="text-gray-600">Waiting for activity data...</span>
              </div>
              <div className="mt-2 text-xs">
                <span className="text-gray-700">No entries in activity.log</span>
              </div>
            </div>
          ) : (
            <div className="space-y-1">
              {activities.map((activity, index) => {
                const logInfo = getLogLevel(activity.confidence);
                const timestamp = formatTimestamp(activity.timestamp);
                
                return (
                  <div 
                    key={activity.id} 
                    className="leading-relaxed hover:bg-gray-900 px-2 py-1 rounded transition-colors duration-150"
                  >
                    {/* Log Line */}
                    <div className="flex flex-wrap items-start space-x-2">
                      {/* Timestamp */}
                      <span className="text-gray-500 shrink-0">
                        [{timestamp}]
                      </span>
                      
                      {/* Log Level */}
                      <span className={`${logInfo.color} shrink-0 font-bold`}>
                        {logInfo.level}
                      </span>
                      
                      {/* Process */}
                      <span className="text-gray-400 shrink-0">
                        [vygil-agent]:
                      </span>
                      
                      {/* Activity Message */}
                      <span className="text-gray-300">
                        {activity.description}
                      </span>
                    </div>
                    
                    {/* Confidence as a separate debug line */}
                    <div className="flex space-x-2 mt-1 text-xs">
                      <span className="text-gray-600">
                        [{timestamp}]
                      </span>
                      <span className="text-gray-500">
                        DEBUG
                      </span>
                      <span className="text-gray-500">
                        [vygil-agent]:
                      </span>
                      <span className="text-gray-500">
                        confidence_score={activity.confidence.toFixed(3)} 
                      </span>
                    </div>
                  </div>
                );
              })}
              
              {/* Cursor blink */}
              <div className="flex items-center mt-2">
                <span className="text-gray-500 mr-2">
                  [{formatTimestamp(new Date())}]
                </span>
                <span className="text-gray-400">INFO</span>
                <span className="text-gray-500 ml-2">[vygil-agent]:</span>
                <span className="text-gray-400 ml-2">monitoring active</span>
                <span className="animate-pulse text-green-400 ml-1">▊</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ActivityLog