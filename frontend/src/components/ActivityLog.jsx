const ActivityLog = ({ activities, isEmpty }) => {
  return (
    <div className="mt-12">
      {/* Header with Organic Divider */}
      <div className="relative mb-6">
        <h2 className="text-lg font-medium text-primary-obsidian">
          Activity Insights
        </h2>
        <div className="absolute top-8 left-0 w-full h-px 
                       bg-gradient-to-r from-primary-neutral via-primary-slate/50 to-transparent">
        </div>
      </div>

      {/* Empty State */}
      {isEmpty ? (
        <div className="text-center py-12 px-6">
          {/* Activity Wave Illustration */}
          <div className="flex justify-center mb-6">
            <svg 
              className="w-16 h-16 text-primary-slate opacity-60 animate-bounce" 
              fill="none" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={1} 
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" 
              />
            </svg>
          </div>
          
          <div className="space-y-1">
            <p className="text-base font-medium text-primary-charcoal">
              Ready to begin insights
            </p>
            <p className="text-sm font-normal text-primary-charcoal opacity-80">
              Start monitoring to capture your digital patterns
            </p>
          </div>
        </div>
      ) : (
        /* Activity Stream */
        <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
          {activities.map((activity, index) => (
            <div
              key={activity.id}
              className="bg-primary-ghost border border-primary-neutral/60 
                        rounded-lg p-4 relative
                        transition-all duration-200 ease-out
                        hover:shadow-floating hover:border-primary-slate/60
                        animate-fade-in"
              style={{
                animationDelay: `${index * 50}ms`,
                animationFillMode: 'both'
              }}
            >
              {/* Timestamp */}
              <div className="absolute top-4 right-4">
                <span className="text-xs font-mono text-semantic-inactive">
                  {new Date(activity.timestamp).toLocaleTimeString()}
                </span>
              </div>
              
              {/* Activity Description */}
              <div className="pr-20">
                <p className="text-base text-primary-obsidian leading-relaxed">
                  {activity.description}
                </p>
                
                {/* AI Insight Card */}
                <div className="bg-accents-frost/5 border border-accents-frost/10 
                               rounded-lg p-4 mt-2">
                  <p className="text-sm text-accents-frost font-normal italic">
                    AI detected: {activity.confidence > 0.8 ? 'High confidence' : 
                                 activity.confidence > 0.6 ? 'Medium confidence' : 
                                 'Low confidence'} activity classification
                    <span className="ml-2 font-mono">
                      ({Math.round(activity.confidence * 100)}%)
                    </span>
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default ActivityLog