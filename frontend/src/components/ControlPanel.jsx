const ControlPanel = ({ isActive, sessionTime, onStart, onStop, onClear, status }) => {
  // Format time as MM:SS for display
  const formatDisplayTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="bg-primary-whisper border border-primary-neutral 
                   rounded-lg p-6 text-center
                   transition-all duration-200 ease-out">
      
      {/* Session Timer */}
      <div className="mb-4">
        <div className={`text-3xl font-light text-primary-obsidian font-mono
                        ${isActive ? 'animate-breathe' : ''}`}>
          {formatDisplayTime(sessionTime)}
        </div>
        <div className="text-sm text-primary-charcoal mt-1 opacity-75">
          {isActive ? 'Session Active' : 'Session Paused'}
        </div>
      </div>
      
      {/* Status Message */}
      <div className="mb-6">
        <p className="text-base text-primary-charcoal font-normal">
          {status}
        </p>
      </div>
      
      {/* Control Buttons */}
      <div className="flex items-center justify-center space-x-4">
        {!isActive ? (
          <button
            onClick={onStart}
            className="bg-transparent border-2 border-accents-sage text-accents-sage
                     px-6 py-3 rounded-lg text-base font-medium
                     transition-all duration-200 ease-out
                     hover:bg-accents-sage/8 hover:-translate-y-0.5 hover:shadow-lg
                     focus:outline-none focus:ring-2 focus:ring-accents-sage/20 focus:ring-offset-2
                     active:bg-accents-sage active:text-primary-ghost"
          >
            <div className="flex items-center space-x-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>Start Monitoring</span>
            </div>
          </button>
        ) : (
          <button
            onClick={onStop}
            className="bg-transparent border-2 border-semantic-error text-semantic-error
                     px-6 py-3 rounded-lg text-base font-medium
                     transition-all duration-200 ease-out
                     hover:bg-semantic-error/8 hover:-translate-y-0.5 hover:shadow-lg
                     focus:outline-none focus:ring-2 focus:ring-semantic-error/20 focus:ring-offset-2
                     active:bg-semantic-error active:text-primary-ghost
                     animate-pulse"
          >
            <div className="flex items-center space-x-2">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 9l6 6m0-6l-6 6" />
              </svg>
              <span>Stop Monitoring</span>
            </div>
          </button>
        )}
        
        {/* Clear Log Button */}
        <button
          onClick={onClear}
          className="bg-transparent border-none text-primary-charcoal
                   px-4 py-2 text-sm font-normal opacity-70
                   transition-all duration-200 ease-out
                   hover:opacity-100 hover:text-primary-obsidian
                   focus:outline-none focus:ring-2 focus:ring-primary-slate/20 focus:ring-offset-2"
        >
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            <span>Clear Log</span>
          </div>
        </button>
      </div>
    </div>
  )
}

export default ControlPanel