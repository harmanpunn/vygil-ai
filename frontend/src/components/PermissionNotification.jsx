const PermissionNotification = () => {
  return (
    <div className="bg-gradient-to-br from-primary-ghost to-primary-whisper 
                   border border-primary-slate/40 rounded-lg p-6 mb-8
                   transition-all duration-200 ease-out
                   hover:shadow-lg hover:bg-gradient-to-br hover:from-primary-whisper hover:to-primary-neutral">
      <div className="flex items-center space-x-4">
        {/* Shield Icon */}
        <div className="flex-shrink-0">
          <svg 
            className="w-6 h-6 text-accents-sage" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={1.5} 
              d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.031 9-11.622 0-1.042-.133-2.052-.382-3.016z" 
            />
          </svg>
        </div>
        
        <div className="flex-1">
          <h3 className="text-base font-medium text-primary-obsidian mb-1 leading-none">
            Screen Permission Required
          </h3>
          <p className="text-sm font-normal text-primary-charcoal leading-relaxed">
            Click "Start Monitoring" to select which screen or window to track. 
            Your screen data is processed securely and only activity insights are stored.
          </p>
        </div>
      </div>
    </div>
  )
}

export default PermissionNotification