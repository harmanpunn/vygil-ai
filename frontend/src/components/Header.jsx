import { useState, useEffect } from 'react'

const Header = ({ status }) => {
  const getStatusConfig = (status) => {
    switch (status) {
      case 'monitoring':
        return {
          text: 'Monitoring Active',
          bgColor: 'bg-semantic-active/10',
          textColor: 'text-semantic-active',
          dotColor: 'bg-semantic-active',
          pulse: true
        }
      case 'error':
        return {
          text: 'Error State',
          bgColor: 'bg-semantic-error/10',
          textColor: 'text-semantic-error',
          dotColor: 'bg-semantic-error',
          pulse: false
        }
      default:
        return {
          text: 'Ready to Monitor',
          bgColor: 'bg-semantic-inactive/10',
          textColor: 'text-semantic-inactive',
          dotColor: 'bg-semantic-inactive',
          pulse: false
        }
    }
  }

  const statusConfig = getStatusConfig(status)

  return (
    <header className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50">
      <div 
        className="bg-primary-ghost/85 backdrop-blur border border-primary-slate/30 
                   h-14 px-6 rounded-xl shadow-lg
                   flex items-center justify-between
                   max-w-xl w-full mx-auto
                   transition-all duration-200 ease-out
                   hover:shadow-xl hover:-translate-y-0.5"
      >
        {/* Logo */}
        <div className="flex items-center space-x-2">
          {/* Target Icon */}
          <svg 
            className="w-5 h-5 text-primary-obsidian" 
            viewBox="0 0 24 24" 
            fill="none" 
            stroke="currentColor"
          >
            <circle cx="12" cy="12" r="10"/>
            <circle cx="12" cy="12" r="6"/>
            <circle cx="12" cy="12" r="2"/>
          </svg>
          
          <h1 className="text-lg font-medium text-primary-obsidian">
            Vygil
          </h1>
        </div>

        {/* Status Badge */}
        <div 
          className={`flex items-center space-x-2 px-3 py-1 rounded-full
                     ${statusConfig.bgColor} ${statusConfig.textColor}
                     transition-all duration-200 ease-out`}
        >
          {/* Status Dot */}
          <div 
            className={`w-2 h-2 rounded-full ${statusConfig.dotColor}
                       ${statusConfig.pulse ? 'animate-pulse' : ''}`}
          />
          
          <span className="text-sm font-normal">
            {statusConfig.text}
          </span>
        </div>
      </div>
    </header>
  )
}

export default Header