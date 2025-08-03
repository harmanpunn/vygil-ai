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
    <header className="fixed top-0 left-0 right-0 z-50 bg-primary-ghost/85 backdrop-blur border-b border-primary-slate/30">
      <div 
        className="max-w-6xl mx-auto px-6 h-14
                   flex items-center justify-between
                   transition-all duration-200 ease-out"
      >
        {/* Logo - Left Side */}
        <div className="flex items-center space-x-3">
          {/* Target Icon */}
          <svg 
            className="w-5 h-5 text-primary-obsidian flex-shrink-0" 
            viewBox="0 0 24 24" 
            fill="none" 
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="10"/>
            <circle cx="12" cy="12" r="6"/>
            <circle cx="12" cy="12" r="2"/>
          </svg>
          
          <h1 className="text-lg font-medium text-primary-obsidian leading-none">
            Vygil
          </h1>
        </div>

        {/* Status Badge - Right Side */}
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