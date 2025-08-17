// Desktop notification system for focus assistant
import { useEffect, useState } from 'react'

const NotificationSystem = ({ currentAgent, isActive }) => {
  const [notificationsEnabled, setNotificationsEnabled] = useState(false)
  const [lastNotification, setLastNotification] = useState(null)

  // Request notification permissions on mount
  useEffect(() => {
    if ('Notification' in window) {
      if (Notification.permission === 'granted') {
        setNotificationsEnabled(true)
      } else if (Notification.permission !== 'denied') {
        Notification.requestPermission().then(permission => {
          setNotificationsEnabled(permission === 'granted')
        })
      }
    }
  }, [])

  // Mock distraction detection for demo purposes
  useEffect(() => {
    if (!isActive || currentAgent !== 'vygil-focus-assistant' || !notificationsEnabled) {
      return
    }

    // Simulate distraction detection every 2 minutes (for demo)
    const interval = setInterval(() => {
      // Only show notification if last one was more than 5 minutes ago (cooldown)
      const now = Date.now()
      if (!lastNotification || (now - lastNotification) > 5 * 60 * 1000) {
        showFocusNotification()
        setLastNotification(now)
      }
    }, 2 * 60 * 1000) // 2 minutes

    return () => clearInterval(interval)
  }, [isActive, currentAgent, notificationsEnabled, lastNotification])

  const showFocusNotification = () => {
    if (!notificationsEnabled) return

    const notification = new Notification('🎯 Focus Assistant', {
      body: 'Stay focused! You seem to be getting distracted.',
      icon: '/vite.svg',
      tag: 'focus-reminder',
      requireInteraction: false
    })

    // Auto-close after 5 seconds
    setTimeout(() => {
      notification.close()
    }, 5000)

    notification.onclick = () => {
      window.focus()
      notification.close()
    }
  }

  const requestPermissions = async () => {
    if ('Notification' in window) {
      const permission = await Notification.requestPermission()
      setNotificationsEnabled(permission === 'granted')
    }
  }

  const testNotification = () => {
    if (notificationsEnabled) {
      showFocusNotification()
      setLastNotification(Date.now())
    }
  }

  if (currentAgent !== 'vygil-focus-assistantt') {
    return null
  }

  return (
    <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
      <h4 className="font-medium text-yellow-800 mb-2 flex items-center gap-2">
        🔔 Focus Notifications
      </h4>
      
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-yellow-700">
            Desktop notifications: 
          </span>
          <span className={`text-xs px-2 py-1 rounded-full ${
            notificationsEnabled 
              ? 'bg-green-100 text-green-700' 
              : 'bg-red-100 text-red-700'
          }`}>
            {notificationsEnabled ? 'Enabled' : 'Disabled'}
          </span>
        </div>

        {!notificationsEnabled && (
          <button
            onClick={requestPermissions}
            className="w-full px-3 py-2 bg-yellow-600 text-white rounded-lg text-sm hover:bg-yellow-700 transition-colors"
          >
            Enable Notifications
          </button>
        )}

        {notificationsEnabled && (
          <button
            onClick={testNotification}
            className="w-full px-3 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 transition-colors"
          >
            Test Notification
          </button>
        )}

        {lastNotification && (
          <div className="text-xs text-yellow-600">
            Last notification: {new Date(lastNotification).toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  )
}

export default NotificationSystem
