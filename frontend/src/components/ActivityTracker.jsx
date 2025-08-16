import { useState, useEffect, useRef, useCallback } from 'react'
import PermissionNotification from './PermissionNotification'
import ControlPanel from './ControlPanel'
import MetricsDisplay from './MetricsDisplay'
import ActivityLog from './ActivityLog'
import NotificationSystem from './NotificationSystem'

const ActivityTracker = ({ onStatusChange, currentAgent }) => {
  const [isActive, setIsActive] = useState(false)
  const [activities, setActivities] = useState([])
  const [status, setStatus] = useState('Ready to begin insights')
  const [sessionTime, setSessionTime] = useState(0)
  const [nextCapture, setNextCapture] = useState(null)
  const [stream, setStream] = useState(null)
  const [error, setError] = useState(null)
  const [focusSummary, setFocusSummary] = useState(null)
  
  const intervalRef = useRef(null)
  const sessionStartRef = useRef(null)
  const canvasRef = useRef(null)
  const videoRef = useRef(null)
  const displayVideoRef = useRef(null)
  const isActiveRef = useRef(false)
  const streamRef = useRef(null)

  // Notify parent of status changes
  useEffect(() => {
    if (isActive) {
      onStatusChange('monitoring')
    } else if (error) {
      onStatusChange('error')
    } else {
      onStatusChange('stopped')
    }
  }, [isActive, error, onStatusChange])

  // Keep isActiveRef in sync with isActive state
  useEffect(() => {
    isActiveRef.current = isActive
    console.log(`🔄 isActiveRef updated to: ${isActive}`)
  }, [isActive])

  // Keep streamRef in sync with stream state
  useEffect(() => {
    streamRef.current = stream
    console.log(`🎥 streamRef updated:`, { hasStream: !!stream })
  }, [stream])

  // Fetch focus summary when using focus assistant
  useEffect(() => {
    const fetchFocusSummary = async () => {
      if (currentAgent === 'vygil-focus-assistant' && isActive) {
        try {
          const response = await fetch('/api/focus/summary')
          if (response.ok) {
            const data = await response.json()
            setFocusSummary(data.summary)
            console.log('Focus Summary: ', data.summary)
          }
        } catch (err) {
          console.error('Failed to fetch focus summary:', err)
        }
      } else {
        setFocusSummary(null)
      }
    }

    // Fetch immediately and then every 30 seconds
    fetchFocusSummary()
    const interval = setInterval(fetchFocusSummary, 30000)
    return () => clearInterval(interval)
  }, [currentAgent, isActive])

  // Format time as MM:SS
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  // Start screen capture
  const startScreenCapture = async () => {
    try {
      setError(null)
      setStatus('Requesting screen permission...')
      
      const mediaStream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          mediaSource: 'screen',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        },
        audio: false
      })
      
      setStream(mediaStream)
      setStatus('Screen capture started')
      
      // Set up both video elements - one for capture, one for display
      const setupVideo = async (videoElement) => {
        if (!videoElement || !mediaStream) return false
        
        return new Promise((resolve) => {
          videoElement.srcObject = mediaStream
          
          const onLoadedMetadata = () => {
            videoElement.play().then(() => {
              console.log('Video started playing successfully')
              resolve(true)
            }).catch((err) => {
              console.error('Video play failed:', err)
              resolve(false)
            })
            videoElement.removeEventListener('loadedmetadata', onLoadedMetadata)
          }
          
          const onError = (err) => {
            console.error('Video load error:', err)
            videoElement.removeEventListener('error', onError)
            resolve(false)
          }
          
          videoElement.addEventListener('loadedmetadata', onLoadedMetadata)
          videoElement.addEventListener('error', onError)
          
          // Fallback timeout
          setTimeout(() => {
            videoElement.removeEventListener('loadedmetadata', onLoadedMetadata)
            videoElement.removeEventListener('error', onError)
            resolve(false)
          }, 5000)
        })
      }
      
      // Setup capture video (hidden)
      await setupVideo(videoRef.current)
      
      // Setup display video if it exists
      if (displayVideoRef.current) {
        await setupVideo(displayVideoRef.current)
      }
      
      return true
    } catch (err) {
      console.error('Screen capture failed:', err)
      setError('Screen capture permission denied or failed')
      setStatus('Screen capture failed')
      return false
    }
  }

  // Save screenshot via API (browser can't write directly to filesystem)
  const saveScreenshotLocally = async (dataUrl) => {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
      const filename = `vygil-screenshot-${timestamp}.png`
      
      // Extract base64 data
      const base64Data = dataUrl.split(',')[1]
      
      // Send to backend to save in screenshots directory
      const response = await fetch('/api/save-screenshot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filename: filename,
          imageData: base64Data
        })
      })
      
      if (response.ok) {
        console.log(`💾 Screenshot saved to: /Users/harman/Home/Projects/vygil-ai/screenshots/${filename}`)
      } else {
        throw new Error(`Failed to save screenshot: ${response.status}`)
      }
    } catch (err) {
      console.error('❌ Failed to save screenshot:', err)
      // Fallback to browser download
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
      const filename = `vygil-screenshot-${timestamp}.png`
      const link = document.createElement('a')
      link.download = filename
      link.href = dataUrl
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      console.log(`💾 Screenshot downloaded as fallback: ${filename}`)
    }
  }

  // Capture screen frame as base64
  const captureFrame = useCallback(async () => {
    console.log('🎯 captureFrame called!', { 
      isActive: isActiveRef.current, 
      hasStream: !!streamRef.current, 
      hasVideo: !!videoRef.current, 
      hasCanvas: !!canvasRef.current,
      videoReady: videoRef.current?.readyState 
    })
    
    if (!isActiveRef.current || !streamRef.current || !videoRef.current || !canvasRef.current) {
      console.warn('⚠️ Cannot capture frame:', {
        isActive: isActiveRef.current,
        hasStream: !!streamRef.current,
        hasVideo: !!videoRef.current, 
        hasCanvas: !!canvasRef.current
      })
      return null
    }

    try {
      const video = videoRef.current
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      
      // Set canvas size to match video
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      
      console.log(`📸 Capturing screen frame: ${canvas.width}x${canvas.height}`)
      
      // Draw current video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      
      // Convert to base64
      const base64Image = canvas.toDataURL('image/png').split(',')[1]
      
      // Save screenshot locally for reference
      await saveScreenshotLocally(canvas.toDataURL('image/png'))
      
      console.log(`✅ Frame captured successfully (${Math.round(base64Image.length / 1024)}KB)`)
      return base64Image
      
    } catch (err) {
      console.error('❌ Frame capture failed:', err)
      return null
    }
  }, []) // No dependencies since we're using refs for everything

  

  // Send image to MCP server for processing
  const processActivity = async (base64Image) => {
    try {
      console.log('🌐 Making API call to: /api/process-activity')
      
      const response = await fetch('/api/process-activity', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Image,
          timestamp: new Date().toISOString()
        })
      })
      
      console.log(`📡 API Response Status: ${response.status} ${response.statusText}`)
      
      if (response.ok) {
        const result = await response.json()
        console.log('✅ API Response:', result)
        return {
          activity: result.activity || 'ACTIVITY: Unknown activity',
          confidence: result.confidence || 0.5
        }
      } else {
        const errorText = await response.text()
        throw new Error(`API request failed: ${response.status} - ${errorText}`)
      }
    } catch (err) {
      console.error('❌ API call failed:', err.message)
      
      // Check if it's a network error (backend not running)
      if (err.message.includes('fetch')) {
        console.warn('⚠️ Backend server not running. Using mock data.')
        setError('Backend server not available. Please start the API server.')
      }
      
      // Return mock data for demo purposes
      const mockActivities = [
        'ACTIVITY: Coding in VS Code',
        'ACTIVITY: Browsing documentation',
        'ACTIVITY: Video call meeting',
        'ACTIVITY: Reading email',
        'ACTIVITY: Writing notes'
      ]
      
      const randomActivity = mockActivities[Math.floor(Math.random() * mockActivities.length)]
      return {
        activity: randomActivity,
        confidence: Math.random() * 0.4 + 0.6 // 0.6 to 1.0
      }
    }
  }

  // Monitor cycle - capture and process
  const executeMonitoringCycle = useCallback(async () => {
    console.log('🚀 executeMonitoringCycle called!', { isActive: isActiveRef.current })
    
    if (!isActiveRef.current) {
      console.warn('⚠️ Monitoring cycle aborted - not active')
      return
    }
    
    const cycleStart = Date.now()
    console.log('\n🔄 === MONITORING CYCLE START ===')
    console.log(`⏰ Cycle started at: ${new Date().toLocaleTimeString()}`)
    
    setStatus('Capturing screen...')
    console.log('📸 About to call captureFrame()...')
    const base64Image = await captureFrame()
    
    if (!base64Image) {
      console.error('❌ Frame capture failed - aborting cycle')
      setStatus('Frame capture failed')
      return
    }
    
    setStatus('Processing activity...')
    console.log('🤖 Sending image to API for processing...')
    const result = await processActivity(base64Image)
    
    // Clean up the activity message for display
    const cleanActivityMessage = (activityText) => {
      // Remove "ACTIVITY: " prefix if present
      let cleaned = activityText.replace(/^ACTIVITY:\s*/, '')
      
      // Split into sentences and clean up
      const sentences = cleaned.split('.').map(s => s.trim()).filter(s => s.length > 0)
      
      if (sentences.length > 1) {
        const firstSentence = sentences[0]
        const secondSentence = sentences[1]
        
        // Check for repetitive patterns like "You've been browsing Instagram. Consider taking a break..."
        if (firstSentence.toLowerCase().includes('browsing') && 
            firstSentence.toLowerCase().includes('instagram') &&
            secondSentence.toLowerCase().includes('consider')) {
          // Return just the suggestion without repetition
          return sentences.slice(1).join('. ')
        }
        
        // Check for other repetitive patterns
        const activityKeywords = ['browsing', 'coding', 'writing', 'reading', 'working', 'watching']
        const hasRepetition = activityKeywords.some(keyword => 
          firstSentence.toLowerCase().includes(keyword) && 
          sentences.slice(1).join(' ').toLowerCase().includes(keyword)
        )
        
        if (hasRepetition) {
          // If there's a suggestion (Consider, Try, etc.), use that
          const suggestionSentences = sentences.slice(1)
          const hasSuggestion = suggestionSentences.some(s => 
            s.includes('Consider') || s.includes('Try') || s.includes('take a break')
          )
          
          if (hasSuggestion) {
            return suggestionSentences.join('. ')
          } else {
            return firstSentence
          }
        }
      }
      
      return cleaned
    }

    // Add new activity to log
    const newActivity = {
      id: Date.now(),
      description: result.activity,
      confidence: result.confidence,
      timestamp: new Date().toISOString()
    }
    
    setActivities(prev => [newActivity, ...prev.slice(0, 49)]) // Keep last 50
    
    // Set clean status message
    const cleanMessage = cleanActivityMessage(result.activity)
    setStatus(`Last activity: ${cleanMessage}`)
    
    const cycleTime = Date.now() - cycleStart
    console.log(`✅ Activity detected: ${result.activity}`)
    console.log(`📊 Confidence: ${(result.confidence * 100).toFixed(1)}%`)
    console.log(`⚡ Cycle completed in: ${cycleTime}ms`)
    console.log('🔄 === MONITORING CYCLE END ===\n')
    
  }, [captureFrame]) // Removed isActive dependency since we're using ref

  // Start monitoring
  const startMonitoring = async () => {
    if (isActive) return
    
    console.log('\n🚀 === STARTING MONITORING SESSION ===')
    console.log('📋 Initializing screen capture...')
    
    const captureStarted = await startScreenCapture()
    if (!captureStarted) {
      console.error('❌ Failed to start screen capture')
      return
    }
    
    setIsActive(true)
    sessionStartRef.current = Date.now()
    
    console.log('✅ Screen capture initialized successfully')
    console.log('⏰ Setting up 60-second monitoring interval')
    console.log('🎯 First capture will happen in 2 seconds...')
    
    // Start 60-second monitoring loop
    console.log('⚡ Creating 60-second interval...')
    intervalRef.current = setInterval(() => {
      console.log('⏰ 60-second interval triggered!', { 
        currentTime: new Date().toLocaleTimeString(),
        isActive: isActiveRef.current 
      })
      executeMonitoringCycle()
    }, 25000)
    
    console.log(`📋 Interval ID: ${intervalRef.current}`)
    
    // Execute first cycle immediately
    setTimeout(() => {
      console.log('🎬 Executing first monitoring cycle...', { isActive: isActiveRef.current })
      executeMonitoringCycle()
    }, 2000) // Give video time to load
  }

  // Stop monitoring
  const stopMonitoring = () => {
    setIsActive(false)
    setStatus('Monitoring stopped')
    setNextCapture(null)
    
    // Clear interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    
    // Stop screen capture and clean up video elements
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
    
    // Clean up video elements
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    if (displayVideoRef.current) {
      displayVideoRef.current.srcObject = null
    }
  }

  // Clear activity log
  const clearActivities = () => {
    setActivities([])
    setFocusSummary(null)
    setSessionTime(0)
    setNextCapture(null)
  }

  // Listen for global clear-logs to reset UI immediately
  useEffect(() => {
    const handleClear = () => {
      setActivities([])
      setFocusSummary(null)
      setSessionTime(0)
      setNextCapture(null)
      setStatus('Logs cleared')
    }
    window.addEventListener('vygil:clear-logs', handleClear)
    return () => window.removeEventListener('vygil:clear-logs', handleClear)
  }, [])

  // Update session timer and countdown
  useEffect(() => {
    if (!isActive) return
    
    const timer = setInterval(() => {
      if (sessionStartRef.current) {
        const elapsed = Math.floor((Date.now() - sessionStartRef.current) / 1000)
        setSessionTime(elapsed)
      }
      
      // Calculate time until next capture
      if (intervalRef.current && sessionStartRef.current) {
        const elapsed = (Date.now() - sessionStartRef.current) / 1000
        // Account for the 2-second initial delay
        const adjustedElapsed = elapsed - 2
        const nextIn = adjustedElapsed > 0 ? 25 - (adjustedElapsed % 25) : 2 - elapsed
        const countdown = Math.floor(Math.max(0, nextIn))
        setNextCapture(countdown)
        
        // Debug log for countdown
        if (countdown <= 5 && countdown > 0) {
          console.log(`⏰ Next capture in: ${countdown} seconds`)
        }
      }
    }, 1000)
    
    return () => clearInterval(timer)
  }, [isActive])

    // Handle stream ended event
  useEffect(() => {
    if (!stream) return

    const handleStreamEnd = () => {
      console.log('Screen sharing ended by user')
      stopMonitoring()
    }

    // Listen for track ended events
    stream.getTracks().forEach(track => {
      track.addEventListener('ended', handleStreamEnd)
    })

    return () => {
      stream.getTracks().forEach(track => {
        track.removeEventListener('ended', handleStreamEnd)
      })
    }
  }, [stream])

  // Update display video when stream changes
  useEffect(() => {
    if (stream && displayVideoRef.current && isActive) {
      const displayVideo = displayVideoRef.current
      displayVideo.srcObject = stream
      displayVideo.play().catch((err) => {
        console.error('Display video play failed:', err)
      })
    }
  }, [stream, isActive])

  return (
    <div className="space-y-8">
      {/* Always present video and canvas elements for screen capture - never removed from DOM */}
      <video 
        ref={videoRef} 
        className="hidden" 
        muted 
        autoPlay
        playsInline
      />
      <canvas ref={canvasRef} className="hidden" />
      
      {/* Screen Preview Window - Show when monitoring is active and stream exists */}
      {isActive && stream && (
        <div className="bg-white rounded-lg shadow-lg overflow-hidden border border-primary-neutral">
          <div className="bg-primary-slate px-4 py-2 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-semantic-active rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-primary-charcoal">Screen Preview</span>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => {
                  console.log('🧪 Manual capture test triggered')
                  executeMonitoringCycle()
                }}
                className="px-2 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Test Capture
              </button>
              <span className="text-xs text-primary-charcoal opacity-75">Live</span>
            </div>
          </div>
          <div className="relative">
            <video 
              ref={displayVideoRef}
              className="w-full h-48 object-cover bg-primary-ghost" 
              muted 
              autoPlay
              playsInline
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/10 to-transparent pointer-events-none"></div>
          </div>
        </div>
      )}
      
      {/* Permission Notification */}
      {!isActive && !error && (
        <PermissionNotification />
      )}
      
      {/* Error Display */}
      {error && (
        <div className="bg-gradient-to-r from-red-50 to-red-100 border border-red-200 rounded-lg p-6 mb-8">
          <div className="flex items-start space-x-4">
            <div className="flex-shrink-0">
              <svg className="w-5 h-5 text-red-500 mt-1" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div>
              <h3 className="text-base font-medium text-red-800">Connection Error</h3>
              <p className="text-sm text-red-700 mt-1 opacity-80">{error}</p>
            </div>
          </div>
        </div>
      )}
      
      {/* Control Panel */}
      <ControlPanel 
        isActive={isActive}
        sessionTime={sessionTime}
        onStart={startMonitoring}
        onStop={stopMonitoring}
        onClear={clearActivities}
        status={status}
      />
      
      {/* Metrics Display */}
      <MetricsDisplay 
        activitiesCount={activities.length}
        sessionTime={sessionTime}
        nextCapture={nextCapture}
        formatTime={formatTime}
      />
      
      {/* Focus Summary (only for focus assistant) */}
      {currentAgent === 'vygil-focus-assistant' && focusSummary && (
        <div className="bg-primary-ghost border border-primary-neutral/30 rounded-xl p-6 transition-all duration-200 hover:transform hover:-translate-y-0.5 hover:shadow-lg shadow-primary-obsidian/8">
          <h3 className="text-lg font-medium text-primary-obsidian mb-6 flex items-center gap-3">
            <div className="w-2 h-2 bg-accents-amber rounded-full animate-pulse"></div>
            Focus Session Summary
          </h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-accents-amber/5 rounded-lg border border-accents-amber/15 transition-all hover:transform hover:scale-102">
              <div className="text-2xl font-light font-mono text-primary-obsidian mb-1">
                {focusSummary.productivity_score?.toFixed(1) || '0.0'}
              </div>
              <div className="text-xs text-primary-charcoal font-medium">Productivity Score</div>
            </div>
            
            <div className="text-center p-4 bg-accents-sage/5 rounded-lg border border-accents-sage/15 transition-all hover:transform hover:scale-102">
              <div className="text-2xl font-light font-mono text-primary-obsidian mb-1">
                {focusSummary.focus_sessions || 0}
              </div>
              <div className="text-xs text-primary-charcoal font-medium">Focus Sessions</div>
            </div>
            
            <div className="text-center p-4 bg-accents-coral/5 rounded-lg border border-accents-coral/15 transition-all hover:transform hover:scale-102">
              <div className="text-2xl font-light font-mono text-primary-obsidian mb-1">
                {focusSummary.distractions || 0}
              </div>
              <div className="text-xs text-primary-charcoal font-medium">Distractions</div>
            </div>
            
            <div className="text-center p-4 bg-accents-frost/5 rounded-lg border border-accents-frost/15 transition-all hover:transform hover:scale-102">
              <div className="text-2xl font-light font-mono text-primary-obsidian mb-1">
                {Math.round(focusSummary.total_focus_time / 60) || 0}m
              </div>
              <div className="text-xs text-primary-charcoal font-medium">Focus Time</div>
            </div>
          </div>
          
          {focusSummary.current_suggestion && (
            <div className="mt-4 flex items-center gap-2 text-sm">
              <span className="text-accents-amber">💡</span>
              <span className="font-medium text-primary-charcoal">AI Insight:</span>
              <span className="text-primary-obsidian italic">{focusSummary.current_suggestion}</span>
            </div>
          )}
        </div>
      )}
      
      {/* Agent-specific status display */}
      {currentAgent && (
        <div className="bg-primary-ghost/30 rounded-xl p-4 border border-primary-whisper/20">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-primary-dark">
              Active Agent: {currentAgent === 'vygil-focus-assistant' ? 'Focus Assistant' : 'Activity Tracker'}
            </span>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-500' : 'bg-gray-400'}`}></div>
              <span className="text-sm text-primary-muted">
                {isActive ? 'Monitoring' : 'Stopped'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Notification System - Focus Assistant only */}
      <NotificationSystem 
        currentAgent={currentAgent}
        isActive={isActive}
      />
      
      {/* Activity Log */}
      <ActivityLog 
        activities={activities}
        isEmpty={activities.length === 0}
      />
    </div>
  )
}

export default ActivityTracker