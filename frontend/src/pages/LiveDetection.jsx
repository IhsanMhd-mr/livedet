import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import useWebSocket from '@/hooks/useWebSocket'
import StatusBadge from '@/components/ui/StatusBadge'
import DetectionCard from '@/components/ui/DetectionCard'
import toast from 'react-hot-toast'

const SEVERITY_COLORS = {
  Low:      '#10b981',
  Medium:   '#eab308',
  High:     '#f97316',
  Critical: '#ef4444',
}

const VEHICLE_PRESETS = [
  {
    id: 'sedan',
    name: 'Sedan (Default)',
    icon: '🚗',
    dimensions: 'H: 1.2m · A: 45° · F: 600px',
    desc: 'Standard passenger car mount height and tilt angle.',
    active: true,
  },
  {
    id: 'suv',
    name: 'SUV / Crossover',
    icon: '🚙',
    dimensions: 'H: 1.5m · A: 40° · F: 600px',
    desc: 'Mid-size utility vehicle mounting configuration.',
    active: false,
  },
  {
    id: 'van',
    name: 'Delivery Van',
    icon: '🚐',
    dimensions: 'H: 2.0m · A: 35° · F: 600px',
    desc: 'Commercial delivery van mounting parameters.',
    active: false,
  },
  {
    id: 'truck',
    name: 'Heavy Truck',
    icon: '🚛',
    dimensions: 'H: 2.8m · A: 30° · F: 600px',
    desc: 'Heavy transport vehicle front-cab mounting config.',
    active: false,
  }
]

export default function LiveDetection() {
  const {
    connect,
    disconnect,
    wsStatus,
    currentFrame,
    detections,
    fps,
    depthActive,
    send,
    reconnectAttempt,
    maxReconnectAttempts,
    currentReconnectDelay,
    isReconnecting,
    hasConnectedBefore,
    hasReconnectExhausted
  } = useWebSocket('', false)

  const isConnected = wsStatus === 'connected'
  const isConnecting = wsStatus === 'connecting'

  const [selectedPreset, setSelectedPreset] = useState('sedan')
  const [isDropdownOpen, setIsDropdownOpen] = useState(false)
  const [isCameraActive, setIsCameraActive] = useState(false)

  // Derive refined WebSocket/Camera UI State
  let uiState = 'idle'
  if (isCameraActive) {
    if (isConnected) {
      uiState = 'connected'
    } else if (isReconnecting) {
      uiState = 'reconnecting'
    } else if (wsStatus === 'connecting') {
      uiState = 'connecting'
    } else if (wsStatus === 'error') {
      uiState = 'error'
    } else {
      uiState = 'connecting' // fallback
    }
  } else if (wsStatus === 'error') {
    uiState = 'error'
  }

  // Derive indicator status, label, and dynamic status description text
  let uiStatus = 'disconnected'
  let uiStatusLabel = 'Stopped'
  let uiDescription = 'Press Start Camera to begin live feed'
  let uiButtonText = 'Start Camera'

  if (uiState === 'idle') {
    uiStatus = 'disconnected'
    uiStatusLabel = 'Stopped'
    uiDescription = 'Press Start Camera to begin live feed'
    uiButtonText = 'Start Camera'
  } else if (uiState === 'connecting') {
    uiStatus = 'connecting'
    uiStatusLabel = 'Connecting'
    uiDescription = 'Connecting to detection server...'
    uiButtonText = 'Connecting...'
  } else if (uiState === 'connected') {
    uiStatus = 'connected'
    uiStatusLabel = 'Connected'
    uiDescription = 'Connected to detection server'
    uiButtonText = 'Stop Camera'
  } else if (uiState === 'reconnecting') {
    uiStatus = 'reconnecting'
    uiStatusLabel = `Reconnecting attempt ${reconnectAttempt} of ${maxReconnectAttempts}`
    uiDescription = `Reconnecting to detection server... Attempt ${reconnectAttempt} of ${maxReconnectAttempts}`
    uiButtonText = 'Stop Reconnecting'
  } else if (uiState === 'error') {
    uiStatus = 'error'
    uiStatusLabel = 'Server unavailable'
    uiDescription = 'Detection server unavailable after reconnect attempts. Please check that the backend is running and try again.'
    uiButtonText = 'Retry Connection'
  }

  // Driver Alert States
  const [audioAlertEnabled, setAudioAlertEnabled] = useState(false)
  const [proximityThreshold, setProximityThreshold] = useState(3.0)
  const [isAlertActive, setIsAlertActive] = useState(false)

  const videoRef        = useRef(null)
  const displayCanvasRef = useRef(null)
  const captureCanvasRef = useRef(null)
  const streamRef       = useRef(null)
  const frameIntervalRef = useRef(null)
  const animFrameRef    = useRef(null)
  const detectionsRef   = useRef([])
  const wsConnectedRef  = useRef(false)
  const dropdownRef      = useRef(null)

  // Refs for access in animation frame loop without stale closures
  const audioAlertEnabledRef = useRef(audioAlertEnabled)
  const proximityThresholdRef = useRef(proximityThreshold)
  const lastBeepTimeRef = useRef(0)

  // Refs for tracking state transitions
  const prevWsStatusRef = useRef(null)
  const prevIsReconnectingRef = useRef(false)
  const prevReconnectAttemptRef = useRef(0)
  const prevHasReconnectExhaustedRef = useRef(false)

  useEffect(() => {
    const prevWsStatus = prevWsStatusRef.current
    const prevIsReconnecting = prevIsReconnectingRef.current
    const prevReconnectAttempt = prevReconnectAttemptRef.current
    const prevHasReconnectExhausted = prevHasReconnectExhaustedRef.current

    // 1. Initial Connection Success
    if (wsStatus === 'connected' && prevWsStatus !== 'connected' && !isReconnecting && !prevIsReconnecting) {
      toast.success('WebSocket connected successfully', { id: 'ws-status' })
      console.log('%c✅ WebSocket connected successfully', 'color: green; font-weight: bold;')
    }

    // 2. Reconnect Success
    if (wsStatus === 'connected' && prevWsStatus !== 'connected' && prevIsReconnecting) {
      toast.success('WebSocket reconnected successfully', { id: 'ws-status' })
      console.log('%c✅ WebSocket reconnect successful', 'color: green; font-weight: bold;')
    }

    // 3. Reconnect Attempt
    if (isReconnecting && reconnectAttempt > 0 && reconnectAttempt !== prevReconnectAttempt) {
      const delayText = currentReconnectDelay ? ` in ${currentReconnectDelay}s` : ''
      console.log(`%c⚠️ WebSocket reconnect attempt ${reconnectAttempt} of ${maxReconnectAttempts}${delayText}...`, 'color: orange; font-weight: bold;')
    }

    // 4. Final Failure
    if (hasReconnectExhausted && !prevHasReconnectExhausted) {
      toast.error('Detection server unavailable after reconnect attempts. Please check that the backend is running and try again.', { id: 'ws-status' })
      console.log(`%c❌ WebSocket server unavailable after ${maxReconnectAttempts} reconnect attempts. Check backend server on port 8765.`, 'color: red; font-weight: bold;')
    }

    // 5. Initial Connection Failure (Optional)
    if (wsStatus === 'error' && prevWsStatus !== 'error' && !hasConnectedBefore && !isReconnecting && !hasReconnectExhausted) {
      toast.error('WebSocket connection failed', { id: 'ws-status' })
    }

    // Keep refs in sync
    prevWsStatusRef.current = wsStatus
    prevIsReconnectingRef.current = isReconnecting
    prevReconnectAttemptRef.current = reconnectAttempt
    prevHasReconnectExhaustedRef.current = hasReconnectExhausted
  }, [wsStatus, isReconnecting, reconnectAttempt, maxReconnectAttempts, currentReconnectDelay, hasReconnectExhausted, hasConnectedBefore])

  useEffect(() => { detectionsRef.current = detections }, [detections])
  useEffect(() => { wsConnectedRef.current = isConnected }, [isConnected])
  useEffect(() => { audioAlertEnabledRef.current = audioAlertEnabled }, [audioAlertEnabled])
  useEffect(() => { proximityThresholdRef.current = proximityThreshold }, [proximityThreshold])

  // Web Audio API driver alarm sound synthesis
  const playWarningSound = useCallback(() => {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext
      if (!AudioContext) return
      const ctx = new AudioContext()
      const now = ctx.currentTime
      
      const playTone = (time, freq, dur) => {
        const osc = ctx.createOscillator()
        const gain = ctx.createGain()
        osc.type = 'sine'
        osc.frequency.setValueAtTime(freq, time)
        
        gain.gain.setValueAtTime(0, time)
        gain.gain.linearRampToValueAtTime(0.3, time + 0.02)
        gain.gain.exponentialRampToValueAtTime(0.001, time + dur)
        
        osc.connect(gain)
        gain.connect(ctx.destination)
        osc.start(time)
        osc.stop(time + dur)
      }
      
      // Urgent, clean double-beep ADAS sound
      playTone(now, 920, 0.12)
      playTone(now + 0.18, 920, 0.15)
    } catch (err) {
      console.error('Audio synthesis failed:', err)
    }
  }, [])

  // Check detections and trigger alerts when websocket receives new detections
  useEffect(() => {
    if (!isConnected) {
      setIsAlertActive(false)
      return
    }

    const hazardExists = detections.some(d => {
      const isHighSeverity = d.severity === 'High' || d.severity === 'Critical'
      const dist = d.distance_m ?? 5.0
      return isHighSeverity && dist <= proximityThreshold
    })

    setIsAlertActive(hazardExists)

    if (hazardExists && audioAlertEnabled) {
      const nowMs = Date.now()
      if (nowMs - lastBeepTimeRef.current > 1000) {
        playWarningSound()
        lastBeepTimeRef.current = nowMs
      }
    }
  }, [detections, proximityThreshold, audioAlertEnabled, isConnected, playWarningSound])

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [dropdownRef])

  // ── Camera ──────────────────────────────────────────────────────────────
  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    })
    streamRef.current = stream
    if (videoRef.current) {
      videoRef.current.srcObject = stream
      await new Promise((resolve) => {
        videoRef.current.onloadedmetadata = () => videoRef.current.play().then(resolve)
      })
    }
  }

  const stopCamera = useCallback(() => {
    clearInterval(frameIntervalRef.current); frameIntervalRef.current = null
    cancelAnimationFrame(animFrameRef.current); animFrameRef.current = null
    streamRef.current?.getTracks().forEach((t) => t.stop()); streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setIsCameraActive(false)
  }, [])

  useEffect(() => () => { stopCamera(); disconnect() }, [stopCamera, disconnect])

  // ── Render loop ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isConnected) return
    let active = true

    const draw = () => {
      if (!active) return

      const canvas = displayCanvasRef.current
      const video  = videoRef.current

      if (wsConnectedRef.current && canvas && video && video.srcObject) {
        // Only set width/height once or if video size changes to avoid clearing canvas
        if (canvas.width !== video.videoWidth && video.videoWidth > 0) {
          canvas.width = video.videoWidth
        }
        if (canvas.height !== video.videoHeight && video.videoHeight > 0) {
          canvas.height = video.videoHeight
        }

        if (canvas.width > 0 && canvas.height > 0) {
          const ctx = canvas.getContext('2d')
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

          const isBlinkOn = Math.floor(Date.now() / 250) % 2 === 0

          for (const d of detectionsRef.current) {
            if (!d.bbox) continue
            const [bx, by, bw, bh] = d.bbox
            
            const isHighSeverity = d.severity === 'High' || d.severity === 'Critical'
            const dist = d.distance_m ?? 5.0
            const isHazard = isHighSeverity && dist <= proximityThresholdRef.current

            let color = SEVERITY_COLORS[d.severity] || '#10b981'
            if (isHazard && isBlinkOn) {
              color = '#ef4444' // Blink red
            }

            ctx.strokeStyle = color; ctx.lineWidth = isHazard ? 3 : 2
            ctx.strokeRect(bx, by, bw, bh)

            const cs = 8; ctx.fillStyle = color
            ctx.fillRect(bx - 1, by - 1, cs, 2); ctx.fillRect(bx - 1, by - 1, 2, cs)
            ctx.fillRect(bx + bw - cs + 1, by - 1, cs, 2); ctx.fillRect(bx + bw - 1, by - 1, 2, cs)
            ctx.fillRect(bx - 1, by + bh - 1, cs, 2); ctx.fillRect(bx - 1, by + bh - cs + 1, 2, cs)
            ctx.fillRect(bx + bw - cs + 1, by + bh - 1, cs, 2); ctx.fillRect(bx + bw - 1, by + bh - cs + 1, 2, cs)

            // Draw a second pulsing border for warning
            if (isHazard) {
              const pulseOffset = Math.abs(Math.sin(Date.now() / 150)) * 5
              ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)'
              ctx.lineWidth = 1.5
              ctx.strokeRect(bx - pulseOffset, by - pulseOffset, bw + pulseOffset * 2, bh + pulseOffset * 2)
            }

            ctx.fillStyle = color + '12'; ctx.fillRect(bx, by, bw, bh)

            const cls = d.class_name || 'Defect'
            const severityScore = d.severity_score !== undefined ? d.severity_score : (d.confidence ?? 0)
            const severityPercent = Math.round(severityScore * 100)
            const lines = [
              `${isHazard ? '⚠️ CRITICAL RISK | ' : ''}${cls} | ${d.severity} ${severityPercent}%`,
              `D: ${d.depth_cm?.toFixed(1) ?? '—'}cm  W: ${d.width_cm?.toFixed(1) ?? '—'}cm${d.distance_m !== undefined ? `  Dist: ${d.distance_m.toFixed(1)}m` : ''}`,
            ]
            ctx.font = 'bold 12px "Outfit","Inter","Segoe UI",sans-serif'
            ctx.textBaseline = 'top'
            const lh  = 16, pad = 5
            const lblW = Math.max(...lines.map((l) => ctx.measureText(l).width)) + pad * 2 + 4
            const lblH = lh * lines.length + pad * 2
            const lblY = by - lblH - 2 > 0 ? by - lblH - 2 : by + bh + 2

            ctx.fillStyle = isHazard && isBlinkOn ? 'rgba(239, 68, 68, 0.95)' : 'rgba(15, 23, 42, 0.85)'
            ctx.fillRect(bx, lblY, lblW, lblH)
            ctx.fillStyle = isHazard && !isBlinkOn ? 'rgba(15, 23, 42, 0.85)' : color
            ctx.fillRect(bx, lblY, 3, lblH)
            ctx.fillStyle = '#f8fafc'
            lines.forEach((line, i) => ctx.fillText(line, bx + pad + 4, lblY + pad + i * lh))
          }
        }
      }
      animFrameRef.current = requestAnimationFrame(draw)
    }
    
    // Start drawing
    animFrameRef.current = requestAnimationFrame(draw)
    
    return () => {
      active = false
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
    }
  }, [isConnected])

  // ── Send loop ────────────────────────────────────────────────────────────
  const beginSendLoop = () => {
    const cc    = captureCanvasRef.current
    const video = videoRef.current
    if (!cc || !video) return
    const ctx = cc.getContext('2d')
    cc.width  = video.videoWidth  || 640
    cc.height = video.videoHeight || 480

    frameIntervalRef.current = setInterval(() => {
      if (!wsConnectedRef.current || !video.srcObject) return
      
      try {
        ctx.drawImage(video, 0, 0, cc.width, cc.height)
        const dataUrl = cc.toDataURL('image/jpeg', 0.70)
        const base64Data = dataUrl.split(',')[1]
        if (base64Data) {
          send(base64Data)
        }
      } catch (err) {
        console.error("Frame capture error:", err)
      }
    }, 120) // ~8 FPS
  }

  // ── Start/stop when WS connects/disconnects ──────────────────────────────
  useEffect(() => {
    if (isConnected) {
      const t = setTimeout(() => { beginSendLoop() }, 300)
      return () => clearTimeout(t)
    } else {
      // Clear send interval but keep camera open while retrying connection
      clearInterval(frameIntervalRef.current)
      frameIntervalRef.current = null
      
      // Stop camera only if we have hit complete error / stop state or exhausted retries
      if (hasReconnectExhausted || wsStatus === 'error') {
        stopCamera()
      }
    }
  }, [isConnected, wsStatus, hasReconnectExhausted, stopCamera])

  const handleStart = async () => {
    try {
      await startCamera()
      connect()
      setIsCameraActive(true)
    } catch (err) {
      toast.error(`Camera error: ${err.message}`)
    }
  }

  const handleStop = () => {
    const wasConnected = isConnected
    stopCamera()
    disconnect()
    if (wasConnected) {
      console.log('%c🛑 Active WebSocket connection stopped by user', 'color: red; font-weight: bold;')
      toast.success('Camera stopped', { id: 'ws-status' })
    } else {
      console.log('%c🛑 WebSocket reconnect manually stopped by user', 'color: gray; font-weight: bold;')
      toast.success('Reconnect stopped', { id: 'ws-status' })
    }
  }


  const activePresetObj = VEHICLE_PRESETS.find(p => p.id === selectedPreset) || VEHICLE_PRESETS[0]

  return (
    <div className="mx-auto max-w-6xl">
      {/* Header */}
      <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Live Detection</h1>
          <p className="mt-1 text-sm text-slate-400">
            {uiDescription}
            {isConnected && (
              <span className={`ml-2.5 inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium border ${
                depthActive
                  ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                  : 'bg-amber-500/10 border-amber-500/20 text-amber-400'
              }`}>
                <span className={`h-1.5 w-1.5 rounded-full ${depthActive ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'}`} />
                Depth: {depthActive ? 'Active (MiDaS)' : 'Loading…'}
              </span>
            )}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <StatusBadge status={uiStatus} label={uiStatusLabel} />

          {/* Vehicle calibration preset dropdown */}
          <div className="relative" ref={dropdownRef}>
            <button
              type="button"
              disabled={isConnected || isConnecting}
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className={`flex items-center gap-2 rounded-xl border border-white/5 bg-surface-800/40 backdrop-blur-md px-4 py-2.5 text-xs font-semibold text-white transition-all hover:bg-surface-800/60 focus:outline-none focus:ring-1 focus:ring-brand-500/50 ${
                (isConnected || isConnecting) ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer'
              }`}
            >
              <span>{activePresetObj.name}</span>
              <svg
                className={`h-3 w-3 text-slate-400 transition-transform duration-200 ${isDropdownOpen ? 'rotate-180' : ''}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            <AnimatePresence>
              {isDropdownOpen && (
                <motion.div
                  initial={{ opacity: 0, y: 8, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 8, scale: 0.95 }}
                  transition={{ duration: 0.15 }}
                  className="absolute right-0 mt-2 w-56 origin-top-right rounded-2xl border border-white/10 bg-surface-900/95 backdrop-blur-xl p-2 shadow-2xl z-[100]"
                >
                  <div className="px-3 py-1.5 border-b border-white/5 mb-1.5">
                    <p className="text-[9px] font-bold uppercase tracking-wider text-slate-400 font-sans">Select Vehicle Preset</p>
                  </div>
                  <div className="space-y-1">
                    {VEHICLE_PRESETS.map((preset) => {
                      const isSelected = preset.id === selectedPreset
                      
                      if (preset.active) {
                        return (
                          <button
                            key={preset.id}
                            type="button"
                            onClick={() => {
                              setSelectedPreset(preset.id)
                              setIsDropdownOpen(false)
                            }}
                            className={`w-full flex items-center justify-between rounded-xl p-2 text-left transition-colors border ${
                              isSelected 
                                ? 'bg-brand-500/20 text-white border-brand-500/30' 
                                : 'hover:bg-white/5 text-slate-300 border-transparent'
                            }`}
                          >
                            <div>
                              <p className="font-semibold text-xs text-white leading-tight">{preset.name}</p>
                              <p className="text-[9px] text-slate-400 mt-0.5 font-mono">{preset.dimensions}</p>
                            </div>
                            {isSelected && (
                              <span className="h-1.5 w-1.5 rounded-full bg-brand-400 mr-1" />
                            )}
                          </button>
                        )
                      } else {
                        return (
                          <div
                            key={preset.id}
                            className="group relative w-full flex items-center justify-between rounded-xl p-2 text-left border border-transparent opacity-40 cursor-not-allowed bg-white/[0.01]"
                          >
                            <div>
                              <p className="font-semibold text-xs text-slate-300 leading-tight">{preset.name}</p>
                              <p className="text-[9px] text-slate-500 mt-0.5 font-mono">{preset.dimensions}</p>
                            </div>
                            <span className="text-[10px] mr-1">🔒</span>
                            
                            {/* Hover tooltip for future scope */}
                            <div className="absolute hidden group-hover:flex flex-col items-center pointer-events-none z-[110] right-full top-1/2 -translate-y-1/2 mr-3 w-48">
                              <div className="bg-slate-950/95 border border-white/10 text-white text-[11px] rounded-xl p-2.5 shadow-2xl backdrop-blur-md text-center">
                                <p className="font-bold text-[9px] text-amber-400 uppercase tracking-wider mb-0.5">🔒 Future Scope</p>
                                <p className="text-slate-400 leading-normal text-[10px]">This mounting preset is under future development and will be configurable soon.</p>
                              </div>
                            </div>
                          </div>
                        )
                      }
                    })}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {uiState === 'connected' && (
            <button
              onClick={handleStop}
              className="rounded-xl border border-danger-500/40 bg-danger-500/10 px-5 py-2.5 text-sm font-semibold text-danger-400 transition-colors hover:bg-danger-500/20"
            >
              {uiButtonText}
            </button>
          )}
          {(uiState === 'connecting' || uiState === 'reconnecting') && (
            <button
              onClick={handleStop}
              className="rounded-xl border border-warning-500/40 bg-warning-500/10 px-5 py-2.5 text-sm font-semibold text-warning-400 transition-colors hover:bg-warning-500/20"
            >
              {uiButtonText}
            </button>
          )}
          {(uiState === 'idle' || uiState === 'error') && (
            <button
              onClick={handleStart}
              className="rounded-xl bg-brand-500 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-brand-600"
            >
              {uiButtonText}
            </button>
          )}
        </div>
      </div>

      {/* Off-screen capture elements — must stay in DOM for frame decoding */}
      <video ref={videoRef} style={{ position: 'absolute', width: '1px', height: '1px', opacity: 0, pointerEvents: 'none' }} playsInline muted />
      <canvas ref={captureCanvasRef} style={{ position: 'absolute', width: '1px', height: '1px', opacity: 0, pointerEvents: 'none' }} />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Video feed */}
        <div className="lg:col-span-2 space-y-4">
          <div className="relative overflow-hidden rounded-2xl border border-white/5 bg-black aspect-video flex items-center justify-center">
            <AnimatePresence mode="wait">
              {isConnected ? (
                <motion.canvas
                  key="canvas"
                  ref={displayCanvasRef}
                  className="h-full w-full object-contain"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.15 }}
                />
              ) : (
                <motion.div
                  key="placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex h-full flex-col items-center justify-center text-slate-700"
                >
                  <svg className="mb-3 h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8.288 15.038a5.25 5.25 0 017.424 0M5.106 11.856c3.807-3.808 9.98-3.808 13.788 0M1.924 8.674c5.565-5.565 14.587-5.565 20.152 0M12.53 18.22l-.53.53-.53-.53a.75.75 0 011.06 0z" />
                  </svg>
                  <p className="text-sm px-6 text-center leading-relaxed">
                    {uiDescription}
                  </p>
                </motion.div>
              )}
            </AnimatePresence>

            {isConnected && (
              <div className="absolute right-3 top-3 rounded-lg bg-black/60 px-2.5 py-1 font-mono text-xs font-bold text-accent-400 backdrop-blur-sm">
                {fps} FPS
              </div>
            )}
            {isConnecting && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm">
                <div className="flex flex-col items-center gap-3">
                  <div className="h-8 w-8 animate-spin rounded-full border-2 border-brand-500/20 border-t-brand-400" />
                  <p className="text-xs text-slate-400">Connecting…</p>
                </div>
              </div>
            )}

            {/* Top Dashboard Warning Bar (Vehicle Hazard Blinkers) */}
            <AnimatePresence>
              {isAlertActive && (
                <motion.div
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ type: "spring", stiffness: 300, damping: 25 }}
                  className="absolute top-0 left-0 right-0 h-10 bg-[#7f1d1d]/90 backdrop-blur-md border-b border-danger-500/30 px-6 flex items-center justify-between text-xs z-50 select-none font-sans font-extrabold"
                >
                  {/* Left Blinker Arrow */}
                  <div className="flex gap-1.5 text-danger-500 text-sm animate-pulse tracking-tighter">
                    <span>◀</span><span>◀</span><span>◀</span>
                  </div>

                  {/* Warning center text */}
                  <div className="flex items-center gap-2 text-red-200 tracking-widest uppercase text-[10px]">
                    <span className="text-xs">⚠️</span>
                    <span>HAZARD ALERT: POTHOLE CLOSE</span>
                  </div>

                  {/* Right Blinker Arrow */}
                  <div className="flex gap-1.5 text-danger-500 text-sm animate-pulse tracking-tighter">
                    <span>▶</span><span>▶</span><span>▶</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Status',     value: uiStatusLabel },
              { label: 'FPS',        value: isConnected ? fps : '—' },
              { label: 'Detections', value: detections.length },
            ].map(({ label, value }) => (
              <div key={label} className="rounded-xl border border-white/5 bg-surface-800/60 p-3 text-center">
                <p className="text-xs text-slate-500">{label}</p>
                <p className="mt-0.5 font-mono text-lg font-bold text-white">{value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Detections sidebar */}
        <div className="space-y-4">
          {/* Driver Alert Settings Panel */}
          <div className="rounded-2xl border border-white/5 bg-surface-800/40 backdrop-blur-md p-4 space-y-4">
            <div className="flex items-center justify-between border-b border-white/5 pb-2">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 font-sans flex items-center gap-1.5">
                <span>🛡️</span> Driver Collision Alerts
              </h3>
              {isAlertActive && (
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-danger-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-danger-500"></span>
                </span>
              )}
            </div>
            
            <div className="space-y-4">
              {/* Audio alert toggle switch */}
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-xs font-semibold text-white">Audio Warning Beeps</label>
                  <p className="text-[10px] text-slate-400">ADAS alarm when pothole is near</p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={playWarningSound}
                    className="p-1.5 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 text-slate-300 transition-colors"
                    title="Test warning sound"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                    </svg>
                  </button>
                  <button
                    type="button"
                    onClick={() => setAudioAlertEnabled(!audioAlertEnabled)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                      audioAlertEnabled ? 'bg-brand-500' : 'bg-slate-700'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        audioAlertEnabled ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
              </div>

              {/* Proximity Threshold slider */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold text-white">Alert Proximity Threshold</label>
                  <span className="font-mono text-xs font-bold text-brand-400">{proximityThreshold.toFixed(1)}m</span>
                </div>
                <input
                  type="range"
                  min="1.5"
                  max="4.5"
                  step="0.1"
                  value={proximityThreshold}
                  onChange={(e) => setProximityThreshold(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-brand-500"
                />
                <div className="flex justify-between text-[9px] text-slate-500 font-mono">
                  <span>1.5m (Urgent)</span>
                  <span>3.0m (Standard)</span>
                  <span>4.5m (Early)</span>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-white">
                Live Detections
                {detections.length > 0 && (
                  <span className="ml-2 rounded-full bg-brand-500/20 px-2 py-0.5 text-xs text-brand-400">{detections.length}</span>
                )}
              </h2>
            </div>
          <div className="max-h-[520px] space-y-2 overflow-y-auto pr-1">
            <AnimatePresence mode="popLayout">
              {detections.length > 0 ? (
                detections.map((d, i) => (
                  <DetectionCard key={`${d.class_name ?? i}-${i}`} detection={d} index={i} />
                ))
              ) : (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center justify-center rounded-xl border border-white/5 bg-surface-800/20 py-12 text-slate-600"
                >
                  <svg className="mb-2 h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                  </svg>
                  <p className="text-xs">No detections yet</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          </div>
        </div>
      </div>
    </div>
  )
}
