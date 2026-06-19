import { useEffect, useRef, useCallback } from 'react'
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

export default function LiveDetection() {
  const { connect, disconnect, wsStatus, currentFrame, detections, fps, depthActive, send } = useWebSocket('', false)

  const isConnected = wsStatus === 'connected'
  const isConnecting = wsStatus === 'connecting'

  const videoRef        = useRef(null)
  const displayCanvasRef = useRef(null)
  const captureCanvasRef = useRef(null)
  const streamRef       = useRef(null)
  const frameIntervalRef = useRef(null)
  const animFrameRef    = useRef(null)
  const detectionsRef   = useRef([])
  const wsConnectedRef  = useRef(false)

  useEffect(() => { detectionsRef.current = detections }, [detections])
  useEffect(() => { wsConnectedRef.current = isConnected }, [isConnected])

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

          for (const d of detectionsRef.current) {
            if (!d.bbox) continue
            const [bx, by, bw, bh] = d.bbox
            const color = SEVERITY_COLORS[d.severity] || '#10b981'

            ctx.strokeStyle = color; ctx.lineWidth = 2
            ctx.strokeRect(bx, by, bw, bh)

            const cs = 8; ctx.fillStyle = color
            ctx.fillRect(bx - 1, by - 1, cs, 2); ctx.fillRect(bx - 1, by - 1, 2, cs)
            ctx.fillRect(bx + bw - cs + 1, by - 1, cs, 2); ctx.fillRect(bx + bw - 1, by - 1, 2, cs)
            ctx.fillRect(bx - 1, by + bh - 1, cs, 2); ctx.fillRect(bx - 1, by + bh - cs + 1, 2, cs)
            ctx.fillRect(bx + bw - cs + 1, by + bh - 1, cs, 2); ctx.fillRect(bx + bw - 1, by + bh - cs + 1, 2, cs)

            ctx.fillStyle = color + '12'; ctx.fillRect(bx, by, bw, bh)

            const cls   = d.class_name || 'Defect'
            const conf  = Math.round(d.confidence * 100)
            const lines = [
              `${cls} | ${d.severity} ${conf}%`,
              `D: ${d.depth_cm?.toFixed(1) ?? '—'}cm  W: ${d.width_cm?.toFixed(1) ?? '—'}cm`,
            ]
            ctx.font = 'bold 12px "Outfit","Inter","Segoe UI",sans-serif'
            ctx.textBaseline = 'top'
            const lh  = 16, pad = 5
            const lblW = Math.max(...lines.map((l) => ctx.measureText(l).width)) + pad * 2 + 4
            const lblH = lh * lines.length + pad * 2
            const lblY = by - lblH - 2 > 0 ? by - lblH - 2 : by + bh + 2

            ctx.fillStyle = 'rgba(15,23,42,0.85)'; ctx.fillRect(bx, lblY, lblW, lblH)
            ctx.fillStyle = color;                 ctx.fillRect(bx, lblY, 3, lblH)
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
      stopCamera()
    }
  }, [isConnected, stopCamera])

  const handleStart = async () => {
    try {
      await startCamera()
      connect()
    } catch (err) {
      toast.error(`Camera error: ${err.message}`)
    }
  }

  return (
    <div className="mx-auto max-w-6xl">
      {/* Header */}
      <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Live Detection</h1>
          <p className="mt-1 text-sm text-slate-400">
            Real-time WebSocket AI inference stream
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
          <StatusBadge status={wsStatus} />
          {isConnected || isConnecting ? (
            <button
              onClick={() => { stopCamera(); disconnect() }}
              className="rounded-xl border border-danger-500/40 bg-danger-500/10 px-5 py-2.5 text-sm font-semibold text-danger-400 transition-colors hover:bg-danger-500/20"
            >
              Stop Camera
            </button>
          ) : (
            <button
              onClick={handleStart}
              className="rounded-xl bg-brand-500 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-brand-600"
            >
              Start Camera
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
                  <p className="text-sm">
                    {wsStatus === 'error' ? 'Connection error — retry' : 'Press Start Camera to begin live feed'}
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
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Status',     value: wsStatus.charAt(0).toUpperCase() + wsStatus.slice(1) },
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
  )
}
