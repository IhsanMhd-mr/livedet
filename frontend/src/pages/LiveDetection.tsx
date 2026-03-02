import { useEffect, useRef, useState, useCallback } from 'react'
import { Play, Square, RefreshCw, Camera, CameraOff } from 'lucide-react'
import { useWebSocket } from '@/hooks/useWebSocket'
import { ConnStatusBadge } from '@/components/StatusBadge'
import type { WSFrame, WSDetection } from '@/types/detection'
import { SEVERITY_COLORS } from '@/types/detection'

const FRAME_INTERVAL_MS = 100 // ~10 fps send rate
const CANVAS_QUALITY = 0.6

export default function LiveDetection() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const [streaming, setStreaming] = useState(false)
  const [camError, setCamError] = useState<string | null>(null)
  const [frame, setFrame] = useState<WSFrame | null>(null)
  const [displayFps, setDisplayFps] = useState(0)

  const onFrame = useCallback((f: WSFrame) => {
    setFrame(f)
    setDisplayFps(f.fps)
    drawOverlay(f.detections)
  }, [])

  const { status, error: wsError, connect, disconnect, sendFrame } = useWebSocket({
    url: import.meta.env.VITE_WS_URL ?? 'ws://localhost:8765',
    onFrame,
  })

  // ── Camera ────────────────────────────────────────────────────────────────

  const startCamera = useCallback(async () => {
    setCamError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
    } catch (e) {
      setCamError(e instanceof Error ? e.message : 'Camera access denied')
    }
  }, [])

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }, [])

  // ── Frame sender loop ─────────────────────────────────────────────────────

  const startStream = useCallback(async () => {
    await startCamera()
    connect()
    setStreaming(true)

    intervalRef.current = setInterval(() => {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas || video.readyState < 2) return

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      canvas.width = video.videoWidth || 640
      canvas.height = video.videoHeight || 480
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      const base64 = canvas.toDataURL('image/jpeg', CANVAS_QUALITY)
      sendFrame(base64)
    }, FRAME_INTERVAL_MS)
  }, [startCamera, connect, sendFrame])

  const stopStream = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    disconnect()
    stopCamera()
    setStreaming(false)
    setFrame(null)
    setDisplayFps(0)
    clearOverlay()
  }, [disconnect, stopCamera])

  // Cleanup on unmount
  useEffect(() => () => { stopStream() }, [stopStream])

  // ── Overlay drawing ───────────────────────────────────────────────────────

  function drawOverlay(detections: WSDetection[]) {
    const video = videoRef.current
    const overlay = overlayRef.current
    if (!video || !overlay) return

    overlay.width = video.clientWidth
    overlay.height = video.clientHeight

    const scaleX = video.clientWidth / (video.videoWidth || 640)
    const scaleY = video.clientHeight / (video.videoHeight || 480)

    const ctx = overlay.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, overlay.width, overlay.height)

    detections.forEach((d) => {
      const [x, y, w, h] = d.bbox
      const sx = x * scaleX
      const sy = y * scaleY
      const sw = w * scaleX
      const sh = h * scaleY

      const color = SEVERITY_COLORS[d.severity] ?? '#6366f1'

      // Box
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(sx, sy, sw, sh)

      // Label background
      const label = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%  ${d.severity}`
      ctx.font = 'bold 12px Inter, sans-serif'
      const tw = ctx.measureText(label).width
      ctx.fillStyle = color + 'cc'
      ctx.fillRect(sx, sy - 20, tw + 10, 20)

      // Label text
      ctx.fillStyle = '#fff'
      ctx.fillText(label, sx + 5, sy - 5)

      // Sub-label: depth + width
      const sub = `D:${d.depth_cm}cm  W:${d.width_cm}cm`
      ctx.font = '10px JetBrains Mono, monospace'
      ctx.fillStyle = color
      ctx.fillText(sub, sx + 3, sy + sh + 13)
    })
  }

  function clearOverlay() {
    const overlay = overlayRef.current
    if (!overlay) return
    const ctx = overlay.getContext('2d')
    ctx?.clearRect(0, 0, overlay.width, overlay.height)
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <main className="mx-auto max-w-6xl px-4 py-10">
      {/* Header */}
      <div className="mb-6 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-100">Live Detection</h2>
          <p className="mt-1 text-sm text-slate-400">
            Real-time pothole detection via WebSocket + webcam using YOLOv8
          </p>
        </div>
        <div className="flex items-center gap-3">
          <ConnStatusBadge status={status} />
          {frame?.depth_active && (
            <span className="rounded-full bg-cyan-500/20 px-2.5 py-1 text-[11px] font-semibold text-cyan-400 ring-1 ring-cyan-500/30">
              MiDaS ✓
            </span>
          )}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Video feed */}
        <div className="relative lg:col-span-2 overflow-hidden rounded-2xl border border-white/5 bg-black">
          <video
            ref={videoRef}
            muted
            playsInline
            className="w-full"
            style={{ display: 'block', minHeight: 300 }}
          />
          {/* Overlay canvas, absolutely positioned on top of video */}
          <canvas
            ref={overlayRef}
            className="pointer-events-none absolute inset-0 w-full h-full"
            style={{ position: 'absolute', top: 0, left: 0 }}
          />
          {/* Hidden capture canvas */}
          <canvas ref={canvasRef} className="hidden" />

          {/* Placeholder when not streaming */}
          {!streaming && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-surface-900/80">
              <CameraOff className="h-12 w-12 text-slate-600" />
              <p className="text-sm text-slate-500">Camera inactive</p>
            </div>
          )}

          {/* FPS badge */}
          {streaming && (
            <div className="absolute bottom-3 left-3 rounded-lg bg-black/60 px-2.5 py-1 font-mono text-xs text-emerald-400 backdrop-blur-sm">
              {displayFps.toFixed(1)} fps
            </div>
          )}

          {/* Frame counter */}
          {frame && (
            <div className="absolute bottom-3 right-3 rounded-lg bg-black/60 px-2.5 py-1 font-mono text-xs text-slate-400 backdrop-blur-sm">
              #{frame.frame_count}
            </div>
          )}
        </div>

        {/* Side panel */}
        <div className="flex flex-col gap-4">
          {/* Controls */}
          <div className="rounded-2xl border border-white/5 bg-surface-800 p-4 flex flex-col gap-3">
            <p className="text-xs font-semibold uppercase tracking-widest text-slate-500">Controls</p>

            {!streaming ? (
              <button
                type="button"
                onClick={startStream}
                className="flex h-10 items-center justify-center gap-2 rounded-xl bg-emerald-600 text-sm font-semibold text-white shadow transition-all hover:bg-emerald-500"
              >
                <Play className="h-4 w-4 fill-white" />
                Start Streaming
              </button>
            ) : (
              <button
                type="button"
                onClick={stopStream}
                className="flex h-10 items-center justify-center gap-2 rounded-xl bg-red-600 text-sm font-semibold text-white transition-all hover:bg-red-500"
              >
                <Square className="h-4 w-4 fill-white" />
                Stop
              </button>
            )}

            {streaming && status === 'error' && (
              <button
                type="button"
                onClick={connect}
                className="flex h-9 items-center justify-center gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 text-xs font-semibold text-amber-400 hover:bg-amber-500/20"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Reconnect WS
              </button>
            )}

            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Camera className="h-3.5 w-3.5" />
              <span>ws://localhost:8765</span>
            </div>
          </div>

          {/* Errors */}
          {(camError ?? wsError) && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2.5 text-xs text-red-400">
              {camError ?? wsError}
            </div>
          )}

          {/* Live detections list */}
          <div className="flex-1 rounded-2xl border border-white/5 bg-surface-800 p-4">
            <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-slate-500">
              Detections{frame ? ` · ${frame.num_detections}` : ''}
            </p>

            {frame && frame.detections.length > 0 ? (
              <ul className="flex flex-col gap-2">
                {frame.detections.map((d) => (
                  <li
                    key={d.id}
                    className="rounded-lg border border-white/5 bg-surface-700 px-3 py-2 text-xs"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-slate-200 capitalize">{d.class_name}</span>
                      <span
                        className="rounded-full px-2 py-0.5 text-[10px] font-bold ring-1"
                        style={{ color: SEVERITY_COLORS[d.severity], backgroundColor: SEVERITY_COLORS[d.severity] + '22', borderColor: SEVERITY_COLORS[d.severity] + '55' }}
                      >
                        {d.severity}
                      </span>
                    </div>
                    <div className="mt-1.5 flex gap-3 font-mono text-slate-400">
                      <span>{(d.confidence * 100).toFixed(0)}%</span>
                      <span>D:<span className="text-cyan-400">{d.depth_cm}cm</span></span>
                      <span>W:<span className="text-violet-400">{d.width_cm}cm</span></span>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-slate-600">
                {streaming ? 'No defects in current frame' : 'Start streaming to see results'}
              </p>
            )}
          </div>
        </div>
      </div>
    </main>
  )
}
