// frontend/src/components/LiveDetect.jsx
// ═══════════════════════════════════════════════════════════════════════
//  Real-time pothole detection via WebSocket
//  Sends webcam frames → receives detection JSON → draws overlays
// ═══════════════════════════════════════════════════════════════════════

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { WS_URL, createLogger } from '../config'

const log = createLogger('LiveDetect')

// ── constants ──────────────────────────────────────────────────────────
const SEND_INTERVAL_MS = 120          // ~8 FPS send rate
const JPEG_QUALITY     = 0.70
const CANVAS_W         = 640
const CANVAS_H         = 480

const SEVERITY_COLORS = {
  Low:      '#00c853',
  Medium:   '#ffab00',
  High:     '#ff6d00',
  Critical: '#ff1744',
}

const SEVERITY_BG = {
  Low:      'rgba(0,200,83,0.15)',
  Medium:   'rgba(255,171,0,0.15)',
  High:     'rgba(255,109,0,0.15)',
  Critical: 'rgba(255,23,68,0.15)',
}

// ═══════════════════════════════════════════════════════════════════════
//  Component
// ═══════════════════════════════════════════════════════════════════════

export default function LiveDetect() {
  // refs
  const videoRef      = useRef(null)
  const displayCanvas = useRef(null)   // visible – video + overlays
  const captureCanvas = useRef(null)   // hidden  – frame grab for WS
  const wsRef         = useRef(null)
  const animRef       = useRef(null)
  const sendTimerRef  = useRef(null)
  const pendingRef    = useRef(false)
  const detsRef       = useRef([])     // latest detections (non-reactive)
  const statsRef      = useRef({ fps: 0, depthActive: false })

  // state
  const [connected, setConnected]     = useState(false)
  const [running, setRunning]         = useState(false)
  const [camError, setCamError]       = useState(null)
  const [detections, setDetections]   = useState([])
  const [stats, setStats]             = useState({
    fps: 0, frameCount: 0, numPotholes: 0, depthActive: false,
  })

  // ── WebSocket helpers ────────────────────────────────────────────────

  const connectWS = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState <= 1) return

    log.log('WS', `Connecting to ${WS_URL} …`)
    const ws = new WebSocket(WS_URL)

    ws.onopen = () => {
      log.log('WS', 'Connected ✓')
      setConnected(true)
    }

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data)
        if (data.error) { log.warn('WS', data.error); pendingRef.current = false; return }

        detsRef.current = data.detections || []
        statsRef.current = { fps: data.fps || 0, depthActive: data.depth_active || false }

        setDetections(data.detections || [])
        setStats({
          fps:         data.fps          || 0,
          frameCount:  data.frame_count  || 0,
          numPotholes: data.num_potholes || 0,
          depthActive: data.depth_active || false,
        })
      } catch (e) {
        log.error('WS', 'parse error', e)
      }
      pendingRef.current = false
    }

    ws.onerror = () => setConnected(false)
    ws.onclose = () => { log.log('WS', 'Disconnected'); setConnected(false) }

    wsRef.current = ws
  }, [])

  const disconnectWS = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  // auto-connect on mount
  useEffect(() => { connectWS(); return disconnectWS }, [connectWS, disconnectWS])

  // ── Camera ───────────────────────────────────────────────────────────

  const startCamera = async () => {
    setCamError(null)

    // ensure WS is connected
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connectWS()
      await new Promise(r => setTimeout(r, 800))
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        setCamError('WebSocket server is not running. Start it with:  python backend/live_ws.py')
        return
      }
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: CANVAS_W }, height: { ideal: CANVAS_H } },
        audio: false,
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        setRunning(true)
        beginRenderLoop()
        beginSendLoop()
      }
    } catch (err) {
      setCamError(`Camera access denied: ${err.message}`)
    }
  }

  const stopCamera = useCallback(() => {
    if (sendTimerRef.current)  { clearInterval(sendTimerRef.current); sendTimerRef.current = null }
    if (animRef.current)       { cancelAnimationFrame(animRef.current); animRef.current = null }

    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      videoRef.current.srcObject = null
    }
    detsRef.current = []
    setDetections([])
    setRunning(false)
  }, [])

  // cleanup on unmount
  useEffect(() => () => { stopCamera(); disconnectWS() }, [stopCamera, disconnectWS])

  // ── Render loop (requestAnimationFrame) ──────────────────────────────

  const beginRenderLoop = () => {
    const canvas = displayCanvas.current
    const video  = videoRef.current
    if (!canvas || !video) return
    const ctx = canvas.getContext('2d')

    const draw = () => {
      if (!videoRef.current?.srcObject) return

      // video background
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      // detection overlays
      const dets  = detsRef.current
      const sInfo = statsRef.current
      ctx.textBaseline = 'top'

      for (const d of dets) {
        const [bx, by, bw, bh] = d.bbox
        const confPercent = Math.round((d.confidence ?? 0) * 100)
        const color = confPercent > 85
          ? '#ef4444' // Red
          : confPercent > 70
          ? '#eab308' // Yellow
          : '#10b981' // Green/Accent

        // bounding box
        ctx.strokeStyle = color
        ctx.lineWidth   = 2
        ctx.strokeRect(bx, by, bw, bh)

        // corner markers
        const cs = 8
        ctx.fillStyle = color
        ctx.fillRect(bx - 1, by - 1, cs, 2); ctx.fillRect(bx - 1, by - 1, 2, cs)
        ctx.fillRect(bx + bw - cs + 1, by - 1, cs, 2); ctx.fillRect(bx + bw - 1, by - 1, 2, cs)
        ctx.fillRect(bx - 1, by + bh - 1, cs, 2); ctx.fillRect(bx - 1, by + bh - cs + 1, 2, cs)
        ctx.fillRect(bx + bw - cs + 1, by + bh - 1, cs, 2); ctx.fillRect(bx + bw - 1, by + bh - cs + 1, 2, cs)

        // label
        const cls = d.class_name || 'Defect'
        const lines = [
          `${cls}  ${d.severity}  ${(d.confidence * 100).toFixed(0)}%`,
          `D: ${d.depth_cm} cm   W: ${d.width_cm} cm`,
        ]
        ctx.font = 'bold 12px "Segoe UI", monospace'
        const lineH = 16, pad = 5
        const maxTw = Math.max(...lines.map(l => ctx.measureText(l).width))
        const lblW  = maxTw + pad * 2
        const lblH  = lineH * lines.length + pad * 2
        const lblY  = by - lblH - 2 > 0 ? by - lblH - 2 : by + bh + 2

        ctx.fillStyle = 'rgba(0,0,0,0.70)'
        ctx.fillRect(bx, lblY, lblW, lblH)
        ctx.fillStyle = color
        ctx.fillRect(bx, lblY, 3, lblH)  // accent bar
        ctx.fillStyle = '#fff'
        lines.forEach((line, i) => ctx.fillText(line, bx + pad + 3, lblY + pad + i * lineH))
      }

      // HUD bar
      const hudH = 30
      ctx.fillStyle = 'rgba(0,0,0,0.55)'
      ctx.fillRect(0, 0, canvas.width, hudH)
      ctx.font = 'bold 13px "Segoe UI", sans-serif'
      ctx.fillStyle = '#ffeb3b'
      ctx.fillText(`FPS: ${sInfo.fps}`, 10, 20)
      ctx.fillStyle = '#fff'
      ctx.fillText(`Detections: ${dets.length}`, canvas.width / 2 - 50, 20)
      ctx.fillStyle = sInfo.depthActive ? '#69f0ae' : '#ffab40'
      ctx.fillText(sInfo.depthActive ? 'Depth: ON' : 'Depth: loading…', canvas.width - 130, 20)

      animRef.current = requestAnimationFrame(draw)
    }

    draw()
  }

  // ── Send loop ────────────────────────────────────────────────────────

  const beginSendLoop = () => {
    const cap   = captureCanvas.current
    const video = videoRef.current
    if (!cap || !video) return
    const ctx = cap.getContext('2d')

    sendTimerRef.current = setInterval(() => {
      if (pendingRef.current) return
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
      if (!videoRef.current?.srcObject) return

      ctx.drawImage(video, 0, 0, cap.width, cap.height)
      cap.toBlob(
        (blob) => {
          if (!blob) return
          const reader = new FileReader()
          reader.onloadend = () => {
            const b64 = reader.result.split(',')[1]
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              pendingRef.current = true
              wsRef.current.send(b64)
            }
          }
          reader.readAsDataURL(blob)
        },
        'image/jpeg',
        JPEG_QUALITY,
      )
    }, SEND_INTERVAL_MS)
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  Render
  // ═══════════════════════════════════════════════════════════════════════

  return (
    <div className="live-detect-container" style={S.root}>
      {/* ── LEFT: Video feed ─────────────────────────────────────────── */}
      <div style={S.videoCol}>
        <div style={S.canvasWrap}>
          <video ref={videoRef} style={{ display: 'none' }} width={CANVAS_W} height={CANVAS_H} playsInline muted />
          <canvas ref={displayCanvas} width={CANVAS_W} height={CANVAS_H} style={S.canvas} />
          <canvas ref={captureCanvas} width={CANVAS_W} height={CANVAS_H} style={{ display: 'none' }} />

          {!running && (
            <div style={S.placeholder}>
              <span style={{ fontSize: 48 }}>📷</span>
              <p style={{ margin: '12px 0 0', color: '#aaa' }}>
                Click <b>Start Camera</b> to begin live detection
              </p>
            </div>
          )}
        </div>

        {/* Controls */}
        <div style={S.controls}>
          <button
            onClick={startCamera}
            disabled={running}
            className="btn btn-primary"
            style={{ ...S.btn, opacity: running ? 0.5 : 1 }}
          >
            {running ? '🎥 Camera ON' : '▶️ Start Camera'}
          </button>

          <button
            onClick={stopCamera}
            disabled={!running}
            className="btn btn-danger"
            style={{ ...S.btn, background: '#dc3545', opacity: !running ? 0.5 : 1 }}
          >
            ⏹️ Stop
          </button>

          <button
            onClick={() => { disconnectWS(); setTimeout(connectWS, 300) }}
            className="btn btn-secondary"
            style={{ ...S.btn, background: '#6c757d' }}
          >
            🔄 Reconnect
          </button>

          <div style={S.statusPill(connected)}>
            <span style={S.statusDot(connected)} />
            {connected ? 'Connected' : 'Disconnected'}
          </div>
        </div>

        {camError && <div style={S.error}>⚠️ {camError}</div>}
      </div>

      {/* ── RIGHT: Stats panel ───────────────────────────────────────── */}
      <div style={S.statsCol}>
        <h3 style={S.statsTitle}>📊 Live Statistics</h3>

        {/* Metric cards */}
        <div style={S.cardRow}>
          <StatCard label="Server FPS" value={stats.fps} accent="#ffeb3b" />
          <StatCard label="Frames" value={stats.frameCount} accent="#64b5f6" />
          <StatCard label="Detections" value={stats.numPotholes} accent="#ef5350" />
        </div>

        {/* Depth status */}
        <div style={{
          ...S.depthBadge,
          background: stats.depthActive ? 'rgba(105,240,174,0.15)' : 'rgba(255,171,64,0.15)',
          color:      stats.depthActive ? '#69f0ae' : '#ffab40',
        }}>
          {stats.depthActive ? '✅ MiDaS Depth Active' : '⏳ Depth Model Loading…'}
        </div>

        {/* Per-pothole table */}
        {detections.length > 0 ? (
          <div style={S.tableWrap}>
            <h4 style={{ margin: '0 0 8px', color: '#e0e0e0' }}>�️ Detected Objects</h4>
            <table style={S.table}>
              <thead>
                <tr>
                  {['#', 'Class', 'Severity', 'Depth', 'Width', 'Conf'].map(h => (
                    <th key={h} style={S.th}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {detections.map(d => (
                  <tr key={d.id} style={{ background: SEVERITY_BG[d.severity] || 'transparent' }}>
                    <td style={S.td}>{d.id}</td>
                    <td style={{ ...S.td, color: '#81d4fa', fontWeight: 600 }}>{d.class_name || '—'}</td>
                    <td style={{ ...S.td, color: SEVERITY_COLORS[d.severity], fontWeight: 700 }}>
                      {d.severity}
                    </td>
                    <td style={S.td}>{d.depth_cm} cm</td>
                    <td style={S.td}>{d.width_cm} cm</td>
                    <td style={S.td}>{(d.confidence * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Summary row */}
            {detections.length > 1 && (
              <div style={S.summaryRow}>
                <span>Avg depth: <b>{avg(detections, 'depth_cm')} cm</b></span>
                <span style={{ marginLeft: 16 }}>Avg width: <b>{avg(detections, 'width_cm')} cm</b></span>
              </div>
            )}
          </div>
        ) : (
          <div style={S.noDet}>
            {running
              ? 'Scanning for road defects …'
              : 'Start the camera to begin detection'}
          </div>
        )}

        {/* Tips */}
        <div style={S.tipsBox}>
          <h4 style={{ margin: '0 0 6px' }}>💡 Tips</h4>
          <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.7, color: '#bbb', fontSize: 13 }}>
            <li>Angle camera ~45° over pavement</li>
            <li>Good lighting improves accuracy</li>
            <li>Depth runs every 3 frames for performance</li>
            <li>Width uses: <code style={{ color: '#81d4fa' }}>RealW = Px×Depth / Focal</code></li>
          </ul>
        </div>
      </div>
    </div>
  )
}

// ── Small helpers ──────────────────────────────────────────────────────

function StatCard({ label, value, accent }) {
  return (
    <div style={S.card}>
      <div style={{ fontSize: 11, color: '#999', textTransform: 'uppercase', letterSpacing: 1 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 700, color: accent, marginTop: 2 }}>{value}</div>
    </div>
  )
}

function avg(arr, key) {
  if (!arr.length) return '—'
  return (arr.reduce((s, d) => s + d[key], 0) / arr.length).toFixed(1)
}

// ═══════════════════════════════════════════════════════════════════════
//  Inline Styles
// ═══════════════════════════════════════════════════════════════════════

const S = {
  root: {
    display: 'flex', gap: 20, flexWrap: 'wrap',
    padding: 16, maxWidth: 1100, margin: '0 auto',
  },

  videoCol: { flex: '1 1 640px', minWidth: 320 },

  canvasWrap: {
    position: 'relative', width: '100%', aspectRatio: '4/3',
    background: '#111', borderRadius: 10, overflow: 'hidden',
    border: '1px solid #333',
  },
  canvas: { width: '100%', height: '100%', display: 'block', borderRadius: 10 },
  placeholder: {
    position: 'absolute', inset: 0,
    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
  },

  controls: {
    display: 'flex', gap: 10, alignItems: 'center',
    marginTop: 12, flexWrap: 'wrap',
  },
  btn: {
    padding: '8px 18px', border: 'none', borderRadius: 6, color: '#fff',
    fontWeight: 600, cursor: 'pointer', fontSize: 14,
  },

  statusPill: (on) => ({
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '5px 12px', borderRadius: 20, fontSize: 13, fontWeight: 600,
    background: on ? 'rgba(105,240,174,0.12)' : 'rgba(255,82,82,0.12)',
    color:      on ? '#69f0ae' : '#ff5252',
  }),
  statusDot: (on) => ({
    width: 8, height: 8, borderRadius: '50%', display: 'inline-block',
    background: on ? '#69f0ae' : '#ff5252',
    boxShadow:  on ? '0 0 6px #69f0ae' : 'none',
  }),

  error: {
    marginTop: 10, padding: '10px 14px', borderRadius: 8,
    background: 'rgba(255,82,82,0.1)', color: '#ff5252',
    fontSize: 14, fontWeight: 500,
  },

  statsCol: {
    flex: '1 1 300px', minWidth: 280,
    background: '#1a1a2e', borderRadius: 12, padding: 20,
    border: '1px solid #333', maxHeight: 700, overflowY: 'auto',
  },
  statsTitle: { margin: '0 0 14px', color: '#e0e0e0' },

  cardRow: { display: 'flex', gap: 10, marginBottom: 14 },
  card: {
    flex: 1, background: '#16213e', borderRadius: 8, padding: '10px 12px',
    textAlign: 'center', border: '1px solid #1a1a40',
  },

  depthBadge: {
    padding: '8px 14px', borderRadius: 8, textAlign: 'center',
    fontWeight: 600, fontSize: 13, marginBottom: 14,
  },

  tableWrap: { marginBottom: 14 },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
  th: {
    padding: '6px 8px', borderBottom: '1px solid #333',
    color: '#999', fontWeight: 600, textAlign: 'left', fontSize: 11,
    textTransform: 'uppercase', letterSpacing: 0.5,
  },
  td: { padding: '6px 8px', borderBottom: '1px solid #222', color: '#ddd' },
  summaryRow: {
    marginTop: 8, padding: '6px 8px', fontSize: 13,
    color: '#aaa', borderTop: '1px solid #333',
  },

  noDet: {
    padding: 20, textAlign: 'center', color: '#666', fontSize: 14,
    background: '#16213e', borderRadius: 8, marginBottom: 14,
  },

  tipsBox: {
    padding: 14, background: '#16213e', borderRadius: 8,
    border: '1px solid #1a1a40', color: '#ccc', fontSize: 13,
  },
}
