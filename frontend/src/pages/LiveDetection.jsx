import { motion, AnimatePresence } from 'framer-motion'
import useWebSocket from '@/hooks/useWebSocket'
import StatusBadge from '@/components/ui/StatusBadge'
import DetectionCard from '@/components/ui/DetectionCard'

export default function LiveDetection() {
  const { connect, disconnect, wsStatus, currentFrame, detections, fps } = useWebSocket('/ws/live', false)

  const isConnected = wsStatus === 'connected'
  const isConnecting = wsStatus === 'connecting'

  return (
    <div className="mx-auto max-w-6xl">
      {/* Header */}
      <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Live Detection</h1>
          <p className="mt-1 text-sm text-slate-400">Real-time WebSocket AI inference stream</p>
        </div>

        <div className="flex items-center gap-3">
          <StatusBadge status={wsStatus} />

          {isConnected || isConnecting ? (
            <button
              onClick={disconnect}
              className="rounded-xl border border-danger-500/40 bg-danger-500/10 px-5 py-2.5 text-sm font-semibold text-danger-400 transition-colors hover:bg-danger-500/20"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={connect}
              className="rounded-xl bg-brand-500 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-brand-600"
            >
              Start
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Video feed — takes 2 cols */}
        <div className="lg:col-span-2 space-y-4">
          {/* Frame canvas */}
          <div className="relative overflow-hidden rounded-2xl border border-white/5 bg-black aspect-video">
            <AnimatePresence>
              {currentFrame ? (
                <motion.img
                  key="frame"
                  src={currentFrame}
                  alt="Live frame"
                  className="h-full w-full object-contain"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.05 }}
                />
              ) : (
                <motion.div
                  key="placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex h-full flex-col items-center justify-center text-slate-700"
                >
                  <svg className="mb-3 h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8.288 15.038a5.25 5.25 0 017.424 0M5.106 11.856c3.807-3.808 9.98-3.808 13.788 0M1.924 8.674c5.565-5.565 14.587-5.565 20.152 0M12.53 18.22l-.53.53-.53-.53a.75.75 0 011.06 0z" />
                  </svg>
                  <p className="text-sm">
                    {wsStatus === 'error' ? 'Connection error — retry' : 'Press Start to connect'}
                  </p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* FPS overlay */}
            {isConnected && (
              <div className="absolute right-3 top-3 rounded-lg bg-black/60 px-2.5 py-1 font-mono text-xs font-bold text-accent-400 backdrop-blur-sm">
                {fps} FPS
              </div>
            )}

            {/* Connecting overlay */}
            {isConnecting && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm">
                <div className="flex flex-col items-center gap-3">
                  <div className="h-8 w-8 animate-spin rounded-full border-2 border-brand-500/20 border-t-brand-400" />
                  <p className="text-xs text-slate-400">Connecting…</p>
                </div>
              </div>
            )}
          </div>

          {/* Stats bar */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Status', value: wsStatus.charAt(0).toUpperCase() + wsStatus.slice(1) },
              { label: 'FPS', value: isConnected ? fps : '—' },
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
