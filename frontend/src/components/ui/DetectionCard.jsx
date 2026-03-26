import { motion } from 'framer-motion'

export function DetectionCard({ detection, index = 0 }) {
  const confidence = detection.confidence ?? 0
  const confPercent = Math.round(confidence * 100)
  const confColor =
    confPercent >= 80 ? 'from-accent-500 to-accent-600' : confPercent >= 50 ? 'from-yellow-500 to-yellow-600' : 'from-danger-500 to-danger-600'

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className="rounded-xl border border-white/5 bg-surface-800/60 p-3 backdrop-blur-sm"
    >
      <div className="mb-2 flex items-center justify-between">
        <p className="text-sm font-semibold text-white">{detection.class_name || `Detection #${index}`}</p>
        <span className="text-xs font-mono font-bold text-slate-300">{confPercent}%</span>
      </div>

      <div className="mb-2 h-1.5 overflow-hidden rounded-full bg-surface-700">
        <motion.div className={`h-full bg-gradient-to-r ${confColor}`} initial={{ width: 0 }} animate={{ width: `${confPercent}%` }} transition={{ duration: 0.6 }} />
      </div>

      {(detection.bbox || detection.x || detection.y || detection.width || detection.height) && (
        <div className="text-xs text-slate-400">
          <p>
            x: {Math.round(detection.x || detection.bbox?.[0] || 0)} · y: {Math.round(detection.y || detection.bbox?.[1] || 0)}
          </p>
          <p>
            w: {Math.round(detection.width || detection.bbox?.[2] || 0)} · h: {Math.round(detection.height || detection.bbox?.[3] || 0)}
          </p>
        </div>
      )}
    </motion.div>
  )
}

export default DetectionCard
