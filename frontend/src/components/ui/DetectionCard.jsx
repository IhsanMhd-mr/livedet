import { motion } from 'framer-motion'

export function DetectionCard({ detection, index = 0, mode }) {
  const isImageMode = mode === 'image'
  const confidence = detection.confidence ?? 0
  const confPercent = Math.round(confidence * 100)
  
  const severity = detection.severity || 'Low'
  const severityColors = isImageMode
    ? {
        Low:      'from-accent-500 to-accent-400',
        Medium:   'from-yellow-500 to-yellow-400',
        High:     'from-orange-500 to-orange-400',
        Critical: 'from-danger-500 to-danger-400',
      }
    : {
        Low:      'from-accent-500 to-accent-500',
        Medium:   'from-yellow-500 to-yellow-500',
        High:     'from-orange-500 to-orange-500',
        Critical: 'from-danger-500 to-danger-500',
      }
  const severityColor = severityColors[severity] || severityColors.Low

  const severityBadgeStyles = {
    Low:      'text-accent-400 border-accent-500/20 bg-accent-500/5',
    Medium:   'text-yellow-400 border-yellow-500/20 bg-yellow-500/5',
    High:     'text-orange-400 border-orange-500/20 bg-orange-500/5',
    Critical: 'text-danger-400 border-danger-500/20 bg-danger-500/5',
  }
  const badgeStyle = severityBadgeStyles[severity] || severityBadgeStyles.Low

  // Bar width represents severity score for image mode, confidence for other modes
  const barWidth = isImageMode
    ? Math.round((detection.severity_score ?? confidence) * 100)
    : confPercent

  // Calculate coordinates correctly using nullish coalescing (preventing 0 from falling back to pixel values)
  const xVal = detection.x ?? (detection.bbox ? detection.bbox[0] : 0)
  const yVal = detection.y ?? (detection.bbox ? detection.bbox[1] : 0)
  const wVal = detection.width_cm ?? detection.width ?? (detection.bbox ? detection.bbox[2] : 0)
  const hVal = detection.height_cm ?? detection.height ?? (detection.bbox ? detection.bbox[3] : 0)

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className="rounded-xl border border-white/5 bg-surface-800/60 p-3 backdrop-blur-sm"
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <p className="text-sm font-semibold text-white truncate">
          #{isImageMode ? (detection.id ?? index + 1) : (detection.id || index + 1)} {detection.class_name || 'Detection'}
        </p>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className={`rounded-md border px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider ${badgeStyle}`}>
            {severity}
          </span>
          <span className="text-xs font-mono font-bold text-slate-300">
            {isImageMode ? `${Math.round((detection.severity_score ?? confidence) * 100)}% Sev` : `${confPercent}%`}
          </span>
        </div>
      </div>

      <div className="mb-2 h-1.5 overflow-hidden rounded-full bg-surface-700">
        <motion.div
          className={`h-full bg-gradient-to-r ${severityColor}`}
          initial={{ width: 0 }}
          animate={{ width: `${barWidth}%` }}
          transition={{ duration: 0.6 }}
        />
      </div>
      {(detection.bbox || detection.x !== undefined || detection.y !== undefined) && (
        <div className="text-xs text-slate-400 space-y-0.5">
          <p>
            x: {typeof xVal === 'number' ? xVal.toFixed(1) : xVal} cm · y: {typeof yVal === 'number' ? yVal.toFixed(1) : yVal} cm
          </p>
          <p>
            w: {typeof wVal === 'number' ? wVal.toFixed(1) : wVal} cm · h: {typeof hVal === 'number' ? hVal.toFixed(1) : hVal} cm
          </p>
          {detection.depth_cm !== undefined && (
            <p>
              depth: {typeof detection.depth_cm === 'number' ? detection.depth_cm.toFixed(1) : detection.depth_cm} cm
            </p>
          )}
        </div>
      )}
    </motion.div>
  )
}

export default DetectionCard

