import { motion } from 'framer-motion'

export function StatusBadge({ status = 'disconnected' }) {
  const statusMap = {
    connected: { label: 'Connected', bg: 'bg-accent-500/10', text: 'text-accent-400', pulse: true },
    connecting: { label: 'Connecting…', bg: 'bg-brand-500/10', text: 'text-brand-400', pulse: true },
    disconnected: { label: 'Disconnected', bg: 'bg-slate-500/10', text: 'text-slate-400', pulse: false },
    error: { label: 'Error', bg: 'bg-danger-500/10', text: 'text-danger-400', pulse: false },
  }

  const config = statusMap[status] || statusMap.disconnected

  return (
    <div className={`flex items-center gap-2 rounded-full ${config.bg} px-3 py-1`}>
      {config.pulse && (
        <motion.div
          className="h-2 w-2 rounded-full bg-accent-400"
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      )}
      <span className={`text-xs font-medium ${config.text}`}>{config.label}</span>
    </div>
  )
}

export default StatusBadge
