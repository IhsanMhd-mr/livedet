import type { SeverityLevel } from '@/types/detection'
import { SEVERITY_BG } from '@/types/detection'
import type { WSStatus } from '@/api/websocket'

interface SeverityProps {
  severity: SeverityLevel
  className?: string
}

export default function StatusBadge({ severity, className = '' }: SeverityProps) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-semibold ring-1 ${SEVERITY_BG[severity]} ${className}`}
    >
      {severity}
    </span>
  )
}

const WS_STYLES: Record<WSStatus, string> = {
  disconnected: 'bg-slate-700/60 text-slate-400 ring-slate-600/30',
  connecting: 'bg-amber-500/20 text-amber-400 ring-amber-500/30',
  connected: 'bg-green-500/20 text-green-400 ring-green-500/30',
  error: 'bg-red-500/20 text-red-400 ring-red-500/30',
}

const WS_DOT: Record<WSStatus, string> = {
  disconnected: 'bg-slate-500',
  connecting: 'bg-amber-400 animate-pulse',
  connected: 'bg-green-400 animate-pulse-slow',
  error: 'bg-red-400',
}

interface ConnStatusProps {
  status: WSStatus
}

export function ConnStatusBadge({ status }: ConnStatusProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ring-1 ${WS_STYLES[status]}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${WS_DOT[status]}`} />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  )
}
