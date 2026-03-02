import type { Detection, SummaryStats } from '@/types/detection'
import { SEVERITY_BG } from '@/types/detection'
import StatusBadge from './StatusBadge'

interface Props {
  detections: Detection[]
  summary?: SummaryStats
}

export default function DetectionResult({ detections, summary }: Props) {
  if (detections.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center rounded-xl border border-white/5 bg-surface-800 py-10 text-center">
        <div className="mb-2 text-2xl">✅</div>
        <p className="text-sm font-medium text-slate-300">No potholes detected</p>
        <p className="mt-1 text-xs text-slate-500">Road surface appears clear</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4 animate-slide-up">
      {/* Summary row */}
      {summary && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard label="Total" value={summary.total} color="text-slate-100" />
          <StatCard label="Avg Depth" value={`${summary.avg_depth_cm} cm`} color="text-cyan-400" />
          <StatCard label="Avg Width" value={`${summary.avg_width_cm} cm`} color="text-violet-400" />
          <StatCard label="Max Depth" value={`${summary.max_depth_cm} cm`} color="text-red-400" />
        </div>
      )}

      {/* Severity count pills */}
      {summary && (
        <div className="flex flex-wrap gap-2">
          {(Object.entries(summary.severity_counts) as [string, number][])
            .filter(([, v]) => v > 0)
            .map(([sev, count]) => (
              <span
                key={sev}
                className={`rounded-full px-3 py-1 text-xs font-semibold ring-1 ${SEVERITY_BG[sev as keyof typeof SEVERITY_BG] ?? ''}`}
              >
                {sev}: {count}
              </span>
            ))}
        </div>
      )}

      {/* Detection cards */}
      <ul className="flex flex-col gap-2">
        {detections.map((d) => (
          <li
            key={d.id}
            className="flex flex-wrap items-center gap-3 rounded-xl border border-white/5 bg-surface-700 px-4 py-3 text-sm"
          >
            <span className="font-mono text-xs text-slate-500">#{d.id}</span>
            <span className="font-semibold text-slate-100 capitalize">{d.class_name}</span>
            <span className="text-slate-400">{(d.confidence * 100).toFixed(0)}%</span>

            <StatusBadge severity={d.severity} />

            <div className="ml-auto flex flex-wrap gap-3 text-xs text-slate-400 font-mono">
              <span>D: <span className="text-cyan-400">{d.depth_cm} cm</span></span>
              <span>W: <span className="text-violet-400">{d.width_cm} cm</span></span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

function StatCard({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div className="flex flex-col gap-1 rounded-xl border border-white/5 bg-surface-700 px-4 py-3">
      <span className="text-[11px] uppercase tracking-wide text-slate-500">{label}</span>
      <span className={`text-lg font-bold ${color}`}>{value}</span>
    </div>
  )
}
