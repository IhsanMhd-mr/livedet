import { type LucideIcon } from 'lucide-react'
import { Link } from 'react-router-dom'

interface Props {
  to: string
  icon: LucideIcon
  title: string
  description: string
  badge?: string
  accentColor?: string
}

export default function CardOption({
  to,
  icon: Icon,
  title,
  description,
  badge,
  accentColor = 'from-indigo-500 to-violet-600',
}: Props) {
  return (
    <Link
      to={to}
      className="group relative flex flex-col gap-5 rounded-2xl border border-white/5 bg-surface-800 p-6 shadow-card transition-all duration-300 hover:-translate-y-1 hover:border-white/10 hover:shadow-glow focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
    >
      {/* Icon */}
      <div
        className={`flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br ${accentColor} shadow-lg transition-transform duration-300 group-hover:scale-110`}
      >
        <Icon className="h-6 w-6 text-white" />
      </div>

      {/* Text */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center gap-2">
          <h3 className="text-base font-semibold text-slate-100">{title}</h3>
          {badge && (
            <span className="rounded-full bg-accent/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-accent-light ring-1 ring-accent/30">
              {badge}
            </span>
          )}
        </div>
        <p className="text-sm text-slate-400 leading-relaxed">{description}</p>
      </div>

      {/* Arrow */}
      <div className="mt-auto flex items-center gap-1 text-xs font-medium text-slate-500 transition-colors group-hover:text-accent-light">
        Open
        <svg
          className="h-3.5 w-3.5 transition-transform duration-150 group-hover:translate-x-0.5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
      </div>
    </Link>
  )
}
