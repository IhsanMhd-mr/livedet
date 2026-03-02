import { Link, useLocation } from 'react-router-dom'
import { Activity, Image as ImageIcon, Video, Radio } from 'lucide-react'

const NAV_ITEMS = [
  { to: '/image', label: 'Image', icon: ImageIcon },
  { to: '/video', label: 'Video', icon: Video },
  { to: '/live', label: 'Live', icon: Radio },
]

export default function Navbar() {
  const { pathname } = useLocation()

  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/5 bg-surface-900/80 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5 group">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent shadow-glow">
            <Activity className="h-4 w-4 text-white" />
          </span>
          <span className="text-sm font-bold tracking-widest text-white uppercase">
            Live<span className="text-accent-light">Det</span>
          </span>
        </Link>

        {/* Nav Links */}
        <nav className="flex items-center gap-1">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => {
            const active = pathname === to
            return (
              <Link
                key={to}
                to={to}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all duration-150 ${
                  active
                    ? 'bg-accent/20 text-accent-light ring-1 ring-accent/40'
                    : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </Link>
            )
          })}
        </nav>
      </div>
    </header>
  )
}
