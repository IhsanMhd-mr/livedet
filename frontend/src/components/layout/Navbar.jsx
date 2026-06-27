import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'

export default function Navbar() {
  const location = useLocation()

  const links = [
    { label: 'Home', to: '/' },
    { label: 'Image', to: '/image' },
    { label: 'Video', to: '/video' },
    { label: 'Live', to: '/live' },
  ]

  return (
    <nav className="sticky top-0 z-50 border-b border-white/5 bg-surface-900/95 backdrop-blur-lg">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <Link to="/" className="flex items-center gap-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-brand-500">
            <span className="font-bold text-white">LD</span>
          </div>
          <span className="hidden font-bold text-white sm:inline">LIVEDET</span>
        </Link>

        <div className="flex gap-1 items-center">
          {links.map(({ label, to }) => (
            <Link
              key={to}
              to={to}
              className="relative px-3 py-2 text-sm font-medium text-slate-300 transition-colors hover:text-white"
            >
              {location.pathname === to && (
                <motion.div
                  layoutId="nav-indicator"
                  className="absolute inset-x-0 bottom-0 h-0.5 bg-brand-500"
                  transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                />
              )}
              {label}
            </Link>
          ))}

          <Link
            to="/demo"
            title="Demos"
            className="relative ml-1 p-2 rounded-lg text-slate-300 transition-colors hover:text-white hover:bg-white/5"
          >
            {location.pathname === '/demo' && (
              <motion.div
                layoutId="nav-indicator"
                className="absolute inset-x-0 bottom-0 h-0.5 bg-brand-500"
                transition={{ type: 'spring', stiffness: 380, damping: 30 }}
              />
            )}
            <svg
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </Link>
        </div>
      </div>
    </nav>
  )
}
