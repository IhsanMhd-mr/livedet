import { Outlet } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { useLocation } from 'react-router-dom'
import Navbar from './Navbar'

export default function Layout() {
  const location = useLocation()

  return (
    <div className="flex min-h-screen flex-col bg-surface-900">
      <Navbar />
      <main className="flex-1">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="px-6 py-8"
          >
            <div className="mx-auto max-w-7xl">
              <Outlet />
            </div>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
