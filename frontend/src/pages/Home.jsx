import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import CardOption from '@/components/ui/CardOption'

export default function Home() {
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.2,
      },
    },
  }

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
  }

  return (
    <div className="space-y-12">
      {/* Hero */}
      <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="text-center">
        <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl">
          <span className="bg-gradient-to-r from-brand-400 to-accent-400 bg-clip-text text-transparent">LIVEDET</span> — AI Road Defect Detection
        </h1>
        <p className="mt-4 text-lg text-slate-400">
          Real-time pothole and road crack detection using state-of-the-art computer vision. Upload images, videos, or stream live.
        </p>
      </motion.div>

      {/* Options Grid */}
      <motion.div variants={container} initial="hidden" animate="show" className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <motion.div variants={item} className="h-full">
          <CardOption
            icon={(props) => (
              <svg {...props} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            )}
            title="Image Detection"
            description="Upload a single image for instant defect analysis"
            to="/image"
            accentColor="brand"
          />
        </motion.div>

        <motion.div variants={item} className="h-full">
          <CardOption
            icon={(props) => (
              <svg {...props} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
            title="Video Detection"
            description="Process multiple frames with frame-by-frame analysis"
            to="/video"
            accentColor="accent"
          />
        </motion.div>

        <motion.div variants={item} className="h-full">
          <CardOption
            icon={(props) => (
              <svg {...props} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            )}
            title="Live Detection"
            description="Real-time WebSocket streaming for continuous analysis"
            to="/live"
            accentColor="purple"
          />
        </motion.div>
      </motion.div>

      {/* Footer Note */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }} className="text-center">
        <p className="text-xs text-slate-500">Backend running on localhost:8000</p>
      </motion.div>
    </div>
  )
}
