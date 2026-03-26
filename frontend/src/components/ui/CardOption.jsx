import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'

export function CardOption({ icon: Icon, title, description, to, accentColor = 'brand' }) {
  const colorMap = {
    brand: 'to-brand-500',
    accent: 'to-accent-500',
    purple: 'to-purple-500',
  }

  return (
    <Link to={to}>
      <motion.div
        whileHover={{ scale: 1.05, y: -4 }}
        className="group relative overflow-hidden rounded-2xl border border-white/5 bg-surface-800/60 p-6 backdrop-blur-sm transition-all hover:border-white/10"
      >
        <div className={`absolute -right-8 -top-8 h-32 w-32 rounded-full bg-gradient-to-br from-transparent ${colorMap[accentColor]} opacity-0 blur-2xl transition-all group-hover:opacity-20`} />

        <div className="relative z-10">
          <div className={`mb-4 inline-flex rounded-xl bg-${accentColor}-500/10 p-3`}>
            <Icon className={`h-6 w-6 text-${accentColor}-400`} />
          </div>

          <h3 className="mb-2 text-lg font-bold text-white">{title}</h3>
          <p className="text-sm leading-relaxed text-slate-400">{description}</p>

          <motion.div
            initial={{ x: -4, opacity: 0 }}
            whileHover={{ x: 4, opacity: 1 }}
            className="mt-4 inline-flex items-center gap-1 text-xs font-semibold text-slate-300"
          >
            Try it →
          </motion.div>
        </div>
      </motion.div>
    </Link>
  )
}

export default CardOption
