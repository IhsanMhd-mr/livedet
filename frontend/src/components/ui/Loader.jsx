import { motion } from 'framer-motion'

export function Loader({ size = 'md', text = '' }) {
  const sizeMap = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8',
  }

  return (
    <div className="flex flex-col items-center gap-2">
      <motion.div
        className={`${sizeMap[size]} rounded-full border-2 border-brand-500/20 border-t-brand-400`}
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
      />
      {text && <p className="text-xs text-slate-400">{text}</p>}
    </div>
  )
}

export default Loader
