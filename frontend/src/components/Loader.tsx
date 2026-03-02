interface Props {
  size?: 'sm' | 'md' | 'lg'
  label?: string
}

const SIZE = {
  sm: 'h-4 w-4 border-2',
  md: 'h-8 w-8 border-2',
  lg: 'h-12 w-12 border-[3px]',
}

export default function Loader({ size = 'md', label }: Props) {
  return (
    <div className="flex flex-col items-center gap-3">
      <div
        className={`${SIZE[size]} animate-spin rounded-full border-accent/30 border-t-accent`}
      />
      {label && (
        <p className="text-xs text-slate-400 animate-pulse">{label}</p>
      )}
    </div>
  )
}
