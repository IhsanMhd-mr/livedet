import { useRef, useState, useCallback, type DragEvent } from 'react'
import { UploadCloud, X, SlidersHorizontal } from 'lucide-react'
import { useDetection } from '@/hooks/useDetection'
import DetectionResult from '@/components/DetectionResult'
import Loader from '@/components/Loader'

const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']

export default function ImageDetection() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [confidence, setConfidence] = useState(0.5)
  const [showConf, setShowConf] = useState(false)

  const { loading, result, error, detect, reset } = useDetection()

  const handleFile = useCallback((f: File) => {
    if (!ACCEPTED.includes(f.type)) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    reset()
  }, [reset])

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) handleFile(f)
  }

  const clearFile = () => {
    setFile(null)
    setPreview(null)
    reset()
    if (inputRef.current) inputRef.current.value = ''
  }

  const onSubmit = async () => {
    if (file) await detect(file, confidence)
  }

  return (
    <main className="mx-auto max-w-5xl px-4 py-10">
      <PageHeader
        title="Image Detection"
        subtitle="Upload an image to detect and classify potholes with your trained YOLOv8 model"
      />

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        {/* Upload Panel */}
        <div className="flex flex-col gap-4">
          {/* Drop Zone */}
          <div
            role="button"
            tabIndex={0}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => !file && inputRef.current?.click()}
            onKeyDown={(e) => e.key === 'Enter' && !file && inputRef.current?.click()}
            className={`relative flex min-h-[220px] cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed transition-all duration-200 ${
              dragging
                ? 'border-accent bg-accent/10'
                : file
                ? 'cursor-default border-white/10 bg-surface-800'
                : 'border-white/10 bg-surface-800 hover:border-accent/50 hover:bg-surface-700'
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED.join(',')}
              className="sr-only"
              onChange={onInputChange}
            />

            {preview ? (
              <>
                <img
                  src={preview}
                  alt="preview"
                  className="max-h-48 w-full rounded-xl object-contain"
                />
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); clearFile() }}
                  className="absolute right-3 top-3 rounded-full bg-surface-900/80 p-1 text-slate-400 hover:text-red-400"
                >
                  <X className="h-4 w-4" />
                </button>
              </>
            ) : (
              <>
                <UploadCloud className="h-10 w-10 text-slate-500" />
                <div className="text-center">
                  <p className="text-sm font-medium text-slate-300">
                    Drag & drop or <span className="text-accent-light underline underline-offset-2">browse</span>
                  </p>
                  <p className="mt-1 text-xs text-slate-500">JPEG, PNG, WebP · Max 20 MB</p>
                </div>
              </>
            )}
          </div>

          {/* Confidence toggle */}
          <div className="rounded-xl border border-white/5 bg-surface-800 px-4 py-3">
            <button
              type="button"
              onClick={() => setShowConf((p) => !p)}
              className="flex w-full items-center gap-2 text-xs font-medium text-slate-400 hover:text-slate-300"
            >
              <SlidersHorizontal className="h-3.5 w-3.5" />
              Confidence threshold: <span className="text-accent-light">{(confidence * 100).toFixed(0)}%</span>
            </button>
            {showConf && (
              <input
                type="range"
                min={0.1}
                max={0.95}
                step={0.05}
                value={confidence}
                onChange={(e) => setConfidence(Number(e.target.value))}
                className="mt-3 w-full accent-indigo-500"
              />
            )}
          </div>

          {/* Submit */}
          <button
            type="button"
            disabled={!file || loading}
            onClick={onSubmit}
            className="flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-accent text-sm font-semibold text-white shadow-glow transition-all hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {loading ? <Loader size="sm" /> : 'Analyse Image'}
          </button>

          {error && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
              {error}
            </div>
          )}
        </div>

        {/* Result Panel */}
        <div className="flex flex-col gap-4">
          {loading && (
            <div className="flex flex-1 items-center justify-center rounded-2xl border border-white/5 bg-surface-800 py-20">
              <Loader label="Running inference…" size="lg" />
            </div>
          )}

          {result && !loading && (
            <>
              {/* Annotated image */}
              <div className="overflow-hidden rounded-2xl border border-white/5 bg-surface-800">
                <img
                  src={`data:image/jpeg;base64,${result.image}`}
                  alt="annotated"
                  className="w-full object-contain"
                />
              </div>

              {/* Detections */}
              <DetectionResult
                detections={result.detections}
                summary={result.summary}
              />

              {/* Meta */}
              <div className="flex flex-wrap gap-3 text-[11px] text-slate-500 font-mono">
                <span>Session: {result.session_id}</span>
                <span>Model: {result.model}</span>
                <span>{result.image_shape[1]}×{result.image_shape[0]}</span>
              </div>
            </>
          )}

          {!loading && !result && (
            <div className="flex flex-1 flex-col items-center justify-center rounded-2xl border border-white/5 bg-surface-800 py-20 text-center">
              <p className="text-sm text-slate-500">Results will appear here</p>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}

function PageHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div>
      <h2 className="text-2xl font-bold text-slate-100">{title}</h2>
      <p className="mt-1 text-sm text-slate-400">{subtitle}</p>
    </div>
  )
}
