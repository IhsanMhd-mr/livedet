import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'
import api from '@/api/axios'
import useAppStore from '@/store/useAppStore'
import Loader from '@/components/ui/Loader'
import DetectionCard from '@/components/ui/DetectionCard'

const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp']
const MAX_MB = 20

export default function ImageDetection() {
  const [dragOver, setDragOver] = useState(false)
  const [preview, setPreview] = useState(null)
  const [file, setFile] = useState(null)
  const inputRef = useRef(null)

  const loading = useAppStore((s) => s.loading)
  const setLoading = useAppStore((s) => s.setLoading)
  const detections = useAppStore((s) => s.detections)
  const annotatedImage = useAppStore((s) => s.annotatedImage)
  const setResult = useAppStore((s) => s.setDetectionResult)
  const clearDetections = useAppStore((s) => s.clearDetections)

  const validate = (f) => {
    if (!ACCEPTED.includes(f.type)) {
      toast.error('Invalid file — use JPG, PNG or WebP')
      return false
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      toast.error(`File exceeds ${MAX_MB} MB`)
      return false
    }
    return true
  }

  const loadFile = useCallback(
    (f) => {
      if (!validate(f)) return
      setFile(f)
      clearDetections()
      const r = new FileReader()
      r.onload = (e) => setPreview(e.target.result)
      r.readAsDataURL(f)
      toast.success('Image selected')
    },
    [clearDetections],
  )

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0])
  }

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    const form = new FormData()
    form.append('file', file)
    try {
      const { data } = await api.post('/predict', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResult({
        detections: data.detections || [],
        annotated_image: data.annotated_image || null,
      })
      toast.success('Detection complete')
    } catch {
      /* handled by interceptor */
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setPreview(null)
    clearDetections()
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="mx-auto max-w-5xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Image Detection</h1>
        <p className="mt-1 text-sm text-slate-400">Upload an image to analyze road defects</p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Upload */}
        <div className="space-y-4">
          <div
            onDragOver={(e) => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => !file && inputRef.current?.click()}
            className={`min-h-64 flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 text-center transition-all
              ${dragOver ? 'border-accent-400 bg-accent-500/10 cursor-copy' : file ? 'border-white/10 bg-surface-800/40' : 'cursor-pointer border-white/10 bg-surface-800/40 hover:border-accent-500/50 hover:bg-accent-500/5'}`}
          >
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED.join(',')}
              onChange={(e) => e.target.files[0] && loadFile(e.target.files[0])}
              className="hidden"
            />
            {file ? (
              <div className="space-y-3 text-center w-full h-full flex flex-col items-center justify-center">
                {preview ? (
                  <div className="relative overflow-hidden rounded-xl border border-white/5 max-h-48 max-w-full bg-black">
                    <img src={preview} alt="Selected preview" className="max-h-48 max-w-full object-contain" />
                  </div>
                ) : (
                  <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-xl bg-accent-500/10">
                    <svg className="h-7 w-7 text-accent-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  </div>
                )}
                <div>
                  <p className="max-w-xs truncate text-sm font-medium text-slate-200">{file.name}</p>
                  <p className="text-xs text-slate-500">{(file.size / 1024).toFixed(0)} KB</p>
                </div>
              </div>
            ) : (
              <>
                <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-xl bg-accent-500/10">
                  <svg className="h-7 w-7 text-accent-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33A3 3 0 0116.5 19.5H6.75z" />
                  </svg>
                </div>
                <p className="text-sm font-medium text-slate-300">{dragOver ? 'Drop here' : 'Drop image or click to browse'}</p>
                <p className="mt-1 text-xs text-slate-500">PNG, JPG, WebP · max {MAX_MB} MB</p>
              </>
            )}
          </div>

          {file && (
            <div className="flex gap-3">
              <button
                onClick={handleSubmit}
                disabled={loading}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-accent-500 px-5 py-3 text-sm font-semibold text-white transition-colors hover:bg-accent-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading && <Loader size="sm" />}
                {loading ? 'Detecting…' : 'Run Detection'}
              </button>
              <button
                onClick={reset}
                disabled={loading}
                className="rounded-xl border border-white/10 px-4 py-3 text-sm text-slate-400 transition-colors hover:bg-white/5 disabled:opacity-40"
              >
                Reset
              </button>
            </div>
          )}
        </div>

        {/* Result */}
        <div className="space-y-4">
          <AnimatePresence mode="wait">
            {loading && (
              <motion.div
                key="loader"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex min-h-64 items-center justify-center rounded-2xl border border-white/5 bg-surface-800/40"
              >
                <Loader size="lg" text="Running detection…" />
              </motion.div>
            )}
            {!loading && annotatedImage && (
              <motion.div key="image" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
                <div className="overflow-hidden rounded-2xl border border-white/5 bg-black">
                  <img src={annotatedImage} alt="Annotated" className="w-full" />
                </div>
              </motion.div>
            )}
            {!loading && !annotatedImage && (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex min-h-64 flex-col items-center justify-center rounded-2xl border border-white/5 bg-surface-800/20 text-slate-600"
              >
                <svg className="mb-3 h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p className="text-sm">Annotated image appears here</p>
              </motion.div>
            )}
          </AnimatePresence>

          {detections.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-xl border border-accent-500/20 bg-accent-500/5 p-4"
            >
              <h3 className="mb-3 font-semibold text-accent-400">
                Detections <span className="text-accent-400">({detections.length})</span>
              </h3>
              <div className="space-y-2">
                {detections.map((d, i) => (
                  <DetectionCard key={i} detection={d} index={i} mode="image" />
                ))}
              </div>
            </motion.div>
          )}

          {!loading && annotatedImage && detections.length === 0 && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-5 text-center"
            >
              <div className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-full bg-emerald-500/10 text-emerald-400">
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h3 className="font-semibold text-emerald-400">Clean Road!</h3>
              <p className="mt-1 text-sm text-slate-400">No potholes or road defects detected.</p>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}
