import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'
import api from '@/api/axios'
import Loader from '@/components/ui/Loader'

const ACCEPTED = ['video/mp4', 'video/webm', 'video/quicktime', 'video/x-msvideo']
const MAX_BYTES = 500 * 1024 * 1024

const fmtBytes = (b) => (b < 1024 ** 2 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1024 ** 2).toFixed(1)} MB`)

export default function VideoDetection() {
  const [file, setFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [videoUrl, setVideoUrl] = useState(null)
  const [meta, setMeta] = useState(null)
  const inputRef = useRef(null)

  const validate = (f) => {
    if (!ACCEPTED.includes(f.type)) {
      toast.error('Invalid file — use MP4, WebM, MOV or AVI')
      return false
    }
    if (f.size > MAX_BYTES) {
      toast.error('File exceeds 500 MB')
      return false
    }
    return true
  }

  const loadFile = useCallback((f) => {
    if (!validate(f)) return
    setFile(f)
    setVideoUrl(null)
    setMeta(null)
    setProgress(0)
    toast.success('Video selected')
  }, [])

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0])
  }

  const handleSubmit = async () => {
    if (!file) return
    setUploading(true)
    setProgress(0)
    const form = new FormData()
    form.append('file', file)
    try {
      const { data } = await api.post('/video/process', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => setProgress(Math.round((e.loaded * 100) / (e.total || file.size))),
      })
      if (data.output_url) {
        setVideoUrl(
          data.output_url.startsWith('http')
            ? data.output_url
            : `${import.meta.env.VITE_API_BASE_URL}${data.output_url}`,
        )
      }
      setMeta({
        frames: data.total_frames,
        detections: data.total_detections,
        fps: data.fps,
        duration: data.duration,
      })
      toast.success('Video processed successfully')
    } catch {
      /* handled by interceptor */
    } finally {
      setUploading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setVideoUrl(null)
    setMeta(null)
    setProgress(0)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="mx-auto max-w-5xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Video Detection</h1>
        <p className="mt-1 text-sm text-slate-400">Upload a video for frame-by-frame analysis</p>
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
              ${dragOver ? 'border-brand-400 bg-brand-500/10 cursor-copy' : file ? 'border-white/10 bg-surface-800/40' : 'cursor-pointer border-white/10 bg-surface-800/40 hover:border-brand-500/50 hover:bg-brand-500/5'}`}
          >
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED.join(',')}
              onChange={(e) => e.target.files[0] && loadFile(e.target.files[0])}
              className="hidden"
            />
            {file ? (
              <div className="space-y-3 text-center">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-xl bg-brand-500/10">
                  <svg className="h-7 w-7 text-brand-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  </svg>
                </div>
                <div>
                  <p className="max-w-xs truncate text-sm font-medium text-slate-200">{file.name}</p>
                  <p className="text-xs text-slate-500">{fmtBytes(file.size)}</p>
                </div>
              </div>
            ) : (
              <>
                <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-xl bg-brand-500/10">
                  <svg className="h-7 w-7 text-brand-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33A3 3 0 0116.5 19.5H6.75z" />
                  </svg>
                </div>
                <p className="text-sm font-medium text-slate-300">{dragOver ? 'Drop here' : 'Drop video or click to browse'}</p>
                <p className="mt-1 text-xs text-slate-500">MP4, WebM, MOV, AVI · max 500 MB</p>
              </>
            )}
          </div>

          {uploading && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-slate-400">
                <span>{progress < 100 ? 'Uploading…' : 'Processing…'}</span>
                <span>{progress}%</span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-700">
                <motion.div
                  className="h-full bg-brand-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ ease: 'easeOut' }}
                />
              </div>
            </div>
          )}

          {file && (
            <div className="flex gap-3">
              <button
                onClick={handleSubmit}
                disabled={uploading}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-brand-500 px-5 py-3 text-sm font-semibold text-white transition-colors hover:bg-brand-600 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {uploading && <Loader size="sm" />}
                {uploading ? 'Processing…' : 'Process Video'}
              </button>
              <button
                onClick={reset}
                disabled={uploading}
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
            {uploading && (
              <motion.div
                key="loader"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex min-h-64 items-center justify-center rounded-2xl border border-white/5 bg-surface-800/40"
              >
                <Loader size="lg" text={progress < 100 ? 'Uploading…' : 'Processing…'} />
              </motion.div>
            )}
            {!uploading && videoUrl && (
              <motion.div key="video" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
                <div className="overflow-hidden rounded-2xl border border-white/5 bg-black">
                  <video src={videoUrl} controls className="w-full" playsInline />
                </div>
                <a
                  href={videoUrl}
                  download="livedet_result.mp4"
                  className="flex items-center justify-center gap-2 rounded-xl bg-surface-700 px-5 py-3 text-sm font-medium text-slate-200 transition-colors hover:bg-surface-600"
                >
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                  </svg>
                  Download Result
                </a>
              </motion.div>
            )}
            {!uploading && !videoUrl && (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex min-h-64 flex-col items-center justify-center rounded-2xl border border-white/5 bg-surface-800/20 text-slate-600"
              >
                <svg className="mb-3 h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                </svg>
                <p className="text-sm">Processed video appears here</p>
              </motion.div>
            )}
          </AnimatePresence>

          {meta && (
            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-2 gap-3">
              {[
                { label: 'Frames', value: meta.frames ?? '—' },
                { label: 'Detections', value: meta.detections ?? '—' },
                { label: 'FPS', value: meta.fps ?? '—' },
                { label: 'Duration', value: meta.duration != null ? `${meta.duration.toFixed(1)}s` : '—' },
              ].map(({ label, value }) => (
                <div key={label} className="rounded-xl border border-white/5 bg-surface-800/60 p-3">
                  <p className="text-xs text-slate-500">{label}</p>
                  <p className="mt-0.5 font-mono text-lg font-bold text-white">{value}</p>
                </div>
              ))}
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}
