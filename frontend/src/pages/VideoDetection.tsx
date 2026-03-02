import { UploadCloud, Download, Video as VideoIcon } from 'lucide-react'
import { useRef, useState, type DragEvent } from 'react'

export default function VideoDetection() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)

  const handleFile = (f: File) => {
    if (!f.type.startsWith('video/')) return
    setFile(f)
    setVideoUrl(URL.createObjectURL(f))
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  return (
    <main className="mx-auto max-w-4xl px-4 py-10">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-slate-100">Video Detection</h2>
        <p className="mt-1 text-sm text-slate-400">
          Upload a road video for frame-by-frame pothole analysis
        </p>
      </div>

      {/* Coming Soon Banner */}
      <div className="mb-6 flex items-center gap-3 rounded-xl border border-amber-500/20 bg-amber-500/10 px-4 py-3">
        <span className="text-amber-400 text-xl">⚠️</span>
        <div>
          <p className="text-sm font-semibold text-amber-400">Backend endpoint coming soon</p>
          <p className="text-xs text-amber-400/70 mt-0.5">
            Video processing requires a <code className="font-mono">/predict/video</code> endpoint on the backend.
            Add it to <code className="font-mono">app.py</code> and connect it here.
          </p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Upload */}
        <div className="flex flex-col gap-4">
          <div
            role="button"
            tabIndex={0}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => !file && inputRef.current?.click()}
            onKeyDown={(e) => e.key === 'Enter' && !file && inputRef.current?.click()}
            className={`flex min-h-[200px] cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed transition-all duration-200 ${
              dragging
                ? 'border-cyan-400 bg-cyan-500/10'
                : 'border-white/10 bg-surface-800 hover:border-cyan-400/50'
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              className="sr-only"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
            />
            {file ? (
              <div className="flex flex-col items-center gap-2 text-center px-4">
                <VideoIcon className="h-10 w-10 text-cyan-400" />
                <p className="text-sm font-semibold text-slate-200">{file.name}</p>
                <p className="text-xs text-slate-500">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
              </div>
            ) : (
              <>
                <UploadCloud className="h-10 w-10 text-slate-500" />
                <div className="text-center">
                  <p className="text-sm font-medium text-slate-300">
                    Drag & drop or <span className="text-cyan-400 underline underline-offset-2">browse</span>
                  </p>
                  <p className="mt-1 text-xs text-slate-500">MP4, MOV, AVI, MKV</p>
                </div>
              </>
            )}
          </div>

          {/* Fake progress for future use */}
          {file && (
            <button
              type="button"
              disabled
              className="flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-cyan-500/30 text-sm font-semibold text-cyan-300 cursor-not-allowed opacity-60"
            >
              Process Video (not yet available)
            </button>
          )}
        </div>

        {/* Preview */}
        <div className="flex flex-col gap-4">
          {videoUrl ? (
            <>
              <video
                src={videoUrl}
                controls
                className="w-full rounded-2xl border border-white/5 bg-black"
              />
              <a
                href={videoUrl}
                download={file?.name ?? 'video'}
                className="flex h-10 items-center justify-center gap-2 rounded-xl border border-white/10 bg-surface-700 text-sm font-medium text-slate-300 hover:bg-surface-600 transition-colors"
              >
                <Download className="h-4 w-4" />
                Download Original
              </a>
            </>
          ) : (
            <div className="flex flex-1 items-center justify-center rounded-2xl border border-white/5 bg-surface-800 py-24 text-center">
              <p className="text-sm text-slate-500">Video preview will appear here</p>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
