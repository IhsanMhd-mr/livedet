import { Image as ImageIcon, Video, Radio, Activity } from 'lucide-react'
import CardOption from '@/components/CardOption'

const CARDS = [
  {
    to: '/image',
    icon: ImageIcon,
    title: 'Image Detection',
    description:
      'Upload a road image for instant pothole detection with your trained YOLOv8 model. Get annotated results with depth, width, and severity scores.',
    badge: undefined,
    accentColor: 'from-indigo-500 to-violet-600',
  },
  {
    to: '/video',
    icon: Video,
    title: 'Video Detection',
    description:
      'Submit a video file for frame-by-frame pothole detection. Download the fully annotated output.',
    badge: 'Coming soon',
    accentColor: 'from-cyan-500 to-sky-600',
  },
  {
    to: '/live',
    icon: Radio,
    title: 'Live Detection',
    description:
      'Stream your webcam in real time over WebSocket. Detects potholes live, overlays bounding boxes with depth and severity at runtime.',
    badge: 'Real-time',
    accentColor: 'from-emerald-500 to-teal-600',
  },
]

export default function Home() {
  return (
    <main className="mx-auto flex max-w-5xl flex-col items-center px-4 py-20">
      {/* Hero */}
      <div className="mb-14 flex flex-col items-center gap-5 text-center animate-fade-in">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-accent to-violet-600 shadow-glow">
          <Activity className="h-7 w-7 text-white" />
        </div>
        <div>
          <h1 className="text-4xl font-extrabold tracking-tight text-slate-100 sm:text-5xl">
            Live<span className="text-accent-light">Det</span>
          </h1>
          <p className="mt-3 max-w-xl text-base text-slate-400">
            Pothole detection powered by a trained YOLOv8 model. Includes MiDaS depth estimation
            and real-time severity classification.
          </p>
        </div>

        {/* Pills */}
        <div className="flex flex-wrap justify-center gap-2">
          {['YOLOv8', 'MiDaS Depth', 'Real-Time WS', 'REST API'].map((tag) => (
            <span
              key={tag}
              className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[11px] font-medium tracking-wide text-slate-400"
            >
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* Cards */}
      <div className="grid w-full gap-5 sm:grid-cols-3 animate-slide-up">
        {CARDS.map((card) => (
          <CardOption key={card.to} {...card} />
        ))}
      </div>
    </main>
  )
}
