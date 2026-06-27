import { motion } from 'framer-motion'

export default function Demo() {
  const demos = [
    { 
      name: 'Image Detection', 
      src: '/videos/image_demo.mp4',
      aspectClass: 'aspect-video w-full',
      videoClass: 'w-full h-full object-cover'
    },
    { 
      name: 'Video Detection', 
      src: '/videos/video_demo.mp4',
      aspectClass: 'aspect-video w-full',
      videoClass: 'w-full h-full object-cover'
    },
    { 
      name: 'Live Detection', 
      src: '/videos/live_demo.mp4',
      aspectClass: 'aspect-[9/16] max-h-[360px] w-full mx-auto',
      videoClass: 'w-full h-full object-contain'
    },
  ]

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="border-b border-white/10 pb-4">
        <h1 className="text-3xl font-mono text-white">Demos</h1>
        <span className="text-[10px] text-amber-500 font-mono">[temporary reference page]</span>
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-3 items-start">
        {demos.map((demo) => (
          <div key={demo.name} className="p-4 rounded-lg border border-white/10 bg-surface-950/40 space-y-3">
            <div>
              <h3 className="font-semibold text-white font-mono mb-3">{demo.name}</h3>
              <div className={`relative rounded border border-white/5 bg-black/60 overflow-hidden flex items-center justify-center ${demo.aspectClass}`}>
                <video
                  src={demo.src}
                  controls
                  className={demo.videoClass}
                />
              </div>
            </div>
            <div className="text-[10px] text-slate-500 font-mono break-all bg-black/20 p-2 rounded mt-3">
              Path: frontend/public{demo.src}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
