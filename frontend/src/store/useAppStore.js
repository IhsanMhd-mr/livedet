import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

const useAppStore = create(
  devtools(
    (set, get) => ({
      detections: [],
      annotatedImage: null,
      currentFrame: null,
      totalDetections: 0,
      loading: false,
      error: null,
      wsStatus: 'disconnected',
      wsError: null,
      fps: 0,
      _fpsFrameCount: 0,
      _fpsLastTs: Date.now(),
      selectedMode: 'image',

      setLoading: (loading) => set({ loading }),
      setError: (error) => set({ error }),
      setSelectedMode: (selectedMode) => set({ selectedMode }),

      setDetectionResult: ({ detections = [], annotated_image = null, total_detections = null } = {}) => {
        set({
          detections,
          annotatedImage: annotated_image,
          totalDetections: total_detections ?? detections.length,
          error: null,
        })
      },

      clearDetections: () => {
        set({ detections: [], annotatedImage: null, totalDetections: 0, error: null })
      },

      setWsStatus: (wsStatus) => set({ wsStatus }),
      setWsError: (wsError) => set({ wsError }),

      setLiveFrame: (currentFrame) => {
        const now = Date.now()
        const count = get()._fpsFrameCount + 1
        const lastTs = get()._fpsLastTs
        const elapsed = now - lastTs

        if (elapsed >= 1000) {
          const fps = Math.round((count * 1000) / elapsed)
          set({
            currentFrame,
            fps,
            _fpsFrameCount: 0,
            _fpsLastTs: now,
          })
        } else {
          set({
            currentFrame,
            _fpsFrameCount: count,
          })
        }
      },

      setLiveDetections: (detections = []) => {
        set({ detections, totalDetections: detections.length })
      },

      resetLive: () => {
        set({
          currentFrame: null,
          detections: [],
          totalDetections: 0,
          fps: 0,
          _fpsFrameCount: 0,
          _fpsLastTs: Date.now(),
          wsError: null,
        })
      },
    }),
    { name: 'livedet-app-store' },
  ),
)

export default useAppStore
