// ─── Detection Core ──────────────────────────────────────────────────────────

export type SeverityLevel = 'Low' | 'Medium' | 'High' | 'Critical'

export interface BoundingBox {
  x: number
  y: number
  w: number
  h: number
}

export interface Detection {
  id: number
  bbox: [number, number, number, number]
  class_name: string
  confidence: number
  depth_cm: number
  width_cm: number
  severity: SeverityLevel
  severity_score?: number
  area?: number
}

// ─── Predict API ─────────────────────────────────────────────────────────────

export interface SummaryStats {
  total: number
  severity_counts: Record<SeverityLevel, number>
  avg_depth_cm: number
  avg_width_cm: number
  max_depth_cm: number
  max_width_cm: number
}

export interface PredictResponse {
  status: 'success' | 'error'
  session_id: string
  timestamp: string
  model: string
  detections: Detection[]
  count: number
  num_potholes: number
  image: string // base64
  image_shape: [number, number, number]
  summary: SummaryStats
  average_depth_cm: number
  average_width_cm: number
  max_depth_cm: number
  max_width_cm: number
}

// ─── WebSocket ───────────────────────────────────────────────────────────────

export interface WSDetection {
  id: number
  bbox: [number, number, number, number]
  class_name: string
  confidence: number
  depth_cm: number
  width_cm: number
  severity: SeverityLevel
}

export interface WSFrame {
  detections: WSDetection[]
  num_detections: number
  fps: number
  frame_count: number
  depth_active: boolean
  error?: string
}

// ─── Health ──────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string
  model_loaded: boolean
  model_name: string | null
  device: string
  timestamp: string
}

// ─── Severity helpers ─────────────────────────────────────────────────────────

export const SEVERITY_COLORS: Record<SeverityLevel, string> = {
  Low: '#22c55e',
  Medium: '#f59e0b',
  High: '#ef4444',
  Critical: '#dc2626',
}

export const SEVERITY_BG: Record<SeverityLevel, string> = {
  Low: 'bg-green-500/20 text-green-400 ring-green-500/30',
  Medium: 'bg-amber-500/20 text-amber-400 ring-amber-500/30',
  High: 'bg-red-500/20 text-red-400 ring-red-500/30',
  Critical: 'bg-rose-700/20 text-rose-400 ring-rose-700/30',
}
