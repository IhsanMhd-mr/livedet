import type { WSFrame } from '@/types/detection'

export type WSStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

export interface WSManagerOptions {
  url?: string
  onFrame: (frame: WSFrame) => void
  onStatusChange: (status: WSStatus) => void
  onError?: (msg: string) => void
  reconnectDelay?: number
  maxReconnectAttempts?: number
}

// ─── WebSocket Manager ────────────────────────────────────────────────────────
// The live_ws server expects raw base64-encoded JPEG bytes (no JSON wrapper).
// It responds with a JSON object: { detections, fps, frame_count, depth_active }

export class LiveWSManager {
  private ws: WebSocket | null = null
  private url: string
  private opts: WSManagerOptions
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private reconnectAttempt = 0
  private destroyed = false

  constructor(opts: WSManagerOptions) {
    this.url = opts.url ?? import.meta.env.VITE_WS_URL ?? 'ws://localhost:8765'
    this.opts = opts
  }

  connect() {
    if (this.ws) this.disconnect()
    this.destroyed = false
    this._open()
  }

  private _open() {
    this.opts.onStatusChange('connecting')
    try {
      this.ws = new WebSocket(this.url)
    } catch (e) {
      this.opts.onStatusChange('error')
      this.opts.onError?.(`Could not create WebSocket: ${e}`)
      return
    }

    this.ws.onopen = () => {
      this.reconnectAttempt = 0
      this.opts.onStatusChange('connected')
    }

    this.ws.onmessage = (ev) => {
      try {
        const frame: WSFrame = JSON.parse(ev.data as string)
        if (frame.error) {
          this.opts.onError?.(frame.error)
          return
        }
        this.opts.onFrame(frame)
      } catch {
        // ignore malformed frames
      }
    }

    this.ws.onerror = () => {
      this.opts.onStatusChange('error')
      this.opts.onError?.('WebSocket connection error')
    }

    this.ws.onclose = () => {
      if (this.destroyed) return
      this.opts.onStatusChange('disconnected')
      const max = this.opts.maxReconnectAttempts ?? 0 // 0 = no reconnect by default
      if (max > 0 && this.reconnectAttempt < max) {
        const delay = this.opts.reconnectDelay ?? 2000
        this.reconnectTimer = setTimeout(() => {
          this.reconnectAttempt++
          this._open()
        }, delay)
      }
    }
  }

  /**
   * Send a raw base64-encoded JPEG frame to the server.
   * Strip the data-URI prefix if present.
   */
  sendFrame(base64: string) {
    if (this.ws?.readyState !== WebSocket.OPEN) return
    const cleaned = base64.startsWith('data:')
      ? base64.split(',')[1]
      : base64
    this.ws.send(cleaned)
  }

  disconnect() {
    this.destroyed = true
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    if (this.ws) {
      this.ws.onclose = null
      this.ws.close()
      this.ws = null
    }
    this.opts.onStatusChange('disconnected')
  }

  get readyState() {
    return this.ws?.readyState ?? WebSocket.CLOSED
  }
}
