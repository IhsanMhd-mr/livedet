import { useCallback, useEffect, useRef, useState } from 'react'
import { LiveWSManager, type WSStatus } from '@/api/websocket'
import type { WSFrame } from '@/types/detection'

interface Options {
  url?: string
  autoConnect?: boolean
  onFrame?: (frame: WSFrame) => void
}

export function useWebSocket({ url, autoConnect = false, onFrame }: Options = {}) {
  const [status, setStatus] = useState<WSStatus>('disconnected')
  const [lastFrame, setLastFrame] = useState<WSFrame | null>(null)
  const [error, setError] = useState<string | null>(null)
  const managerRef = useRef<LiveWSManager | null>(null)

  // Keep onFrame ref stable so the manager closure doesn't stale
  const onFrameRef = useRef(onFrame)
  useEffect(() => { onFrameRef.current = onFrame }, [onFrame])

  // Initialise manager once
  useEffect(() => {
    managerRef.current = new LiveWSManager({
      url,
      onFrame: (frame) => {
        setLastFrame(frame)
        onFrameRef.current?.(frame)
      },
      onStatusChange: setStatus,
      onError: (msg) => setError(msg),
    })

    if (autoConnect) {
      managerRef.current.connect()
    }

    return () => {
      managerRef.current?.disconnect()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url])

  const connect = useCallback(() => {
    setError(null)
    managerRef.current?.connect()
  }, [])

  const disconnect = useCallback(() => {
    managerRef.current?.disconnect()
  }, [])

  const sendFrame = useCallback((base64: string) => {
    managerRef.current?.sendFrame(base64)
  }, [])

  return { status, lastFrame, error, connect, disconnect, sendFrame }
}
