import { useEffect, useRef, useCallback, useState } from 'react'
import { createWebSocketManager } from '@/api/websocket'

const BASE_DELAY = 1000
const MAX_DELAY = 30000
const MAX_RETRIES = 8

export function useWebSocket(path = '', autoConnect = false) {
  const wsRef = useRef(null)
  const retryCount = useRef(0)
  const retryTimer = useRef(null)
  const shouldRetry = useRef(false)
  const [wsStatus, setWsStatus] = useState('disconnected')
  const [wsError, setWsError] = useState(null)
  const [currentFrame, setCurrentFrame] = useState(null)
  const [detections, setDetections] = useState([])
  const [fps, setFps] = useState(0)
  const [depthActive, setDepthActive] = useState(false)
  const fpsFrameCount = useRef(0)
  const fpsLastTs = useRef(Date.now())

  // Reconnect state variables
  const [reconnectAttempt, setReconnectAttempt] = useState(0)
  const [currentReconnectDelay, setCurrentReconnectDelay] = useState(0)
  const [isReconnecting, setIsReconnecting] = useState(false)
  const [hasConnectedBefore, setHasConnectedBefore] = useState(false)
  const [hasReconnectExhausted, setHasReconnectExhausted] = useState(false)

  // Ref to hold the latest connectInternal method to break circular dependency
  const connectInternalRef = useRef(null)

  const scheduleRetry = useCallback(() => {
    if (retryCount.current >= MAX_RETRIES) {
      setWsStatus('error')
      setWsError(`Failed after ${MAX_RETRIES} reconnect attempts`)
      setHasReconnectExhausted(true)
      setIsReconnecting(false)
      shouldRetry.current = false
      return
    }

    const delay = Math.min(BASE_DELAY * 2 ** retryCount.current, MAX_DELAY)
    
    setIsReconnecting(true)
    setReconnectAttempt(retryCount.current + 1)
    setCurrentReconnectDelay(delay / 1000)
    
    retryCount.current += 1
    retryTimer.current = setTimeout(() => {
      if (shouldRetry.current && connectInternalRef.current) {
        connectInternalRef.current()
      }
    }, delay)
  }, [setWsStatus, setWsError])

  const connectInternal = useCallback(() => {
    let isFirstTime = false
    if (!wsRef.current) {
      wsRef.current = createWebSocketManager(path)
      isFirstTime = true
    }
    const ws = wsRef.current

    setWsStatus('connecting')

    if (isFirstTime) {
      ws.on('open', () => {
        retryCount.current = 0
        setReconnectAttempt(0)
        setIsReconnecting(false)
        setHasConnectedBefore(true)
        setHasReconnectExhausted(false)
        setWsStatus('connected')
        setWsError(null)
        fpsFrameCount.current = 0
        fpsLastTs.current = Date.now()
      })

      ws.on('message', (data) => {
        if (data.frame) {
          setCurrentFrame(`data:image/jpeg;base64,${data.frame}`)
          const now = Date.now()
          fpsFrameCount.current += 1
          const elapsed = now - fpsLastTs.current
          if (elapsed >= 1000) {
            setFps(Math.round((fpsFrameCount.current * 1000) / elapsed))
            fpsFrameCount.current = 0
            fpsLastTs.current = now
          }
        } else if (data.fps !== undefined) {
          setFps(data.fps)
        }
        if (data.detections) setDetections(data.detections)
        if (data.depth_active !== undefined) setDepthActive(data.depth_active)
      })

      ws.on('error', () => setWsStatus('error'))

      ws.on('close', (event) => {
        setWsStatus('disconnected')
        if (shouldRetry.current && event.code !== 1000) {
          scheduleRetry()
        }
      })
    }

    ws.connect()
  }, [path, scheduleRetry])

  // Sync ref to connectInternal
  useEffect(() => {
    connectInternalRef.current = connectInternal
  }, [connectInternal])

  const connect = useCallback(() => {
    shouldRetry.current = true
    retryCount.current = 0
    setReconnectAttempt(0)
    setIsReconnecting(false)
    setHasConnectedBefore(false)
    setHasReconnectExhausted(false)
    connectInternal()
  }, [connectInternal])

  const disconnect = useCallback(() => {
    shouldRetry.current = false
    if (retryTimer.current) {
      clearTimeout(retryTimer.current)
      retryTimer.current = null
    }
    wsRef.current?.disconnect()
    setWsStatus('disconnected')
    setWsError(null)
    setCurrentFrame(null)
    setDetections([])
    setFps(0)
    fpsFrameCount.current = 0
    fpsLastTs.current = Date.now()

    setReconnectAttempt(0)
    setIsReconnecting(false)
    setHasConnectedBefore(false)
    setHasReconnectExhausted(false)
  }, [])

  useEffect(() => {
    if (autoConnect) connect()
    return () => disconnect()
  }, [autoConnect, connect, disconnect])

  const send = useCallback((data) => {
    wsRef.current?.send(data)
  }, [])

  return {
    connect,
    disconnect,
    wsStatus,
    wsError,
    currentFrame,
    detections,
    fps,
    depthActive,
    send,
    reconnectAttempt,
    maxReconnectAttempts: MAX_RETRIES,
    currentReconnectDelay,
    isReconnecting,
    hasConnectedBefore,
    hasReconnectExhausted
  }
}

export default useWebSocket
