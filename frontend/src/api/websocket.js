export function createWebSocketManager(path) {
  let socket = null
  const listeners = {}

  return {
    connect() {
      const wsBase = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8765'
      const url = path ? `${wsBase}${path}` : wsBase

      socket = new WebSocket(url)

      socket.addEventListener('open', () => {
        if (listeners['open']) listeners['open'].forEach((cb) => cb())
      })

      socket.addEventListener('message', (event) => {
        try {
          const data = JSON.parse(event.data)
          if (listeners['message']) listeners['message'].forEach((cb) => cb(data))
        } catch {
          console.error('Failed to parse WS message:', event.data)
        }
      })

      socket.addEventListener('error', (event) => {
        if (listeners['error']) listeners['error'].forEach((cb) => cb(event))
      })

      socket.addEventListener('close', (event) => {
        if (listeners['close']) listeners['close'].forEach((cb) => cb(event))
        socket = null
      })
    },

    disconnect() {
      if (socket) {
        socket.close(1000)
        socket = null
      }
    },

    send(data) {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(data))
      }
    },

    isConnected() {
      return socket && socket.readyState === WebSocket.OPEN
    },

    on(event, callback) {
      if (!listeners[event]) listeners[event] = []
      listeners[event].push(callback)
    },

    off(event, callback) {
      if (listeners[event]) {
        listeners[event] = listeners[event].filter((cb) => cb !== callback)
      }
    },

    get url() {
      return socket?.url || null
    },
  }
}
