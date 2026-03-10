import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: 'localhost',
    // HMR configuration for proper hot module replacement
    hmr: {
      protocol: 'ws',
      host: 'localhost',
      port: 5173
    },
    // Proxy configuration to forward API calls to backend
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        ws: true
      },
      '/predict': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true
      },
      '/images': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true
      },
      '/report': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true
      },
      '/session': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true
      },
      '/stats': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true
      },
      '/cleanup': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true
      }
    }
  }
})
