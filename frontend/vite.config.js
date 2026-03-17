import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    host: 'localhost',
    hmr: { protocol: 'ws', host: 'localhost', port: 5173 },
    proxy: {
      '/predict': { target: 'http://localhost:8000', changeOrigin: true, ws: true },
      '/video':   { target: 'http://localhost:8000', changeOrigin: true, ws: true },
      '/images':  { target: 'http://localhost:8000', changeOrigin: true, ws: true },
      '/report':  { target: 'http://localhost:8000', changeOrigin: true, ws: true },
      '/session': { target: 'http://localhost:8000', changeOrigin: true, ws: true },
      '/stats':   { target: 'http://localhost:8000', changeOrigin: true, ws: true },
      '/cleanup': { target: 'http://localhost:8000', changeOrigin: true, ws: true },
      '/health':  { target: 'http://localhost:8000', changeOrigin: true, ws: true },
      '/ws':      { target: 'ws://localhost:8000',   changeOrigin: true, ws: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          motion: ['framer-motion'],
          state:  ['zustand'],
        },
      },
    },
  },
})
