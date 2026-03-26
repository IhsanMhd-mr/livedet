import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

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
    proxy: {
      '/ws': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
      },
      '/predict': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/video': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          ui: ['framer-motion', 'react-hot-toast'],
        },
      },
    },
  },
})
