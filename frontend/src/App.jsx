import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import ErrorBoundary from '@/components/ErrorBoundary'
import Layout from '@/components/layout/Layout'
import Home from '@/pages/Home'
import ImageDetection from '@/pages/ImageDetection'
import VideoDetection from '@/pages/VideoDetection'
import LiveDetection from '@/pages/LiveDetection'
import Demo from '@/pages/Demo'

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/"      element={<Home />} />
            <Route path="/image" element={<ImageDetection />} />
            <Route path="/video" element={<VideoDetection />} />
            <Route path="/live"  element={<LiveDetection />} />
            <Route path="/demo"  element={<Demo />} />
          </Route>
        </Routes>
      </BrowserRouter>

      <Toaster
        position="bottom-right"
        toastOptions={{
          style: {
            background: '#1e293b',
            color: '#f1f5f9',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: '12px',
            fontSize: '13px',
          },
          success: { iconTheme: { primary: '#10b981', secondary: '#f1f5f9' } },
          error:   { iconTheme: { primary: '#ef4444', secondary: '#f1f5f9' } },
        }}
      />
    </ErrorBoundary>
  )
}
