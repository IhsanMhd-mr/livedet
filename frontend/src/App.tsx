import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from '@/components/Navbar'
import Home from '@/pages/Home'
import ImageDetection from '@/pages/ImageDetection'
import VideoDetection from '@/pages/VideoDetection'
import LiveDetection from '@/pages/LiveDetection'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-surface-900 text-slate-100">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/image" element={<ImageDetection />} />
          <Route path="/video" element={<VideoDetection />} />
          <Route path="/live" element={<LiveDetection />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
