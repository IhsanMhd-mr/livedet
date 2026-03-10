import React, { useState } from 'react'
import ImageDetect from './components/ImageDetect'
import VideoDetect from './components/VideoDetect'
import LiveDetect from './components/LiveDetect'

export default function App() {
  const [mode, setMode] = useState('image')

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>LIVEDET — Road Defect Detection</h1>
        <p>Real-time detection powered by YOLOv8s (finetuned)</p>
      </header>

      <nav className="mode-selector">
        <button 
          className={`mode-btn ${mode === 'image' ? 'active' : ''}`}
          onClick={() => setMode('image')}
        >
          Image Detection
        </button>
        <button 
          className={`mode-btn ${mode === 'video' ? 'active' : ''}`}
          onClick={() => setMode('video')}
        >
          Video Detection
        </button>
        <button 
          className={`mode-btn ${mode === 'live' ? 'active' : ''}`}
          onClick={() => setMode('live')}
        >
          Live Camera
        </button>
      </nav>

      <main className="app-main">
        {mode === 'image' && <ImageDetect />}
        {mode === 'video' && <VideoDetect />}
        {mode === 'live' && <LiveDetect />}
      </main>

      <footer className="app-footer">
        <p>Final Year Project | ML-powered Infrastructure Monitoring</p>
        <p>Model: YOLOv8s (Finetuned) | Framework: PyTorch | API: Flask + WebSocket</p>
      </footer>
    </div>
  )
}
