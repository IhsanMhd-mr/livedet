import React, { useState, useRef } from 'react'

export default function VideoDetect() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [frameInterval, setFrameInterval] = useState(30)
  const [detectionResults, setDetectionResults] = useState([])
  const [error, setError] = useState(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file)
      setError(null)
      console.log('[VideoDetect] Video selected:', file.name)

      // Load video
      const url = URL.createObjectURL(file)
      if (videoRef.current) {
        videoRef.current.src = url
      }
    } else {
      setError('Please select a valid video file')
    }
  }

  const handleProcessVideo = async () => {
    if (!selectedFile || !videoRef.current) {
      setError('Please select a video first')
      return
    }

    setIsProcessing(true)
    setError(null)
    setDetectionResults([])

    try {
      const video = videoRef.current
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      let frameCount = 0
      const results = []

      // Function to process frames
      const processFrame = async () => {
        if (video.paused) {
          setIsProcessing(false)
          console.log('[VideoDetect] Processing complete:', frameCount, 'frames analyzed')
          setDetectionResults(results)
          return
        }

        if (frameCount % frameInterval === 0) {
          // Draw frame to canvas
          ctx.drawImage(video, 0, 0)

          // Convert to blob and send to API
          canvas.toBlob(async (blob) => {
            try {
              const formData = new FormData()
              formData.append('image', blob)

              const response = await fetch('/api/detect', {
                method: 'POST',
                body: formData
              })

              if (response.ok) {
                const data = await response.json()
                results.push({
                  frame: frameCount,
                  detections: data.detections,
                  timestamp: data.timestamp
                })
                console.log(`[VideoDetect] Frame ${frameCount}: ${data.detections.length} potholes`)
              }
            } catch (err) {
              console.error('[VideoDetect] Frame processing error:', err)
            }
          })
        }

        frameCount++
        requestAnimationFrame(processFrame)
      }

      // Start video playback
      video.currentTime = 0
      video.play()
      processFrame()
    } catch (err) {
      console.error('[VideoDetect] Error:', err)
      setError(`Video processing failed: ${err.message}`)
      setIsProcessing(false)
    }
  }

  const handleClear = () => {
    setSelectedFile(null)
    setDetectionResults([])
    setError(null)
    if (videoRef.current) {
      videoRef.current.src = ''
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="detection-container">
      <h2 className="detection-title">Video Detection</h2>
      
      {error && <div className="message error">{error}</div>}

      <div className="video-container">
        {/* Video Input */}
        <div className="video-input">
          <h3>Upload Video</h3>
          <div 
            className="upload-area"
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              className="upload-input"
              accept="video/*"
              onChange={handleFileSelect}
            />
            <div className="upload-text">
              {selectedFile ? `Selected: ${selectedFile.name}` : 'Click to upload video'}
            </div>
            <div className="upload-subtext">
              Supported: MP4, WebM, AVI, MOV
            </div>
          </div>

          {selectedFile && (
            <>
              <video 
                ref={videoRef}
                className="video-element"
                controls
              />

              <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <label>Frame Interval:</label>
                <input
                  type="number"
                  className="control-input"
                  value={frameInterval}
                  onChange={(e) => setFrameInterval(Math.max(1, parseInt(e.target.value) || 30))}
                  min="1"
                  disabled={isProcessing}
                />
              </div>

              <div className="btn-group">
                <button 
                  className="btn btn-primary"
                  onClick={handleProcessVideo}
                  disabled={isProcessing}
                >
                  {isProcessing ? (
                    <>
                      <span className="spinner"></span> Processing...
                    </>
                  ) : (
                    'Process Video'
                  )}
                </button>
                <button 
                  className="btn btn-secondary"
                  onClick={handleClear}
                >
                  Clear
                </button>
              </div>
            </>
          )}
        </div>

        {/* Results */}
        <div className="video-input">
          <h3>Detection Results</h3>
          {detectionResults.length > 0 ? (
            <div style={{ 
              background: '#f0f4ff', 
              padding: '1rem', 
              borderRadius: '8px',
              maxHeight: '500px',
              overflowY: 'auto'
            }}>
              <p><strong>Total frames analyzed:</strong> {detectionResults.length}</p>
              <p><strong>Total potholes detected:</strong> {detectionResults.reduce((acc, r) => acc + r.detections.length, 0)}</p>
              
              <div style={{ marginTop: '1rem' }}>
                <strong>Frame-by-frame results:</strong>
                {detectionResults.map((result, idx) => (
                  <div key={idx} style={{ 
                    marginTop: '0.5rem', 
                    padding: '0.5rem', 
                    background: 'white',
                    borderRadius: '4px',
                    fontSize: '0.9rem'
                  }}>
                    Frame {result.frame}: {result.detections.length} detection(s)
                    {result.detections.length > 0 && (
                      <div style={{ marginLeft: '1rem', fontSize: '0.85rem', color: '#666' }}>
                        {result.detections.map((det, i) => {
                          const severityScore = det.severity_score !== undefined ? det.severity_score : (det.confidence ?? 0)
                          return (
                            <div key={i}>
                              • Severity: {(severityScore * 100).toFixed(1)}%
                            </div>
                          )
                        })}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div style={{ color: '#999' }}>
              No results yet. Upload a video and click "Process Video" to analyze.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
