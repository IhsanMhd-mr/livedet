import React, { useState, useRef } from 'react'
import { API_BASE_URL, PATHS, apiClient, createLogger } from '../config'

const logger = createLogger('ImageDetect')

export default function ImageDetect() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [processedImageUrl, setProcessedImageUrl] = useState(null)
  const [reportData, setReportData] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)

  const handleFileSelect = (e) => {
    console.log('[FileSelect] User selected file')
    const file = e.target.files?.[0]
    if (file) {
      console.log('[FileSelect] File details:', {
        name: file.name,
        size: file.size,
        type: file.type,
        lastModified: new Date(file.lastModified).toISOString()
      })
      setSelectedFile(file)
      setError(null)

      // Create preview
      const reader = new FileReader()
      reader.onload = (event) => {
        console.log('[FileSelect] Preview created')
        setPreview(event.target.result)
      }
      reader.readAsDataURL(file)
    } else {
      console.log('[FileSelect] No file selected')
    }
  }

  const handleDetect = async () => {
    console.log('═'.repeat(80))
    console.log('[FLOW START] User clicked "Detect Potholes"')
    console.log('═'.repeat(80))
    
    if (!selectedFile) {
      console.log('[ERROR] No file selected')
      setError('Please select an image first')
      return
    }

    console.log('[STEP 1] File selected:', {
      name: selectedFile.name,
      size: selectedFile.size,
      type: selectedFile.type
    })

    setLoading(true)
    setError(null)

    try {
      console.log('[STEP 2] Creating FormData and sending to backend...')
      console.log('[API] URL:', `${API_BASE_URL}/predict`)
      
      // Create FormData with the image file
      const formData = new FormData()
      formData.append('image', selectedFile)
      console.log('[STEP 2] FormData created with file:', selectedFile.name)

      // Call backend prediction API
      const startTime = performance.now()
      console.log('[STEP 3] Sending POST request to backend...')
      
      const response = await fetch(PATHS.predict, {
        method: 'POST',
        body: formData,
      })

      const endTime = performance.now()
      console.log(`[STEP 4] Response received in ${(endTime - startTime).toFixed(2)}ms`)
      console.log('[STEP 4] Response status:', response.status, response.statusText)

      if (!response.ok) {
        console.error('[ERROR] Response not OK')
        const errorData = await response.json().catch(() => ({}))
        console.error('[ERROR] Error details:', errorData)
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      console.log('[STEP 5] JSON parsed successfully')
      console.log('[STEP 5] Response structure check:')
      console.log('  ├─ result_id:', typeof data.result_id, '=', data.result_id)
      console.log('  ├─ status:', typeof data.status, '=', data.status)
      console.log('  ├─ num_potholes:', typeof data.num_potholes, '=', data.num_potholes)
      console.log('  ├─ average_depth_cm:', typeof data.average_depth_cm, '=', data.average_depth_cm)
      console.log('  ├─ average_width_cm:', typeof data.average_width_cm, '=', data.average_width_cm)
      console.log('  ├─ image:', data.image ? 'base64 string (~' + Math.round(data.image.length / 1024) + 'KB)' : 'missing')
      console.log('  ├─ image_shape:', Array.isArray(data.image_shape) ? data.image_shape : 'invalid')
      console.log('  ├─ detections:', data.detections ? data.detections.length + ' items' : '0 items')
      console.log('  └─ timestamp:', data.timestamp)
      
      // Log individual detections with depth/width
      if (data.detections && data.detections.length > 0) {
        console.log('[STEP 5] Individual Detection Details:')
        data.detections.forEach((det, idx) => {
          console.log(`  [Pothole #${det.id || idx + 1}]`)
          console.log(`    ├─ Confidence: ${(det.confidence * 100).toFixed(1)}%`)
          console.log(`    ├─ Depth: ${det.depth_cm || 'N/A'} cm`)
          console.log(`    ├─ Width: ${det.width_cm || 'N/A'} cm`)
          console.log(`    ├─ Severity: ${det.severity ?? '—'}`)
          console.log(`    └─ BBox: ${det.bbox}`)
        })
      }
      
      // Validate response
      const isValidResponse = data.status === 'success' && data.result_id && data.image
      console.log('[STEP 5] Response validation:', isValidResponse ? '✓ Valid' : '✗ Invalid')

      // Generate result ID
      const resultId = data.result_id || `result_${Date.now()}`
      setSessionId(resultId)
      setResult(data)
      console.log('[STEP 6] State updated with result_id:', resultId)
      console.log('[STEP 6] State updated with depth/width data')

      // Display the result image (already in base64 format from backend)
      if (data.image) {
        console.log('[STEP 7] Converting base64 image to data URL...')
        console.log('[STEP 7] Image base64 length:', data.image.length, 'characters')
        console.log('[STEP 7] Image base64 preview:', data.image.substring(0, 50))
        const imageDataUrl = `data:image/jpeg;base64,${data.image}`
        console.log('[STEP 7] Image data URL length:', imageDataUrl.length, 'characters')
        console.log('[STEP 7] Image data URL preview:', imageDataUrl.substring(0, 80))
        setProcessedImageUrl(imageDataUrl)
        console.log('[STEP 7] ✓ setProcessedImageUrl called with data URL')
        console.log('[STEP 7] Image shape:', data.image_shape)
      } else {
        console.warn('[WARNING] No image in response')
        console.warn('[WARNING] Response keys:', Object.keys(data))
      }

      // Process detection data
      if (data.detections && Array.isArray(data.detections)) {
        console.log('[STEP 8] Processing', data.detections.length, 'detections...')
        
        // Create a simple report-like structure from detections
        const report = {
          result_id: resultId,
          timestamp: data.timestamp,
          image_shape: data.image_shape,
          statistics: {
            total_detections: data.num_potholes,
            detections: data.detections
          },
          detections: data.detections
        }
        
        console.log('[STEP 8] Report structure created')
        console.log('[STEP 8] Total detections:', data.num_potholes)
        
        // Log each detection
        data.detections.forEach((det, idx) => {
          console.log(`[STEP 8] Detection #${idx + 1}: bbox=${JSON.stringify(det.bbox || det.box)}, conf=${det.confidence?.toFixed(3)}`)
        })
        
        setReportData(report)
        console.log('[STEP 8] Report data loaded successfully')
      } else {
        console.log('[STEP 8] No detections in response')
        const emptyReport = {
          result_id: resultId,
          timestamp: data.timestamp,
          image_shape: data.image_shape,
          statistics: {
            total_detections: 0,
            detections: []
          },
          detections: []
        }
        setReportData(emptyReport)
      }

      console.log('═'.repeat(80))
      console.log('[FLOW COMPLETE] Detection finished successfully')
      console.log('═'.repeat(80))
    } catch (err) {
      console.error('═'.repeat(80))
      console.error('[FLOW ERROR] Exception occurred:', err)
      console.error('[ERROR] Stack trace:', err.stack)
      console.error('═'.repeat(80))
      setError(`Detection failed: ${err.message}`)
    } finally {
      setLoading(false)
      console.log('[STEP 10] Loading state set to false')
    }
  }

  const downloadImage = (url, filename) => {
    console.log('[Download] Downloading image:', filename, 'from:', url)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    console.log('[Download] Triggering download...')
    link.click()
    document.body.removeChild(link)
    console.log('[Download] Download initiated')
  }

  const downloadReport = () => {
    console.log('[Download] User requested JSON report download')
    if (!reportData) {
      console.log('[Download] No report data available')
      return
    }
    console.log('[Download] Report data:', reportData)
    const dataStr = JSON.stringify(reportData, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    console.log('[Download] Created object URL:', url)
    downloadImage(url, `livedet_report_${sessionId}.json`)
    URL.revokeObjectURL(url)
    console.log('[Download] Released object URL')
  }

  const handleClear = () => {
    console.log('[Clear] User clicked clear button')
    setSelectedFile(null)
    setPreview(null)
    setResult(null)
    setProcessedImageUrl(null)
    setReportData(null)
    setSessionId(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
    console.log('[Clear] All state cleared')
  }

  return (
    <div className="detection-container">
      <h2 className="detection-title">Image Detection</h2>
      
      {error && <div className="message error">{error}</div>}
      {result && !error && (
        <div className={`message ${(result?.count ?? result?.num_potholes ?? 0) === 0 ? 'info' : 'success'}`} style={(result?.count ?? result?.num_potholes ?? 0) === 0 ? { backgroundColor: '#d4edda', borderColor: '#c3e6cb', color: '#155724' } : {}}>
          ✓ Detection Complete — {(result?.count ?? result?.num_potholes ?? 0) === 0 ? (
            <strong>Clean road! No potholes found. 🎉</strong>
          ) : (
            <><strong>{result?.count ?? result?.num_potholes ?? 0}</strong> object(s) detected</>
          )} &nbsp;|&nbsp; Session: <code style={{fontSize:'0.9em', backgroundColor:'#fff', padding:'2px 6px', borderRadius:'3px'}}>{sessionId}</code>
        </div>
      )}

      <div className="detection-result">
        {/* Upload Area */}
        <div 
          className={`upload-area ${selectedFile ? 'active' : ''}`}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="upload-input"
            accept="image/*"
            onChange={handleFileSelect}
          />
          <div className="upload-text">
            {selectedFile ? `Selected: ${selectedFile.name}` : 'Click or drag to upload image'}
          </div>
          <div className="upload-subtext">
            Supported: JPG, PNG, BMP, WebP
          </div>
        </div>

        {/* Two Column Layout */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '2rem' }}>
          {/* Original Image */}
          {preview && (
            <div style={{ textAlign: 'center' }}>
              <h3>📷 Original Image</h3>
              <img 
                src={preview} 
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '500px',
                  border: '2px solid #ddd',
                  borderRadius: '8px'
                }}
                alt="Original"
              />
            </div>
          )}

          {/* Processed Image with Boxes */}
          {processedImageUrl && (
            <div style={{ textAlign: 'center' }}>
              <h3>🎯 Detection Result</h3>
              <img 
                src={processedImageUrl} 
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '500px',
                  border: '2px solid #f44336',
                  borderRadius: '8px'
                }}
                alt="Processed"
                onLoad={(e) => {
                  console.log('[IMAGE] ✓ Image loaded successfully!')
                  console.log('[IMAGE] Image dimensions:', e.target.naturalWidth, 'x', e.target.naturalHeight)
                }}
                onError={(e) => {
                  console.error('[IMAGE ERROR] Failed to load processed image')
                  console.error('[IMAGE ERROR] URL:', processedImageUrl)
                  console.error('[IMAGE ERROR] URL length:', processedImageUrl.length)
                  console.error('[IMAGE ERROR] URL preview:', processedImageUrl.substring(0, 100))
                }}
              />
            </div>
          )}
          {!processedImageUrl && result && (
            <div style={{ 
              padding: '2rem', 
              backgroundColor: '#fff3cd', 
              borderRadius: '8px', 
              color: '#856404',
              textAlign: 'center'
            }}>
              <p>Image URL not set. Check console for details.</p>
              <p>processedImageUrl value: {typeof processedImageUrl} = {processedImageUrl}</p>
            </div>
          )}
        </div>

        {/* Detection Analysis */}
        {reportData && (
          <div style={{ marginTop: '2rem' }}>
            <h3>Analysis Results</h3>
            <div className="detection-stats">
              <div className="stat-box">
                <div className="stat-label">Total</div>
                <div className="stat-value">{result?.count ?? result?.num_potholes ?? 0}</div>
              </div>
              <div className="stat-box">
                <div className="stat-label">Avg Depth</div>
                <div className="stat-value" style={{color:'#ff6b6b'}}>{result?.summary?.avg_depth_cm ?? result?.average_depth_cm ?? 0} cm</div>
              </div>
              <div className="stat-box">
                <div className="stat-label">Avg Width</div>
                <div className="stat-value" style={{color:'#4ecdc4'}}>{result?.summary?.avg_width_cm ?? result?.average_width_cm ?? 0} cm</div>
              </div>
              <div className="stat-box">
                <div className="stat-label">Max Depth</div>
                <div className="stat-value" style={{color:'#ef5350'}}>{result?.summary?.max_depth_cm ?? result?.max_depth_cm ?? 0} cm</div>
              </div>
              <div className="stat-box">
                <div className="stat-label">Critical</div>
                <div className="stat-value" style={{color:'#ff1744'}}>{result?.summary?.severity_counts?.Critical ?? 0}</div>
              </div>
              <div className="stat-box">
                <div className="stat-label">High</div>
                <div className="stat-value" style={{color:'#ff6d00'}}>{result?.summary?.severity_counts?.High ?? 0}</div>
              </div>
              <div className="stat-box">
                <div className="stat-label">Model</div>
                <div className="stat-value" style={{fontSize:'0.75rem', fontFamily:'monospace'}}>{result?.model || '—'}</div>
              </div>
              <div className="stat-box">
                <div className="stat-label">Time</div>
                <div className="stat-value" style={{fontSize:'0.85rem'}}>{new Date(result?.timestamp).toLocaleTimeString()}</div>
              </div>
            </div>

            {/* Individual Detections Table */}
            {result?.detections && result.detections.length > 0 && (
              <div style={{ marginTop: '2rem' }}>
                <h4>Detection Details</h4>
                <div style={{ overflowX: 'auto', marginTop: '1rem' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem', backgroundColor: '#f9f9f9' }}>
                    <thead>
                      <tr style={{ backgroundColor: '#2c3e50', color: 'white', borderBottom: '2px solid #ddd' }}>
                        {['#', 'Class', 'Confidence', 'Severity', 'Depth (cm)', 'Width (cm)', 'Area (px²)'].map(h => (
                          <th key={h} style={{ padding: '0.65rem 0.75rem', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.detections.map((det, idx) => {
                        const SEV_COLOR = { Low: '#388e3c', Medium: '#f57c00', High: '#e65100', Critical: '#d32f2f' }
                        const SEV_BG    = { Low: '#e8f5e9', Medium: '#fff3e0', High: '#fbe9e7', Critical: '#ffebee' }
                        const depthColor = det.depth_cm > 60 ? '#d32f2f' : det.depth_cm > 30 ? '#f57c00' : '#388e3c'
                        const widthColor = det.width_cm > 30 ? '#d32f2f' : det.width_cm > 20 ? '#f57c00' : '#388e3c'
                        return (
                          <tr key={idx} style={{ borderBottom: '1px solid #eee', backgroundColor: idx % 2 === 0 ? '#fff' : '#f5f5f5' }}>
                            <td style={{ padding: '0.65rem 0.75rem', fontWeight: 'bold' }}>#{det.id || idx + 1}</td>
                            <td style={{ padding: '0.65rem 0.75rem', color: '#1565c0', fontWeight: 600 }}>{det.class_name || '—'}</td>
                            <td style={{ padding: '0.65rem 0.75rem' }}>{(det.confidence * 100).toFixed(1)}%</td>
                            <td style={{ padding: '0.65rem 0.75rem' }}>
                              <span style={{
                                padding: '0.2rem 0.6rem', borderRadius: '4px', fontWeight: 'bold',
                                color: SEV_COLOR[det.severity] || '#555',
                                backgroundColor: SEV_BG[det.severity] || '#eee'
                              }}>
                                {det.severity || (det.severity_score != null ? det.severity_score.toFixed(2) : '—')}
                              </span>
                            </td>
                            <td style={{ padding: '0.65rem 0.75rem', fontWeight: 'bold', color: depthColor }}>{det.depth_cm?.toFixed(1) ?? '—'}</td>
                            <td style={{ padding: '0.65rem 0.75rem', fontWeight: 'bold', color: widthColor }}>{det.width_cm?.toFixed(1) ?? '—'}</td>
                            <td style={{ padding: '0.65rem 0.75rem' }}>{det.area ?? 0}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Buttons */}
      <div className="btn-group" style={{ marginTop: '2rem', display: 'flex', gap: '1rem' }}>
        <button 
          className="btn btn-primary"
          onClick={handleDetect}
          disabled={!selectedFile || loading}
          style={{ flex: 1 }}
        >
          {loading ? (
            <>
              <span className="spinner"></span> Detecting...
            </>
          ) : (
            '� Analyze Image'
          )}
        </button>
        {processedImageUrl && (
          <button 
            className="btn btn-info"
            onClick={() => downloadImage(processedImageUrl, `livedet_detected_${sessionId}.jpg`)}
            style={{ flex: 1 }}
          >
            💾 Download Image
          </button>
        )}
        {reportData && (
          <button 
            className="btn btn-info"
            onClick={downloadReport}
            style={{ flex: 1 }}
          >
            📄 Download Report
          </button>
        )}
        <button 
          className="btn btn-secondary"
          onClick={handleClear}
          style={{ flex: 1 }}
        >
          🔄 Clear
        </button>
      </div>
    </div>
  )
}
