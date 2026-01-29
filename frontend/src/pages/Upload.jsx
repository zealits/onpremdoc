import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadPDF, getDocument, vectorizeDocument, getConfidence } from '../services/api'
import '../styles/app.css'

function Upload() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [documentId, setDocumentId] = useState(null)
  const [currentStatus, setCurrentStatus] = useState(null)
  const [confidence, setConfidence] = useState(null)
  const [loadingConfidence, setLoadingConfidence] = useState(false)
  const [error, setError] = useState(null)
  const [vectorizationTriggered, setVectorizationTriggered] = useState(false)
  const navigate = useNavigate()
  const fileInputRef = useRef(null)

  useEffect(() => {
    if (documentId && currentStatus) {
      // Poll for status updates
      const interval = setInterval(async () => {
        try {
          const doc = await getDocument(documentId)
          setCurrentStatus(doc.status)

          // Auto-trigger vectorization when markdown is ready but vectorization hasn't started
          // Status "processing" means markdown exists, "vectorized" means vectorization files exist
          // We trigger when markdown exists but vectorization files don't
          if (
            doc.markdown_path &&
            !doc.vector_mapping_path &&
            !vectorizationTriggered &&
            doc.status !== 'ready'
          ) {
            setVectorizationTriggered(true)
            try {
              await vectorizeDocument(documentId)
            } catch (err) {
              setError(`Failed to start vectorization: ${err.message}`)
              setVectorizationTriggered(false) // Allow retry
            }
          }

          // Fetch confidence/accuracy once markdown is available
          if (doc.markdown_path && !confidence && !loadingConfidence) {
            setLoadingConfidence(true)
            try {
              const conf = await getConfidence(documentId)
              if (conf.has_confidence) {
                setConfidence(conf)
              }
            } catch (err) {
              // Don't treat as fatal; just log in console
              console.error('Failed to fetch confidence:', err)
            } finally {
              setLoadingConfidence(false)
            }
          }

          // Navigate to chat when ready
          if (doc.status === 'ready') {
            clearInterval(interval)
            setTimeout(() => {
              navigate(`/chat/${documentId}`)
            }, 1000)
          }
        } catch (err) {
          setError(`Failed to check status: ${err.message}`)
          clearInterval(interval)
        }
      }, 2000)

      return () => clearInterval(interval)
    }
  }, [documentId, currentStatus, vectorizationTriggered, navigate])

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      if (selectedFile.type !== 'application/pdf') {
        setError('Please select a PDF file')
        return
      }
      setFile(selectedFile)
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file')
      return
    }

    setUploading(true)
    setError(null)
    setVectorizationTriggered(false)

    try {
      const response = await uploadPDF(file)
      setDocumentId(response.document_id)
      setCurrentStatus(response.status || 'processing')
    } catch (err) {
      setError(err.message)
      setUploading(false)
    } finally {
      setUploading(false)
    }
  }

  const getStepStatus = (step) => {
    if (!documentId) return 'pending'

    if (step === 'upload') {
      return currentStatus ? 'completed' : 'pending'
    }

    if (step === 'markdown') {
      if (['vectorized', 'ready'].includes(currentStatus)) return 'completed'
      if (currentStatus === 'processing') return 'active'
      return 'pending'
    }

    if (step === 'vectorize') {
      if (currentStatus === 'ready') return 'completed'
      if (vectorizationTriggered || currentStatus === 'vectorized') return 'active'
      return 'pending'
    }

    if (step === 'ready') {
      return currentStatus === 'ready' ? 'completed' : 'pending'
    }

    return 'pending'
  }

  const getStepDescription = (step) => {
    if (step === 'upload') {
      return file ? `Uploading ${file.name}...` : 'Select and upload PDF file'
    }
    if (step === 'markdown') {
      if (currentStatus === 'processing') return 'Converting PDF to markdown...'
      if (['vectorized', 'ready'].includes(currentStatus)) return 'Markdown conversion complete'
      return 'Waiting for markdown conversion'
    }
    if (step === 'vectorize') {
      if (currentStatus === 'ready') return 'Vectorization complete'
      if (vectorizationTriggered || currentStatus === 'vectorized') return 'Vectorizing document...'
      return 'Waiting for vectorization'
    }
    if (step === 'ready') {
      return currentStatus === 'ready' ? 'Document ready for chat!' : 'Processing...'
    }
    return ''
  }

  const formatScore = (value) => {
    if (value == null || value === '') return '—'
    if (typeof value === 'number') {
      if (Number.isInteger(value)) return String(value)
      return value.toFixed(2)
    }
    return String(value)
  }

  const renderConfidenceTable = () => {
    if (!confidence || !confidence.has_confidence) return null

    const pageEntries = Object.entries(confidence.pages || {}).sort(
      (a, b) => Number(a[0]) - Number(b[0])
    )

    return (
      <div className="card" style={{ marginTop: '20px' }}>
        <h3 style={{ marginBottom: '10px' }}>Docling Conversion Accuracy</h3>
        <p style={{ fontSize: '13px', color: '#666', marginBottom: '12px' }}>
          Higher scores (closer to 1.0) indicate better layout/OCR quality. Empty values show as —.
        </p>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table confidence-table">
            <thead>
              <tr>
                <th>Scope</th>
                <th className="score-cell">Layout score</th>
                <th className="score-cell">OCR score</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Document</strong></td>
                <td className="score-cell">{formatScore(confidence.layout_score)}</td>
                <td className="score-cell">{formatScore(confidence.ocr_score)}</td>
              </tr>
              {pageEntries.map(([pageNo, scores]) => (
                <tr key={pageNo}>
                  <td>Page {Number(pageNo) + 1}</td>
                  <td className="score-cell">{formatScore(scores.layout_score)}</td>
                  <td className="score-cell">{formatScore(scores.ocr_score)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  return (
    <div>
      <div className="header">
        <div className="header-content">
          <h1>Upload PDF</h1>
          <button className="btn btn-secondary" onClick={() => navigate('/')}>
            Back to Home
          </button>
        </div>
      </div>

      <div className="container">
        {error && <div className="error">Error: {error}</div>}

        {!documentId ? (
          <div className="card">
            <h2 style={{ marginBottom: '20px' }}>Upload Document</h2>
            <div className="form-group">
              <label className="form-label">Select PDF File</label>
              <div className="file-input" onClick={() => fileInputRef.current?.click()}>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  onChange={handleFileChange}
                />
                {file ? (
                  <div>
                    <p style={{ marginBottom: '10px' }}>Selected: {file.name}</p>
                    <p style={{ fontSize: '12px', color: '#666' }}>
                      Size: {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                ) : (
                  <p>Click to select PDF file</p>
                )}
              </div>
            </div>
            <button
              className="btn btn-primary"
              onClick={handleUpload}
              disabled={!file || uploading}
            >
              {uploading ? 'Uploading...' : 'Upload and Process'}
            </button>
          </div>
        ) : (
          <div>
            <div className="card">
              <h2 style={{ marginBottom: '20px' }}>Processing Document</h2>
              <div className="progress-steps">
                <div className="progress-step">
                  <div className={`step-icon ${getStepStatus('upload')}`}>
                    {getStepStatus('upload') === 'completed' ? '✓' : '1'}
                  </div>
                  <div className="step-content">
                    <div className="step-title">Upload PDF</div>
                    <div className="step-description">{getStepDescription('upload')}</div>
                  </div>
                </div>

                <div className="progress-step">
                  <div className={`step-icon ${getStepStatus('markdown')}`}>
                    {getStepStatus('markdown') === 'completed' ? '✓' : '2'}
                  </div>
                  <div className="step-content">
                    <div className="step-title">Convert to Markdown</div>
                    <div className="step-description">{getStepDescription('markdown')}</div>
                  </div>
                </div>

                <div className="progress-step">
                  <div className={`step-icon ${getStepStatus('vectorize')}`}>
                    {getStepStatus('vectorize') === 'completed' ? '✓' : '3'}
                  </div>
                  <div className="step-content">
                    <div className="step-title">Vectorize Document</div>
                    <div className="step-description">{getStepDescription('vectorize')}</div>
                  </div>
                </div>

                <div className="progress-step">
                  <div className={`step-icon ${getStepStatus('ready')}`}>
                    {getStepStatus('ready') === 'completed' ? '✓' : '4'}
                  </div>
                  <div className="step-content">
                    <div className="step-title">Ready for Chat</div>
                    <div className="step-description">{getStepDescription('ready')}</div>
                  </div>
                </div>
              </div>

              <div style={{ marginTop: '20px', padding: '15px', background: '#f8f9fa', borderRadius: '6px' }}>
                <strong>Document ID:</strong> {documentId}
                <br />
                <strong>Status:</strong> <span className={`status-badge status-${currentStatus}`}>{currentStatus}</span>
              </div>

              {renderConfidenceTable()}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default Upload
