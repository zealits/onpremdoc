import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { listDocuments } from '../services/api'
import '../styles/app.css'

function Home() {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    fetchDocuments()
    // Poll every 5 seconds for status updates
    const interval = setInterval(fetchDocuments, 5000)
    return () => clearInterval(interval)
  }, [])

  const fetchDocuments = async () => {
    try {
      const data = await listDocuments()
      setDocuments(data)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleDocumentClick = (documentId, status) => {
    if (status === 'ready') {
      navigate(`/chat/${documentId}`)
    }
  }

  const getStatusClass = (status) => {
    return `status-badge status-${status}`
  }

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A'
    try {
      return new Date(dateString).toLocaleDateString()
    } catch {
      return 'N/A'
    }
  }

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <p>Loading documents...</p>
      </div>
    )
  }

  return (
    <div>
      <div className="header">
        <div className="header-content">
          <h1>Document Processing</h1>
          <button className="btn btn-primary" onClick={() => navigate('/upload')}>
            Upload New PDF
          </button>
        </div>
      </div>

      <div className="container">
        {error && <div className="error">Error: {error}</div>}

        {documents.length === 0 ? (
          <div className="empty-state">
            <h2>No documents found</h2>
            <p>Upload a PDF to get started</p>
            <button className="btn btn-primary" onClick={() => navigate('/upload')} style={{ marginTop: '20px' }}>
              Upload PDF
            </button>
          </div>
        ) : (
          <div>
            <h2 style={{ marginBottom: '20px' }}>Documents ({documents.length})</h2>
            {documents.map((doc) => (
              <div
                key={doc.document_id}
                className="card"
                style={{
                  cursor: doc.status === 'ready' ? 'pointer' : 'default',
                  opacity: doc.status === 'ready' ? 1 : 0.8,
                }}
                onClick={() => handleDocumentClick(doc.document_id, doc.status)}
              >
                <div className="card-header">
                  <div className="card-title">{doc.name || doc.document_id}</div>
                  <span className={getStatusClass(doc.status)}>{doc.status}</span>
                </div>
                <div className="metadata">
                  <span className="metadata-item">
                    <strong>ID:</strong> {doc.document_id}
                  </span>
                  {doc.total_pages && (
                    <span className="metadata-item">
                      <strong>Pages:</strong> {doc.total_pages}
                    </span>
                  )}
                  {doc.total_chunks && (
                    <span className="metadata-item">
                      <strong>Chunks:</strong> {doc.total_chunks}
                    </span>
                  )}
                </div>
                {doc.status === 'ready' && (
                  <div style={{ marginTop: '10px' }}>
                    <span style={{ fontSize: '12px', color: '#666' }}>
                      Click to open chat →
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default Home
