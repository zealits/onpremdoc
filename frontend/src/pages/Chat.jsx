import { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import { getDocument, queryDocument } from '../services/api'
import '../styles/app.css'

function Chat() {
  const { documentId } = useParams()
  const navigate = useNavigate()
  const [document, setDocument] = useState(null)
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    fetchDocument()
  }, [documentId])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const fetchDocument = async () => {
    try {
      const doc = await getDocument(documentId)
      setDocument(doc)
      if (doc.status !== 'ready') {
        setError('Document is not ready for chat. Please wait for processing to complete.')
      }
    } catch (err) {
      setError(`Failed to load document: ${err.message}`)
    }
  }

  const handleSend = async () => {
    if (!inputValue.trim() || loading) return

    const userMessage = inputValue.trim()
    setInputValue('')
    setLoading(true)
    setError(null)

    // Add user message
    const newMessages = [...messages, { role: 'user', content: userMessage }]
    setMessages(newMessages)

    try {
      const response = await queryDocument(documentId, userMessage)
      
      // Add assistant response
      setMessages([
        ...newMessages,
        {
          role: 'assistant',
          content: response.answer,
          retrievalStats: response.retrieval_stats,
        },
      ])
    } catch (err) {
      setError(err.message)
      // Remove the user message if query failed
      setMessages(messages)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  if (!document) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <p>Loading document...</p>
      </div>
    )
  }

  if (document.status !== 'ready') {
    return (
      <div className="container">
        <div className="error">
          Document is not ready for chat. Status: {document.status}
          <br />
          <button className="btn btn-secondary" onClick={() => navigate('/')} style={{ marginTop: '10px' }}>
            Back to Home
          </button>
        </div>
      </div>
    )
  }

  return (
    <div>
      <div className="header">
        <div className="header-content">
          <h1>{document.name || document.document_id}</h1>
          <button className="btn btn-secondary" onClick={() => navigate('/')}>
            Back to Home
          </button>
        </div>
      </div>

      <div className="chat-container">
        <div className="chat-header">
          <div>
            <strong>Document:</strong> {document.name || document.document_id}
          </div>
          {document.total_pages && (
            <div className="metadata" style={{ marginTop: '8px' }}>
              <span className="metadata-item">Pages: {document.total_pages}</span>
              {document.total_chunks && (
                <span className="metadata-item">Chunks: {document.total_chunks}</span>
              )}
            </div>
          )}
        </div>

        <div className="chat-messages">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Start a conversation</h2>
              <p>Ask questions about the document</p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div key={index} className={`message message-${msg.role}`}>
                <div className="message-content">
                  {msg.role === 'assistant' ? (
                    <ReactMarkdown
                      components={{
                        h1: ({ children }) => <h1 style={{ margin: '16px 0 10px 0', fontSize: '1.75em', fontWeight: 700, lineHeight: 1.3, color: '#1a1a1a', borderBottom: '2px solid #e0e0e0', paddingBottom: '8px' }}>{children}</h1>,
                        h2: ({ children }) => <h2 style={{ margin: '20px 0 10px 0', fontSize: '1.5em', fontWeight: 700, lineHeight: 1.3, color: '#1a1a1a' }}>{children}</h2>,
                        h3: ({ children }) => <h3 style={{ margin: '16px 0 8px 0', fontSize: '1.25em', fontWeight: 700, lineHeight: 1.3, color: '#1a1a1a' }}>{children}</h3>,
                        h4: ({ children }) => <h4 style={{ margin: '14px 0 8px 0', fontSize: '1.1em', fontWeight: 700, lineHeight: 1.3, color: '#1a1a1a' }}>{children}</h4>,
                        h5: ({ children }) => <h5 style={{ margin: '12px 0 6px 0', fontSize: '1em', fontWeight: 700, lineHeight: 1.3, color: '#1a1a1a' }}>{children}</h5>,
                        h6: ({ children }) => <h6 style={{ margin: '10px 0 6px 0', fontSize: '0.95em', fontWeight: 700, lineHeight: 1.3, color: '#1a1a1a' }}>{children}</h6>,
                        p: ({ children }) => <p style={{ margin: '10px 0', lineHeight: 1.7, color: '#333' }}>{children}</p>,
                        ul: ({ children }) => <ul style={{ margin: '12px 0', paddingLeft: '28px', lineHeight: 1.7, listStyleType: 'disc' }}>{children}</ul>,
                        ol: ({ children }) => <ol style={{ margin: '12px 0', paddingLeft: '28px', lineHeight: 1.7, listStyleType: 'decimal' }}>{children}</ol>,
                        li: ({ children }) => <li style={{ margin: '6px 0', lineHeight: 1.6, color: '#333' }}>{children}</li>,
                        strong: ({ children }) => <strong style={{ fontWeight: 700, color: '#1a1a1a' }}>{children}</strong>,
                        em: ({ children }) => <em style={{ fontStyle: 'italic', color: '#555' }}>{children}</em>,
                        code: ({ children, className }) => {
                          const isInline = !className;
                          if (isInline) {
                            return <code style={{ backgroundColor: 'rgba(0, 0, 0, 0.05)', padding: '3px 6px', borderRadius: '4px', fontFamily: "'Consolas', 'Monaco', 'Courier New', monospace", fontSize: '0.9em', color: '#d63384' }}>{children}</code>;
                          }
                          return <code>{children}</code>;
                        },
                        pre: ({ children }) => <pre style={{ backgroundColor: '#f8f9fa', padding: '14px', borderRadius: '6px', overflowX: 'auto', margin: '12px 0', border: '1px solid #e9ecef' }}>{children}</pre>,
                        blockquote: ({ children }) => <blockquote style={{ borderLeft: '4px solid #007bff', paddingLeft: '16px', margin: '12px 0', color: '#555', fontStyle: 'italic', backgroundColor: '#f8f9fa', padding: '12px 16px', borderRadius: '4px' }}>{children}</blockquote>,
                        hr: () => <hr style={{ border: 'none', borderTop: '1px solid #e0e0e0', margin: '16px 0' }} />,
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                  ) : (
                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                  )}
                </div>
                {msg.retrievalStats && (
                  <div className="metadata" style={{ marginTop: '12px', fontSize: '11px', opacity: 0.7 }}>
                    Chunks used: {msg.retrievalStats.total_chunks_used || 0}
                    {msg.retrievalStats.iterations > 0 && (
                      <span style={{ marginLeft: '10px' }}>
                        Iterations: {msg.retrievalStats.iterations}
                      </span>
                    )}
                  </div>
                )}
              </div>
            ))
          )}
          {loading && (
            <div className="message message-assistant">
              <div className="message-content">
                <div className="spinner" style={{ width: '20px', height: '20px', margin: '0' }}></div>
                Thinking...
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {error && (
          <div style={{ padding: '0 20px' }}>
            <div className="error">{error}</div>
          </div>
        )}

        <div className="chat-input-container">
          <textarea
            className="chat-input"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about the document..."
            rows={2}
            disabled={loading}
          />
          <button
            className="btn btn-primary"
            onClick={handleSend}
            disabled={!inputValue.trim() || loading}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  )
}

export default Chat
