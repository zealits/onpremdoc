import { useEffect, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useDocument, useVectorize } from '../api/hooks'
import MarkdownViewer from '../components/MarkdownViewer'
import ChatPanel from '../components/ChatPanel'

export default function DocumentPage() {
  const { documentId } = useParams()
  const { data: doc, isLoading, error } = useDocument(documentId)
  const vectorizeMutation = useVectorize(documentId)
  const vectorizeTriggered = useRef(false)
  const [activeHighlight, setActiveHighlight] = useState(null)

  useEffect(() => {
    vectorizeTriggered.current = false
  }, [documentId])

  useEffect(() => {
    if (!doc || vectorizeTriggered.current) return
    if (doc.status === 'processing' && doc.markdown_path) {
      vectorizeTriggered.current = true
      vectorizeMutation.mutate()
    }
  }, [doc?.status, doc?.markdown_path])

  const ready = doc?.status === 'ready'

  if (isLoading && !doc) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-4 bg-gradient-to-br from-slate-50 via-indigo-50/20 to-slate-50">
        <div className="w-12 h-12 rounded-xl border-2 border-indigo-200 border-t-indigo-500 animate-spin" />
        <p className="text-gray-600 font-medium animate-pulse">Loading document…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center text-red-600" role="alert">
        {error?.message || 'Document not found'}
      </div>
    )
  }

  if (!documentId) return null

  return (
    <div className="flex-1 flex flex-col h-full">
      <header className="shrink-0 border-b bg-white px-4 py-2 text-gray-700 font-medium flex items-center gap-2">
        <span className="truncate">{doc?.name || documentId}</span>
        {doc?.status && (
          <span
            className={`text-xs px-2 py-0.5 rounded ${
              ready ? 'bg-green-100 text-green-800' : 'bg-amber-100 text-amber-800'
            }`}
          >
            {doc.status}
          </span>
        )}
      </header>
      <div className="flex-1 grid grid-cols-2 gap-0 min-h-0">
        <div className="min-w-0 min-h-0 flex flex-col overflow-hidden">
          <MarkdownViewer
            documentId={ready ? documentId : null}
            activeHighlight={activeHighlight}
          />
          {!ready && (
            <div className="flex-1 flex items-center justify-center bg-gradient-to-br from-slate-50 via-indigo-50/30 to-slate-50 p-6 min-h-0">
              <div className="animate-processing-fade-in max-w-sm w-full text-center">
                <div className="relative inline-flex justify-center mb-6">
                  <div className="absolute inset-0 rounded-full bg-indigo-200/50 animate-ping" style={{ animationDuration: '2s' }} />
                  <div className="relative w-16 h-16 rounded-2xl bg-white border border-indigo-100 shadow-lg animate-processing-float flex items-center justify-center animate-processing-glow">
                    <svg className="w-8 h-8 text-indigo-500 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5v-7.5H8.25v7.5z" />
                    </svg>
                  </div>
                </div>
                <h3 className="text-gray-700 font-semibold text-lg mb-1">
                  {doc?.status === 'uploaded' && 'Waiting for processing'}
                  {(doc?.status === 'processing' || doc?.status === 'vectorized') && 'Preparing your document'}
                </h3>
                <p className="text-gray-500 text-sm mb-6">
                  {doc?.status === 'uploaded' && 'Your file is in the queue. We’ll start shortly.'}
                  {(doc?.status === 'processing' || doc?.status === 'vectorized') && 'Markdown will appear here when ready.'}
                </p>
                <div className="flex justify-center gap-2 mb-4">
                  {['uploaded', 'processing', 'vectorized', 'ready'].map((step, i) => {
                    const isActive = step === doc?.status
                    const isPast = (step === 'uploaded' && (doc?.status === 'processing' || doc?.status === 'vectorized' || doc?.status === 'ready')) ||
                      (step === 'processing' && (doc?.status === 'vectorized' || doc?.status === 'ready')) ||
                      (step === 'vectorized' && doc?.status === 'ready')
                    return (
                      <div
                        key={step}
                        className={`h-1.5 flex-1 max-w-12 rounded-full transition-all duration-500 ${
                          isPast ? 'bg-indigo-500' : isActive ? 'animate-processing-shimmer bg-indigo-200' : 'bg-gray-200'
                        }`}
                        title={step}
                      />
                    )
                  })}
                </div>
                <div className="h-1 w-full rounded-full bg-gray-200 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-indigo-400 transition-all duration-700 ease-out"
                    style={{
                      width: doc?.status === 'uploaded' ? '25%' : doc?.status === 'processing' || doc?.status === 'vectorized' ? '75%' : '100%',
                    }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
        <div className="min-w-0 min-h-0 flex flex-col overflow-hidden">
          <ChatPanel
            documentId={documentId}
            documentReady={ready}
            onHighlightChunk={(chunk) => setActiveHighlight(chunk)}
          />
        </div>
      </div>
    </div>
  )
}
