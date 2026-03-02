import { useEffect, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useDocument, useVectorize } from '../api/hooks'
import MarkdownViewer from '../components/MarkdownViewer'
import ChatPanel from '../components/ChatPanel'

function getDocumentDisplayName(doc, documentId) {
  if (!doc) return documentId || ''
  if (doc.name && doc.name !== documentId) return doc.name
  const path =
    doc.markdown_path ||
    doc.page_mapping_path ||
    doc.confidence_path ||
    ''
  if (path) {
    const parts = String(path).split(/[\\/]/)
    const file = parts[parts.length - 1] || ''
    const withoutExt = file.replace(/\.(md|pdf)$/i, '')
    if (withoutExt) return withoutExt
  }
  return documentId || ''
}

export default function DocumentPage() {
  const { documentId } = useParams()
  const { data: doc, isLoading, error } = useDocument(documentId)
  const vectorizeMutation = useVectorize(documentId)
  const vectorizeTriggered = useRef(false)
  const [activeHighlight, setActiveHighlight] = useState(null)
  const [isMarkdownOpen, setIsMarkdownOpen] = useState(false)

  useEffect(() => {
    vectorizeTriggered.current = false
    setActiveHighlight(null)
    setIsMarkdownOpen(false)
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
    <div className="flex-1 flex flex-col h-full min-h-0">
      <header className="doc-header shrink-0 border-b border-slate-800 bg-slate-900/70 backdrop-blur flex items-center justify-between gap-3 px-5 py-3">
        <div className="min-w-0">
          <div className="font-medium truncate text-sm sm:text-base">
            {getDocumentDisplayName(doc, documentId)}
          </div>
          <div className="mt-0.5 text-[11px] theme-sidebar-muted truncate">
            {`ID: \`${documentId}\``}
          </div>
        </div>
        {doc?.status && (
          <span
            className={`text-[11px] px-2.5 py-1 rounded-full border whitespace-nowrap ${
              ready
                ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40'
                : 'bg-amber-500/10 text-amber-300 border-amber-400/40'
            }`}
          >
            {doc.status}
          </span>
        )}
      </header>
      <div className="flex-1 flex min-h-0 overflow-hidden">
        <div
          className={`relative min-w-0 min-h-0 flex flex-col overflow-hidden transition-all duration-300 ease-out ${
            ready
              ? isMarkdownOpen
                ? 'w-1/2 opacity-100'
                : 'w-0 opacity-0 pointer-events-none'
              : 'w-1/2'
          }`}
        >
          {ready ? (
            isMarkdownOpen && (
              <MarkdownViewer
                documentId={documentId}
                activeHighlight={activeHighlight}
                onClose={() => setIsMarkdownOpen(false)}
              />
            )
          ) : (
            <div className="flex-1 flex items-center justify-center bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6 min-h-0">
              <div className="animate-processing-fade-in max-w-sm w-full text-center">
                <div className="relative inline-flex justify-center mb-6">
                  <div className="absolute inset-0 rounded-full bg-indigo-500/35 animate-ping" style={{ animationDuration: '2s' }} />
                  <div className="relative w-16 h-16 rounded-2xl bg-slate-900 border border-indigo-400/40 shadow-xl animate-processing-float flex items-center justify-center animate-processing-glow">
                    <svg className="w-8 h-8 text-indigo-300 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5v-7.5H8.25v7.5z" />
                    </svg>
                  </div>
                </div>
                <h3 className="text-slate-100 font-semibold text-lg mb-1">
                  {doc?.status === 'uploaded' && 'Waiting for processing'}
                  {(doc?.status === 'processing' || doc?.status === 'vectorized') && 'Preparing your document'}
                </h3>
                <p className="text-slate-400 text-sm mb-6">
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
                          isPast ? 'bg-indigo-500' : isActive ? 'animate-processing-shimmer bg-indigo-400/80' : 'bg-slate-700'
                        }`}
                        title={step}
                      />
                    )
                  })}
                </div>
                <div className="h-1 w-full rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all duration-700 ease-out"
                    style={{
                      width: doc?.status === 'uploaded' ? '25%' : doc?.status === 'processing' || doc?.status === 'vectorized' ? '75%' : '100%',
                    }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
        <div
          className={`min-w-0 min-h-0 flex flex-col overflow-hidden transition-all duration-300 ease-out ${
            ready && isMarkdownOpen ? 'w-1/2' : 'w-full'
          }`}
        >
          <ChatPanel
            documentId={documentId}
            documentReady={ready}
            onHighlightChunk={(chunk) => {
              setActiveHighlight(chunk)
              setIsMarkdownOpen(true)
            }}
          />
        </div>
      </div>
    </div>
  )
}
