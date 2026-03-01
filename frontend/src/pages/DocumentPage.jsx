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
      <div className="flex-1 flex items-center justify-center text-gray-500">
        Loading document…
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
            <div className="flex-1 flex items-center justify-center text-gray-500 bg-gray-50 p-4">
              {doc?.status === 'uploaded' && 'Waiting for processing…'}
              {(doc?.status === 'processing' || doc?.status === 'vectorized') &&
                'Vectorizing… Markdown will appear when ready.'}
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
