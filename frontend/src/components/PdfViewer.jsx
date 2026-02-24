import { useState } from 'react'
import { getPdfUrl } from '../api/client'

export default function PdfViewer({ documentId }) {
  const [scale, setScale] = useState(1)

  if (!documentId) return null

  const url = getPdfUrl(documentId)

  return (
    <div className="flex flex-col h-full min-h-0 bg-gray-100">
      <div className="flex items-center gap-2 px-2 py-1.5 bg-white border-b border-gray-200 shrink-0">
        <button
          type="button"
          onClick={() => setScale((s) => Math.max(0.5, s - 0.25))}
          className="px-2 py-1 rounded border border-gray-300 text-sm hover:bg-gray-50"
          aria-label="Zoom out"
        >
          −
        </button>
        <span className="text-sm text-gray-600 min-w-[4rem] text-center">
          {Math.round(scale * 100)}%
        </span>
        <button
          type="button"
          onClick={() => setScale((s) => Math.min(2, s + 0.25))}
          className="px-2 py-1 rounded border border-gray-300 text-sm hover:bg-gray-50"
          aria-label="Zoom in"
        >
          +
        </button>
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="ml-2 text-sm text-indigo-600 hover:underline"
        >
          Open in new tab
        </a>
      </div>
      <div className="flex-1 min-h-0 overflow-auto p-2">
        <iframe
          title="PDF document"
          src={url}
          className="w-full border-0 rounded bg-white"
          style={{
            minHeight: 'calc(100vh - 180px)',
            height: '100%',
            transform: `scale(${scale})`,
            transformOrigin: 'top left',
          }}
        />
      </div>
    </div>
  )
}
