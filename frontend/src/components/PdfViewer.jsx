import { useState, useMemo, useRef, useEffect } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import { getPdfUrl } from '../api/client'

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'react-pdf/node_modules/pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

function normalize(text) {
  return (text || '').replace(/\s+/g, ' ').trim().toLowerCase()
}

export default function PdfViewer({ documentId, activeHighlight }) {
  const [scale, setScale] = useState(1)
  const [numPages, setNumPages] = useState(null)
  const pageRefs = useRef({})

  const highlightConfig = useMemo(() => {
    if (!activeHighlight?.text) return null
    const normalized = normalize(activeHighlight.text)
    if (!normalized) return null
    const words = normalized.split(' ').filter((w) => w.length > 3)
    const tokens = words.slice(0, 8)
    return {
      pageNumber: activeHighlight.pageNumber,
      tokens,
    }
  }, [activeHighlight])

  if (!documentId) return null

  const fileUrl = getPdfUrl(documentId)

  const renderTextForPage = (pageNumber) => (textItem) => {
    if (!highlightConfig) return textItem.str
    if (pageNumber !== highlightConfig.pageNumber) return textItem.str
    const normalized = normalize(textItem.str)
    if (!normalized) return textItem.str

    const matches = highlightConfig.tokens.some((token) => normalized.includes(token))
    if (!matches) return textItem.str

    return (
      <span className="bg-yellow-300/70">
        {textItem.str}
      </span>
    )
  }

  useEffect(() => {
    if (!highlightConfig?.pageNumber) return
    const el = pageRefs.current[highlightConfig.pageNumber]
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [highlightConfig])

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
        <span className="ml-2 text-xs text-gray-500">
          {numPages ? `Pages 1–${numPages}` : 'Pages'}
        </span>
        <a
          href={fileUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="ml-auto text-sm text-indigo-600 hover:underline"
        >
          Open in new tab
        </a>
      </div>
      <div className="flex-1 min-h-0 overflow-auto p-2">
        <Document
          file={fileUrl}
          onLoadSuccess={({ numPages: loaded }) => setNumPages(loaded)}
          loading={
            <div className="text-gray-500 text-sm px-2 py-4">Loading PDF…</div>
          }
          error={
            <div className="text-red-600 text-sm px-2 py-4">
              Failed to load PDF.
            </div>
          }
        >
          {numPages &&
            Array.from({ length: numPages }, (_, index) => {
              const pageNumber = index + 1
              return (
                <div
                  key={pageNumber}
                  ref={(el) => {
                    if (el) pageRefs.current[pageNumber] = el
                  }}
                  className="mb-4 flex justify-center"
                >
                  <Page
                    pageNumber={pageNumber}
                    scale={scale}
                    renderTextLayer
                    renderAnnotationLayer={false}
                    customTextRenderer={renderTextForPage(pageNumber)}
                  />
                </div>
              )
            })}
        </Document>
      </div>
    </div>
  )
}
