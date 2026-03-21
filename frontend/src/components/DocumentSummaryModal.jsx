import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { getDocumentChunks } from '../api/client'

function rehypeCitationSpans() {
  return (tree) => {
    function visit(node, parent, idx) {
      if (parent?.type === 'element' && (parent.properties?.['data-citation'] || parent.properties?.dataCitation)) return
      if (node.type === 'text' && node.value && /\[C\d+\]/.test(node.value)) {
        const parts = node.value.split(/(\[C(\d+)\])/g)
        const newNodes = []
        for (let i = 0; i < parts.length; i++) {
          if (i % 3 === 0 && parts[i]) newNodes.push({ type: 'text', value: parts[i] })
          else if (i % 3 === 2)
            newNodes.push({
              type: 'element',
              tagName: 'span',
              properties: {
                'data-citation': parts[i],
                className: 'chunk-citation',
              },
              children: [{ type: 'text', value: `[C${parts[i]}]` }],
            })
        }
        if (parent && typeof idx === 'number' && newNodes.length) {
          parent.children.splice(idx, 1, ...newNodes)
        }
        return
      }
      if (node.children) for (let i = 0; i < node.children.length; i++) visit(node.children[i], node, i)
    }
    visit(tree, null, 0)
  }
}

function expandChunkReferenceGroups(text) {
  if (typeof text !== 'string') return ''
  let out = text

  // Convert footnote-style `^[C1,C5,C12]` into individual tokens: `[C1] [C5] [C12]`
  out = out.replace(/\^\[\s*([^\]]+?)\s*\]/g, (match, inner) => {
    const ids = Array.from(String(inner).matchAll(/C(\d+)/gi)).map((m) => m[1])
    if (!ids.length) return match
    return ids.map((id) => `[C${id}]`).join(' ')
  })

  // Convert bracket-group style `[C1,C5,C12]` into individual tokens.
  out = out.replace(/\[\s*(C\d+(?:\s*,\s*C\d+)*)\s*\]/gi, (match, inner) => {
    const ids = Array.from(String(inner).matchAll(/C(\d+)/gi)).map((m) => m[1])
    if (!ids.length) return match
    return ids.map((id) => `[C${id}]`).join(' ')
  })

  return out
}

function SummaryCitationButton({ chunkId, fetchChunkById, onHighlightChunk }) {
  const [isLoading, setIsLoading] = useState(false)

  const handleClick = async () => {
    if (isLoading) return
    if (!fetchChunkById || typeof fetchChunkById !== 'function') return
    if (!onHighlightChunk || typeof onHighlightChunk !== 'function') return

    setIsLoading(true)
    try {
      const chunk = await fetchChunkById(chunkId)
      if (chunk) {
        const contentStr = Array.isArray(chunk.content) ? chunk.content.join("\n") : chunk.content ?? ''
        const lines = Array.isArray(chunk.content) ? chunk.content.filter((s) => typeof s === 'string' && s.trim().length > 0) : []

        onHighlightChunk({
          pageNumber: chunk.page_number,
          text: contentStr,
          sectionTitle: chunk.section_title,
          heading: chunk.heading,
          lines,
        })
      }
    } finally {
      setIsLoading(false)
    }
  }

  const baseClasses =
    'inline-flex align-middle px-1.5 py-0.5 rounded text-xs font-medium cursor-pointer transition-all duration-200 touch-manipulation whitespace-nowrap'
  const sourceClasses =
    'text-indigo-600 hover:text-indigo-800 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 hover:border-indigo-300 dark:text-indigo-300 dark:hover:text-indigo-200 dark:bg-indigo-900/30 dark:hover:bg-indigo-900/50 dark:border-indigo-800 dark:hover:border-indigo-600 hover:scale-105 active:scale-95 hover:shadow-sm'

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={isLoading}
      className={`${baseClasses} ${sourceClasses}`}
      title="Go to this source (scroll + highlight)"
    >
      {isLoading ? '…' : `${chunkId}`}
    </button>
  )
}

export default function DocumentSummaryModal({
  isOpen,
  onClose,
  summaryData,
  summaryText,
  isSummaryLoading,
  summaryFetching,
  refetchSummary,
  documentName,
  documentId,
  onHighlightChunk,
}) {
  const MAX_VISIBLE_CITATIONS = 7

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = 'unset'
    }
    
    return () => {
      document.body.style.overflow = 'unset'
    }
  }, [isOpen])

  const displaySummary = summaryData?.summary ?? summaryText
  const loadingSummary = isSummaryLoading || summaryFetching

  const chunkLookupCacheRef = useRef(new Map())
  const fetchChunkById = useCallback(
    async (chunkId) => {
      const id = Number(chunkId)
      if (!Number.isFinite(id)) return null
      if (chunkLookupCacheRef.current.has(id)) return chunkLookupCacheRef.current.get(id)
      try {
        const chunks = await getDocumentChunks(documentId, [id])
        const chunk = Array.isArray(chunks) ? chunks.find((c) => c.chunk_index === id) : null
        if (chunk) chunkLookupCacheRef.current.set(id, chunk)
        return chunk
      } catch {
        return null
      }
    },
    [documentId]
  )

  if (!isOpen) return null

  // Keep only first N unique citation chips visible in the rendered summary.
  const visibleCitationIds = new Set()

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="relative w-full max-w-3xl max-h-[90vh] bg-white dark:bg-slate-800 rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-slate-800 dark:to-slate-700">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 text-white shadow-lg">
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Document Summary</h2>
              {documentName && (
                <p className="text-sm text-gray-600 dark:text-gray-400 truncate max-w-md">
                  {documentName}
                </p>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-gray-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700 transition-colors"
            aria-label="Close modal"
          >
            <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="overflow-auto max-h-[calc(90vh-120px)]">
          {loadingSummary && (
            <div className="flex flex-col items-center justify-center gap-4 py-16">
              <div className="h-12 w-12 rounded-full border-4 border-blue-200 border-t-blue-600 animate-spin"></div>
              <p className="text-gray-600 dark:text-gray-400">Generating summary...</p>
            </div>
          )}

          {!loadingSummary && displaySummary && (
            <div className="p-8">
              <div className="prose prose-lg dark:prose-invert max-w-none">
                <div className="mb-6">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="h-1 w-8 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full"></div>
                    <h3 className="text-xl font-semibold text-gray-900 dark:text-white m-0">Key Insights</h3>
                  </div>
                </div>
                
                <div className="summary-content bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-slate-800 rounded-xl p-6 border border-gray-100 dark:border-gray-700">
                  <div className="text-gray-800 dark:text-gray-200 leading-relaxed text-base space-y-4">
                    <ReactMarkdown
                      rehypePlugins={[rehypeCitationSpans]}
                      components={{
                        p: ({ node, ...props }) => (
                          <p
                            {...props}
                            className="m-0 last:mb-0 first-line:font-medium first-line:tracking-wide"
                          />
                        ),
                        span: (props) => {
                          let citationId =
                            props['data-citation'] ??
                            props.dataCitation ??
                            props.node?.properties?.['data-citation'] ??
                            props.node?.properties?.dataCitation

                          if (citationId == null || citationId === '') {
                            const child = props.children
                            const str =
                              typeof child === 'string'
                                ? child
                                : Array.isArray(child) && child.length === 1 && typeof child[0] === 'string'
                                  ? child[0]
                                  : null
                            const m = str && str.match(/^\[C(\d+)\]$/)
                            if (m) citationId = m[1]
                          }

                          if (citationId != null && citationId !== '') {
                            const id = parseInt(String(citationId), 10)
                            if (!Number.isNaN(id)) {
                              if (!visibleCitationIds.has(id) && visibleCitationIds.size >= MAX_VISIBLE_CITATIONS) {
                                return null
                              }
                              visibleCitationIds.add(id)
                              return (
                                <SummaryCitationButton
                                  chunkId={id}
                                  fetchChunkById={fetchChunkById}
                                  onHighlightChunk={onHighlightChunk}
                                />
                              )
                            }
                          }

                          const { node, ...spanProps } = props
                          return <span {...spanProps} />
                        },
                      }}
                    >
                      {expandChunkReferenceGroups(displaySummary)}
                    </ReactMarkdown>
                  </div>
                </div>

                {/* <div className="mt-6 flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09Z" />
                    </svg>
                    <span>Generated by AI</span>
                  </div>
                  <button
                    onClick={refetchSummary}
                    className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 bg-blue-50 hover:bg-blue-100 dark:bg-blue-900/30 dark:hover:bg-blue-900/50 rounded-lg transition-colors"
                  >
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" />
                    </svg>
                    Refresh Summary
                  </button>
                </div> */}
              </div>
            </div>
          )}

          {!loadingSummary && !displaySummary && (
            <div className="flex flex-col items-center gap-4 py-16 text-center">
              <div className="h-16 w-16 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                <svg className="h-8 w-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <div>
                <p className="text-lg font-medium text-gray-900 dark:text-white">No summary available</p>
                <p className="text-gray-600 dark:text-gray-400 mt-1 max-w-sm">Generate a summary by processing the document first.</p>
              </div>
              <button
                onClick={refetchSummary}
                className="mt-2 inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
              >
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" />
                </svg>
                Generate Summary
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}