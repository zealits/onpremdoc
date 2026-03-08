import { useRef, useEffect } from 'react'

function contentPreview(content) {
  if (Array.isArray(content)) {
    const text = content.filter((l) => typeof l === 'string' && l.trim()).slice(0, 3).join(' ')
    return (text || '').slice(0, 200) + (text.length > 200 ? '…' : '')
  }
  return typeof content === 'string' ? content.slice(0, 200) + (content.length > 200 ? '…' : '') : ''
}

export default function SearchResults({ results, onHighlight, onClose }) {
  const panelRef = useRef(null)

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (panelRef.current && !panelRef.current.contains(e.target)) {
        onClose?.()
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [onClose])

  if (!results?.chunks?.length) {
    return (
      <div
        ref={panelRef}
        className="absolute left-4 right-4 sm:left-auto sm:right-4 sm:max-w-md top-full mt-1 z-50 rounded-xl border theme-card shadow-xl p-4"
      >
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium">Search: “{results?.query}”</span>
          <button type="button" onClick={onClose} className="text-slate-500 hover:text-slate-300" aria-label="Close">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <p className="text-sm text-slate-500">No matching passages found.</p>
      </div>
    )
  }

  return (
    <div
      ref={panelRef}
      className="absolute left-4 right-4 sm:left-auto sm:right-4 sm:max-w-lg top-full mt-1 z-50 rounded-xl border theme-card shadow-xl overflow-hidden max-h-[70vh] flex flex-col"
    >
      <div className="flex items-center justify-between px-4 py-2 border-b theme-sidebar shrink-0">
        <span className="text-sm font-medium truncate">Search: “{results.query}” — {results.chunks.length} result{results.chunks.length !== 1 ? 's' : ''}</span>
        <button type="button" onClick={onClose} className="text-slate-500 hover:text-slate-300 shrink-0" aria-label="Close">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <ul className="overflow-y-auto column-scroll flex-1 divide-y theme-sidebar divide-slate-700/50">
        {results.chunks.map((chunk) => {
          const lines = Array.isArray(chunk.content) ? chunk.content : (chunk.content ? [chunk.content] : [])
          const text = Array.isArray(chunk.content) ? chunk.content.join('\n') : (chunk.content || '')
          return (
            <li key={chunk.chunk_index}>
              <button
                type="button"
                onClick={() => {
                  onHighlight?.({
                    pageNumber: chunk.page_number,
                    text,
                    sectionTitle: chunk.section_title,
                    heading: chunk.heading,
                    lines,
                  })
                  onClose?.()
                }}
                className="w-full text-left px-4 py-3 hover:bg-slate-700/40 transition-colors"
              >
                <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
                  {chunk.section_title && <span>{chunk.section_title}</span>}
                  {chunk.page_number != null && <span>Page {chunk.page_number}</span>}
                  {chunk.similarity_score != null && (
                    <span>{Math.round(chunk.similarity_score * 100)}% match</span>
                  )}
                </div>
                <p className="text-sm text-slate-200 line-clamp-2">{contentPreview(chunk.content)}</p>
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
