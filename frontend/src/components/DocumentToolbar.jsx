import { useState } from 'react'
import {
  useSearchDocument,
  useDocumentSummary,
  useExtractFromDocument,
  useEmailDocumentSummary,
} from '../api/hooks'
import SearchResults from './SearchResults'
import ExtractModal from './ExtractModal'
import EmailModal from './EmailModal'

export default function DocumentToolbar({
  documentId,
  documentReady,
  documentName,
  summaryText,
  isSummaryLoading,
  onHighlightChunk,
  onOpenDocument,
}) {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState(null)
  const [showExtract, setShowExtract] = useState(false)
  const [showEmail, setShowEmail] = useState(false)
  const [showSummaryPanel, setShowSummaryPanel] = useState(false)

  const searchMutation = useSearchDocument(documentId)
  const { data: summaryData, refetch: refetchSummary, isFetching: summaryFetching } = useDocumentSummary(documentId, {
    enabled: !!documentId && documentReady && showSummaryPanel,
  })
  const extractMutation = useExtractFromDocument(documentId)
  const emailMutation = useEmailDocumentSummary(documentId)

  const displaySummary = summaryData?.summary ?? summaryText
  const loadingSummary = isSummaryLoading || summaryFetching

  const handleSearch = (e) => {
    e?.preventDefault()
    const q = searchQuery.trim()
    if (!q || !documentId || !documentReady) return
    searchMutation.mutate(
      { query: q, limit: 12 },
      {
        onSuccess: (data) => {
          setSearchResults({ query: q, ...data })
        },
        onError: () => {
          setSearchResults({ query: q, chunks: [] })
        },
      }
    )
  }

  const clearSearch = () => {
    setSearchQuery('')
    setSearchResults(null)
  }

  return (
    <div className="relative shrink-0 border-b theme-sidebar px-3 py-2 flex flex-wrap items-center gap-2">
      {/* Search */}
      {documentReady && (
        <form onSubmit={handleSearch} className="flex-1 min-w-[180px] max-w-md flex gap-2">
          <div className="relative flex-1">
            <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </span>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search in document…"
              className="w-full pl-9 pr-8 py-1.5 rounded-lg text-sm bg-slate-800/60 border border-slate-600 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-400"
              aria-label="Search in document"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={clearSearch}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                aria-label="Clear search"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
          <button
            type="submit"
            disabled={!searchQuery.trim() || searchMutation.isPending}
            className="px-3 py-1.5 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {searchMutation.isPending ? '…' : 'Search'}
          </button>
        </form>
      )}

      <div className="flex items-center gap-1.5">
        {/* View document */}
        {documentReady && typeof onOpenDocument === 'function' && (
          <button
            type="button"
            onClick={onOpenDocument}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-600 text-sm text-slate-300 hover:bg-slate-700/50 hover:border-slate-500"
            title="View document text"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span className="hidden sm:inline">View doc</span>
          </button>
        )}
        {/* Summarize */}
        {documentReady && (
          <button
            type="button"
            onClick={() => setShowSummaryPanel((v) => !v)}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-600 text-sm text-slate-300 hover:bg-slate-700/50 hover:border-slate-500"
            title="Show summary"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span className="hidden sm:inline">Summarize</span>
          </button>
        )}

        {/* Extract */}
        {documentReady && (
          <button
            type="button"
            onClick={() => setShowExtract(true)}
            disabled={extractMutation.isPending}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-600 text-sm text-slate-300 hover:bg-slate-700/50 hover:border-slate-500 disabled:opacity-50"
            title="Extract key facts, entities, dates"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6z" />
            </svg>
            <span className="hidden sm:inline">Extract</span>
          </button>
        )}

        {/* Email */}
        {documentReady && (
          <button
            type="button"
            onClick={() => setShowEmail(true)}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-600 text-sm text-slate-300 hover:bg-slate-700/50 hover:border-slate-500"
            title="Email summary"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
            <span className="hidden sm:inline">Email</span>
          </button>
        )}
      </div>

      {/* Summary panel (inline) */}
      {showSummaryPanel && documentReady && (
        <div className="w-full mt-2 p-3 rounded-xl border theme-card text-sm">
          {loadingSummary ? (
            <div className="flex items-center gap-2 text-slate-500">
              <span className="inline-block w-4 h-4 rounded-full border-2 border-indigo-400/60 border-t-transparent animate-spin" />
              Loading summary…
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between gap-2 mb-2">
                <span className="font-semibold text-inherit">Document summary</span>
                <button
                  type="button"
                  onClick={() => setShowSummaryPanel(false)}
                  className="text-slate-500 hover:text-slate-300"
                  aria-label="Close"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <p className="whitespace-pre-line text-slate-300 leading-relaxed">{displaySummary || 'No summary available.'}</p>
              <button
                type="button"
                onClick={() => refetchSummary()}
                className="mt-2 text-xs text-indigo-400 hover:text-indigo-300"
              >
                Refresh summary
              </button>
            </>
          )}
        </div>
      )}

      {/* Search results dropdown/panel */}
      {searchResults && (
        <SearchResults
          results={searchResults}
          onHighlight={onHighlightChunk}
          onClose={clearSearch}
        />
      )}

      {showExtract && (
        <ExtractModal
          documentId={documentId}
          documentName={documentName}
          extractMutation={extractMutation}
          onClose={() => setShowExtract(false)}
        />
      )}

      {showEmail && (
        <EmailModal
          documentId={documentId}
          documentName={documentName}
          emailMutation={emailMutation}
          onClose={() => setShowEmail(false)}
        />
      )}
    </div>
  )
}
