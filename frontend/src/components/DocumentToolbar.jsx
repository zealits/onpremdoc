import { useState } from 'react'
import { useDocumentSummary, useExtractFromDocument, useEmailDocumentSummary, useDocumentPageRanges } from '../api/hooks'
import ExtractModal from './ExtractModal'
import EmailModal from './EmailModal'
import EconomicsPipelineModal from './EconomicsPipelineModal'

export default function DocumentToolbar({
  documentId,
  documentReady,
  documentName,
  summaryText,
  isSummaryLoading,
  onHighlightChunk,
  onOpenDocument,
}) {
  const [showExtract, setShowExtract] = useState(false)
  const [showEmail, setShowEmail] = useState(false)
  const [showSummaryPanel, setShowSummaryPanel] = useState(false)
  const [showEconomics, setShowEconomics] = useState(false)
  const [showIndexPanel, setShowIndexPanel] = useState(false)

  const { data: summaryData, refetch: refetchSummary, isFetching: summaryFetching } = useDocumentSummary(documentId, {
    enabled: !!documentId && documentReady && showSummaryPanel,
  })
  const {
    data: pageRangesData,
    isFetching: pageRangesFetching,
    isError: pageRangesError,
    refetch: refetchPageRanges,
  } = useDocumentPageRanges(documentId, {
    enabled: !!documentId && documentReady && showIndexPanel,
  })
  const extractMutation = useExtractFromDocument(documentId)
  const emailMutation = useEmailDocumentSummary(documentId)

  const displaySummary = summaryData?.summary ?? summaryText
  const loadingSummary = isSummaryLoading || summaryFetching

  return (
    <div className="relative shrink-0 border-b theme-sidebar px-3 py-2 flex flex-wrap items-center gap-2">
      <div className="flex items-center gap-1.5">
        {/* View document */}
        {documentReady && typeof onOpenDocument === 'function' && (
          <button
            type="button"
            onClick={onOpenDocument}
            className="doc-toolbar-action-btn inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm transition-colors"
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
            className="doc-toolbar-action-btn inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm transition-colors"
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
            className="doc-toolbar-action-btn inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm transition-colors disabled:opacity-50"
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
            className="doc-toolbar-action-btn inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm transition-colors"
            title="Email summary"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
            <span className="hidden sm:inline">Email</span>
          </button>
        )}

        {/* Processing summary */}
        {documentReady && (
          <button
            type="button"
            onClick={() => setShowEconomics(true)}
            className="doc-toolbar-action-btn inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm transition-colors"
            title="Show processing summary"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.8}
                d="M4 19h4V9H4v10zm6 0h4V5h-4v14zm6 0h4v-7h-4v7z"
              />
            </svg>
            <span className="hidden sm:inline">Processing summary</span>
          </button>
        )}

        {/* Index / Sections */}
        {documentReady && (
          <button
            type="button"
            onClick={() => setShowIndexPanel((v) => !v)}
            className="doc-toolbar-action-btn inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm transition-colors"
            title="Show document index (sections)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.8}
                d="M5 5h6M5 9h10M5 13h8M5 17h6"
              />
            </svg>
            <span className="hidden sm:inline">Index</span>
          </button>
        )}
      </div>

      {/* Summary panel (inline) */}
      {showSummaryPanel && documentReady && (
        <div className="w-full mt-2 p-3 rounded-xl border theme-card text-sm">
          {loadingSummary ? (
            <div className="flex items-center gap-2 theme-sidebar-muted">
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
                  className="theme-sidebar-muted hover:opacity-80"
                  aria-label="Close"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <p className="whitespace-pre-line text-inherit leading-relaxed opacity-90">{displaySummary || 'No summary available.'}</p>
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

      {/* Index / page ranges panel */}
      {showIndexPanel && documentReady && (
        <div className="index-panel w-full mt-2 rounded-xl border theme-card overflow-hidden shadow-lg">
          <div className="index-panel-header flex items-center justify-between gap-3 px-4 py-3 border-b theme-sidebar">
            <div className="flex items-center gap-2">
              <div className="index-panel-icon flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-500/15 text-indigo-400">
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h7" />
                </svg>
              </div>
              <div>
                <h3 className="text-sm font-semibold text-inherit">Document index</h3>
                <p className="text-[11px] theme-sidebar-muted">Section summaries by page range</p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => refetchPageRanges()}
                disabled={pageRangesFetching}
                className="index-panel-refresh rounded-lg px-2.5 py-1.5 text-xs font-medium theme-sidebar-muted hover:bg-indigo-500/10 hover:text-indigo-400 transition-colors disabled:opacity-50"
                title="Refresh index"
              >
                <span className="sr-only">Refresh</span>
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.8}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16 4v4m0 0h-4m4 0H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2V8a2 2 0 00-2-2h-2" />
                </svg>
              </button>
              <button
                type="button"
                onClick={() => setShowIndexPanel(false)}
                className="rounded-lg p-1.5 theme-sidebar-muted hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
                aria-label="Close index"
              >
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
          <div className="index-panel-body max-h-72 overflow-y-auto px-3 py-3 theme-card">
            {pageRangesFetching && (
              <div className="flex flex-col items-center justify-center gap-3 py-8 theme-sidebar-muted">
                <span className="inline-block h-8 w-8 rounded-full border-2 border-indigo-400/60 border-t-transparent animate-spin" />
                <span className="text-sm">Loading index…</span>
              </div>
            )}
            {pageRangesError && !pageRangesFetching && (
              <div className="index-panel-error flex flex-col items-center gap-2 py-6 px-4 rounded-xl bg-red-500/10 border border-red-500/20 text-center">
                <svg className="h-10 w-10 text-red-400/80" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <p className="text-sm font-medium text-red-400">Could not load index</p>
                <p className="text-xs text-red-400/80">Try again or ensure the document is fully processed.</p>
                <button
                  type="button"
                  onClick={() => refetchPageRanges()}
                  className="mt-1 rounded-lg bg-red-500/20 px-3 py-1.5 text-xs font-medium text-red-300 hover:bg-red-500/30 transition-colors"
                >
                  Retry
                </button>
              </div>
            )}
            {!pageRangesFetching && !pageRangesError && Array.isArray(pageRangesData) && pageRangesData.length > 0 && (
              <ul className="index-list space-y-3" role="list">
                {pageRangesData.map((r, idx) => (
                  <li
                    key={`${r.start_page}-${r.end_page}-${idx}`}
                    className="index-card group relative rounded-xl border theme-sidebar overflow-hidden transition-all duration-200 hover:border-indigo-400/40 hover:shadow-md"
                  >
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-indigo-500/60 to-violet-500/60 rounded-l-xl opacity-80 group-hover:opacity-100 transition-opacity" aria-hidden />
                    <div className="pl-4 pr-3 py-3">
                      <div className="flex items-center gap-2 flex-wrap mb-2">
                        <span
                          className="index-section-num flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-indigo-500/20 text-xs font-bold text-indigo-400"
                          aria-hidden
                        >
                          {idx + 1}
                        </span>
                        <span className="index-page-badge inline-flex items-center gap-1 rounded-full border border-indigo-400/30 bg-indigo-500/10 px-2.5 py-0.5 text-[11px] font-medium text-indigo-300">
                          <svg className="h-3 w-3 opacity-80" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                          pp. {r.start_page}–{r.end_page}
                        </span>
                      </div>
                      <p className="index-summary text-sm leading-relaxed text-inherit opacity-95 pl-0">
                        {r.summary}
                      </p>
                    </div>
                  </li>
                ))}
              </ul>
            )}
            {!pageRangesFetching && !pageRangesError && Array.isArray(pageRangesData) && pageRangesData.length === 0 && (
              <div className="flex flex-col items-center gap-2 py-8 theme-sidebar-muted text-center">
                <svg className="h-12 w-12 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="text-sm font-medium">No index available</p>
                <p className="text-xs max-w-[240px]">This document has no section summaries yet. Process it to generate an index.</p>
              </div>
            )}
          </div>
        </div>
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

      {showEconomics && (
        <EconomicsPipelineModal
          documentId={documentId}
          documentName={documentName}
          onClose={() => setShowEconomics(false)}
        />
      )}
    </div>
  )
}
