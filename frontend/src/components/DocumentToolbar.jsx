import { useState } from 'react'
import { useDocumentSummary, useExtractFromDocument, useEmailDocumentSummary } from '../api/hooks'
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

  const { data: summaryData, refetch: refetchSummary, isFetching: summaryFetching } = useDocumentSummary(documentId, {
    enabled: !!documentId && documentReady && showSummaryPanel,
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

        {/* Economics */}
        {documentReady && (
          <button
            type="button"
            onClick={() => setShowEconomics(true)}
            className="doc-toolbar-action-btn inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm transition-colors"
            title="Show pipeline economics"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.8}
                d="M3 17l6-6 4 4 8-8"
              />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M3 7h4v10H3z" />
            </svg>
            <span className="hidden sm:inline">Economics</span>
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
