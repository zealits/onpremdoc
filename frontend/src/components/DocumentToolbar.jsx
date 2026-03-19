import { useState } from 'react'
import { useDocumentSummary, useExtractFromDocument, useEmailDocumentSummary, useDocumentPageRanges } from '../api/hooks'
import ExtractModal from './ExtractModal'
import EmailModal from './EmailModal'
import EconomicsPipelineModal from './EconomicsPipelineModal'
import DocumentIndexModal from './DocumentIndexModal'
import DocumentSummaryModal from './DocumentSummaryModal'

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
    enabled: !!documentId && documentReady,
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
            onClick={() => setShowSummaryPanel(true)}
            className="doc-toolbar-action-btn inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm transition-colors"
            title="Show summary"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span className="hidden sm:inline">Summarize</span>
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
            onClick={() => setShowIndexPanel(true)}
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

      {/* Document Summary Modal */}
      <DocumentSummaryModal
        isOpen={showSummaryPanel && documentReady}
        onClose={() => setShowSummaryPanel(false)}
        summaryData={summaryData}
        summaryText={summaryText}
        isSummaryLoading={isSummaryLoading}
        summaryFetching={summaryFetching}
        refetchSummary={refetchSummary}
        documentName={documentName}
        documentId={documentId}
        onHighlightChunk={onHighlightChunk}
      />

      {/* Document Index Modal */}
      <DocumentIndexModal
        isOpen={showIndexPanel && documentReady}
        onClose={() => setShowIndexPanel(false)}
        pageRangesData={pageRangesData}
        pageRangesFetching={pageRangesFetching}
        pageRangesError={pageRangesError}
        refetchPageRanges={refetchPageRanges}
      />

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
