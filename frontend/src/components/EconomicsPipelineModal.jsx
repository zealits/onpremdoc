import { useState } from 'react'
import { useEconomicsPipeline, useQueryEconomics } from '../api/hooks'

export default function EconomicsPipelineModal({ documentId, documentName, onClose }) {
  const [activeTab, setActiveTab] = useState('pipeline')
  const [totalsOpen, setTotalsOpen] = useState(false)

  const {
    data,
    isLoading,
    isError,
    error,
    refetch,
    isFetching,
  } = useEconomicsPipeline(documentId)

  const {
    data: queriesData,
    isLoading: isQueriesLoading,
    isError: isQueriesError,
    error: queriesError,
    refetch: refetchQueries,
    isFetching: queriesFetching,
  } = useQueryEconomics(documentId, null, {
    enabled: !!documentId && activeTab === 'queries',
  })

  const events = data?.events ?? []
  const totals = data?.totals ?? null

  const uploadEvent = events.find((e) => e.step === 'pdf_upload') || null
  const vectorizationEvent = events.find((e) => e.step === 'vectorization') || null
  const pagesFromProcessing =
    (events.find((e) => e.step === 'pdf_processing')?.extra?.total_pages ?? null) ||
    vectorizationEvent?.extra?.total_pages ||
    null

  const filename = uploadEvent?.extra?.filename ?? null
  const fileSizeBytes = uploadEvent?.extra?.file_size_bytes ?? null
  const totalPages = pagesFromProcessing
  const totalWords = vectorizationEvent?.extra?.total_words ?? null

  const formatUsd = (value) =>
    typeof value === 'number'
      ? value.toFixed(3)
      : '-'

  const humanFileSize =
    typeof fileSizeBytes === 'number'
      ? (() => {
          const mb = fileSizeBytes / (1024 * 1024)
          return `${mb.toFixed(2)} MB`
        })()
      : null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50"
      role="dialog"
      aria-modal="true"
      aria-labelledby="economics-modal-title"
    >
      <div className="theme-card rounded-2xl shadow-2xl w-full max-w-3xl border max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between px-5 py-4 border-b theme-sidebar">
          <div>
            <h2 id="economics-modal-title" className="text-lg font-semibold">
              Transaction Summary
            </h2>
            {documentName && (
              <p className="text-xs opacity-70 mt-0.5 truncate max-w-md">
                {documentName}
              </p>
            )}
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1 rounded-lg bg-slate-900/40 p-0.5 border border-slate-700/60">
              <button
                type="button"
                onClick={() => setActiveTab('pipeline')}
                className={`px-2.5 py-1.5 text-xs rounded-md font-medium transition-colors ${
                  activeTab === 'pipeline'
                    ? 'bg-slate-100 text-slate-900 dark:bg-slate-50 dark:text-slate-900'
                    : 'text-slate-300 hover:bg-slate-800/80'
                }`}
              >
                Processing
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('queries')}
                className={`px-2.5 py-1.5 text-xs rounded-md font-medium transition-colors ${
                  activeTab === 'queries'
                    ? 'bg-slate-100 text-slate-900 dark:bg-slate-50 dark:text-slate-900'
                    : 'text-slate-300 hover:bg-slate-800/80'
                }`}
              >
                Queries
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('events')}
                className={`px-2.5 py-1.5 text-xs rounded-md font-medium transition-colors ${
                  activeTab === 'events'
                    ? 'bg-slate-100 text-slate-900 dark:bg-slate-50 dark:text-slate-900'
                    : 'text-slate-300 hover:bg-slate-800/80'
                }`}
              >
                Events
              </button>
            </div>
            {/* <button
              type="button"
              onClick={() => (activeTab === 'queries' ? refetchQueries() : refetch())}
              disabled={activeTab === 'queries' ? queriesFetching : isFetching}
              className="text-xs px-3 py-1.5 rounded-lg border theme-sidebar-muted disabled:opacity-50"
            >
              {(activeTab === 'pipeline' ? isFetching : queriesFetching) ? 'Refreshing…' : 'Refresh'}
            </button> */}
            <button
              type="button"
              onClick={onClose}
              className="p-1 rounded-lg theme-sidebar-muted hover:opacity-80"
              aria-label="Close"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="p-5 overflow-auto text-sm space-y-4">
          {activeTab === 'pipeline' ? (
            isLoading ? (
              <div className="flex items-center gap-2 theme-sidebar-muted">
                <span className="inline-block w-4 h-4 rounded-full border-2 border-indigo-400/60 border-t-transparent animate-spin" />
                Loading economics data…
              </div>
            ) : isError ? (
              <div className="text-sm text-rose-400">
                {error?.message || 'Failed to load economics pipeline data.'}
              </div>
            ) : !data ? (
              <div className="theme-sidebar-muted">No economics data available for this document.</div>
            ) : (
              <>
              <div className="space-y-4">
                {(filename || totalPages != null || totalWords != null || humanFileSize || totals) && (
                  <div>
                    <h3 className="font-semibold mb-2 text-inherit">Document stats</h3>
                    <div className="border rounded-xl overflow-hidden">
                      <table className="min-w-full text-xs sm:text-sm">
                        <tbody>
                          {totals && (
                            <>
                              <tr className="theme-card border-b">
                                <td className="px-3 py-2">Cost estimate (USD)</td>
                                <td className="px-3 py-2 text-right">
                                  {formatUsd(totals.cost_estimate_usd)}
                                </td>
                              </tr>
                              <tr className="theme-card border-b">
                                <td className="px-3 py-2">Pipeline duration (s)</td>
                                <td className="px-3 py-2 text-right">
                                  {totals.pipeline_seconds != null
                                    ? totals.pipeline_seconds.toFixed(2)
                                    : '-'}
                                </td>
                              </tr>
                            </>
                          )}
                          {filename && (
                            <tr className="theme-card border-b">
                              <td className="px-3 py-2">Filename</td>
                              <td className="px-3 py-2 text-right">{filename}</td>
                            </tr>
                          )}
                          {totalPages != null && (
                            <tr className="theme-card border-b">
                              <td className="px-3 py-2">Number of pages</td>
                              <td className="px-3 py-2 text-right">{totalPages}</td>
                            </tr>
                          )}
                          {totalWords != null && (
                            <tr className="theme-card border-b">
                              <td className="px-3 py-2">Number of words</td>
                              <td className="px-3 py-2 text-right">{totalWords}</td>
                            </tr>
                          )}
                          {humanFileSize && (
                            <tr className="theme-card">
                              <td className="px-3 py-2">File size</td>
                              <td className="px-3 py-2 text-right">
                                {humanFileSize}
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>

              {totals && (
                <div>
                  <button
                    type="button"
                    onClick={() => setTotalsOpen((open) => !open)}
                    className="w-full flex items-center justify-between mb-2 text-inherit font-semibold hover:opacity-80"
                  >
                    <span>Totals</span>
                    <svg
                      className={`w-4 h-4 transition-transform ${totalsOpen ? 'rotate-90' : ''}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </button>
                  {totalsOpen && (
                    <div className="border rounded-xl overflow-hidden">
                      <table className="min-w-full text-xs sm:text-sm">
                        <tbody>
                          <tr className="theme-card border-b">
                            <td className="px-3 py-2">Input tokens</td>
                            <td className="px-3 py-2 text-right">{totals.input_tokens}</td>
                          </tr>
                          <tr className="theme-card border-b">
                            <td className="px-3 py-2">Output tokens</td>
                            <td className="px-3 py-2 text-right">{totals.output_tokens}</td>
                          </tr>
                          <tr className="theme-card border-b">
                            <td className="px-3 py-2">Embedding tokens</td>
                            <td className="px-3 py-2 text-right">{totals.embedding_tokens}</td>
                          </tr>
                          <tr className="theme-card">
                            <td className="px-3 py-2">Total tokens</td>
                            <td className="px-3 py-2 text-right">{totals.total_tokens}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

            </>
            )
          ) : activeTab === 'events' ? (
            events.length === 0 ? (
              <div className="theme-sidebar-muted">No events available for this document.</div>
            ) : (
              <div>
                <h3 className="font-semibold mb-2 text-inherit">Events</h3>
                <div className="border rounded-xl overflow-x-auto">
                  <table className="min-w-max w-full text-xs sm:text-sm">
                    <thead className="theme-sidebar text-left">
                      <tr>
                        <th className="px-3 py-2 whitespace-nowrap">Step</th>
                        <th className="px-3 py-2 whitespace-nowrap">Phase</th>
                        <th className="px-3 py-2 whitespace-nowrap">Timestamp</th>
                        <th className="px-3 py-2 whitespace-nowrap text-right">Input</th>
                        <th className="px-3 py-2 whitespace-nowrap text-right">Output</th>
                        <th className="px-3 py-2 whitespace-nowrap text-right">Embedding</th>
                        <th className="px-3 py-2 whitespace-nowrap text-right">Total tokens</th>
                        <th className="px-3 py-2 whitespace-nowrap text-right">Cost (USD)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {events.map((evt, idx) => (
                        <tr key={`${evt.timestamp}-${idx}`} className="border-t theme-card">
                          <td className="px-3 py-2 whitespace-nowrap">{evt.step}</td>
                          <td className="px-3 py-2 whitespace-nowrap">{evt.phase}</td>
                          <td className="px-3 py-2 whitespace-nowrap">
                            {new Date(evt.timestamp).toLocaleString()}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-right">
                            {evt.input_tokens ?? 0}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-right">
                            {evt.output_tokens ?? 0}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-right">
                            {evt.embedding_tokens ?? 0}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-right">
                            {evt.total_tokens ?? 0}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-right">
                            {evt.pricing?.cost_display ??
                              (evt.cost_estimate_usd != null
                                ? evt.cost_estimate_usd.toFixed(6)
                                : '-')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )
          ) : isQueriesLoading && !queriesData ? (
            <div className="flex items-center gap-2 theme-sidebar-muted">
              <span className="inline-block w-4 h-4 rounded-full border-2 border-indigo-400/60 border-t-transparent animate-spin" />
              Loading query economics…
            </div>
          ) : isQueriesError ? (
            <div className="text-sm text-rose-400">
              {queriesError?.message || 'Failed to load query economics data.'}
            </div>
          ) : !queriesData || !Array.isArray(queriesData.items) || queriesData.items.length === 0 ? (
            <div className="theme-sidebar-muted">No query economics data available for this document.</div>
          ) : (
            <>
              <div>
                <h3 className="font-semibold mb-2 text-inherit">Totals</h3>
                <div className="border rounded-xl overflow-hidden">
                  <table className="min-w-full text-xs sm:text-sm">
                    <tbody>
                      <tr className="theme-card border-b">
                        <td className="px-3 py-2">Input tokens</td>
                        <td className="px-3 py-2 text-right">{queriesData.total_input_tokens}</td>
                      </tr>
                      <tr className="theme-card border-b">
                        <td className="px-3 py-2">Output tokens</td>
                        <td className="px-3 py-2 text-right">{queriesData.total_output_tokens}</td>
                      </tr>
                      <tr className="theme-card border-b">
                        <td className="px-3 py-2">Embedding tokens</td>
                        <td className="px-3 py-2 text-right">{queriesData.total_embedding_tokens}</td>
                      </tr>
                      <tr className="theme-card border-b">
                        <td className="px-3 py-2">Total tokens</td>
                        <td className="px-3 py-2 text-right">{queriesData.total_tokens}</td>
                      </tr>
                      <tr className="theme-card">
                        <td className="px-3 py-2">Total cost estimate (USD)</td>
                        <td className="px-3 py-2 text-right">
                          {formatUsd(queriesData.total_cost_estimate_usd)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <div>
                <h3 className="font-semibold mb-2 text-inherit">Per-query breakdown</h3>
                <div className="border rounded-xl overflow-x-auto">
                  <table className="min-w-max w-full text-xs sm:text-sm">
                    <thead className="theme-sidebar text-left">
                      <tr>
                        <th className="px-3 py-2 whitespace-nowrap">Time</th>
                        <th className="px-3 py-2 whitespace-nowrap">Session</th>
                        <th className="px-3 py-2 whitespace-nowrap">Question</th>
                        <th className="px-3 py-2 whitespace-nowrap text-right">Input</th>
                        <th className="px-3 py-2 whitespace-nowrap text-right">Output</th>
                        <th className="px-3 py-2 whitespace-nowrap text-right">Embedding</th>
                        <th className="px-3 py-2 whitespace-nowrap text-right">Total tokens</th>
                        <th className="px-3 py-2 whitespace-nowrap text-right">Cost (USD)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {queriesData.items.map((item) => (
                        <tr key={item.id} className="border-t theme-card align-top">
                          <td className="px-3 py-2 whitespace-nowrap">
                            {item.created_at ? new Date(item.created_at).toLocaleString() : '-'}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap">{item.session_id}</td>
                          <td className="px-3 py-2 max-w-xs sm:max-w-md">
                            <div className="truncate" title={item.query}>
                              {item.query}
                            </div>
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-right">
                            {item.total_input_tokens ?? 0}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-right">
                            {item.total_output_tokens ?? 0}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-right">
                            {item.total_embedding_tokens ?? 0}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-right">
                            {item.total_tokens ?? 0}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-right">
                            {item.pricing?.cost_display ??
                              (item.cost_estimate_usd != null
                                ? item.cost_estimate_usd.toFixed(6)
                                : '-')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
