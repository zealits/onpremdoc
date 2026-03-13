import { useEffect } from 'react'

export default function DocumentIndexModal({
  isOpen,
  onClose,
  pageRangesData,
  pageRangesFetching,
  pageRangesError,
  refetchPageRanges,
}) {
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

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-white dark:bg-slate-800 rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-slate-800 dark:to-slate-700">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-lg">
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h7" />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Document Index</h2>
              <p className="text-sm text-gray-600 dark:text-gray-400">Section summaries organized by page range</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
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
        </div>

        {/* Content */}
        <div className="overflow-auto max-h-[calc(90vh-120px)]">
          {pageRangesFetching && (
            <div className="flex flex-col items-center justify-center gap-4 py-16">
              <div className="h-12 w-12 rounded-full border-4 border-indigo-200 border-t-indigo-600 animate-spin"></div>
              <p className="text-gray-600 dark:text-gray-400">Loading document index...</p>
            </div>
          )}

          {pageRangesError && !pageRangesFetching && (
            <div className="flex flex-col items-center gap-4 py-16 px-6 text-center">
              <div className="h-16 w-16 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
                <svg className="h-8 w-8 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div>
                <p className="text-lg font-medium text-red-600 dark:text-red-400">Failed to load index</p>
                <p className="text-gray-600 dark:text-gray-400 mt-1">Try again or ensure the document is fully processed.</p>
              </div>
              <button
                onClick={refetchPageRanges}
                className="mt-2 rounded-lg bg-red-600 px-4 py-2 text-white hover:bg-red-700 transition-colors"
              >
                Retry
              </button>
            </div>
          )}

          {!pageRangesFetching && !pageRangesError && Array.isArray(pageRangesData) && pageRangesData.length > 0 && (
            <div className="p-6">
              <div className="overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
                <table className="w-full table-auto">
                  <thead className="bg-gray-50 dark:bg-gray-800">
                    <tr>
                      <th className="px-4 py-3 text-center text-xs font-semibold text-gray-900 dark:text-white uppercase tracking-wide">
                        Section
                      </th>
                      <th className="px-4 py-3 w-32 text-center text-xs font-semibold text-gray-900 dark:text-white uppercase tracking-wide whitespace-nowrap">
                        Pages
                      </th>
                      <th className="px-4 py-3 text-center text-xs font-semibold text-gray-900 dark:text-white uppercase tracking-wide">
                        Title
                      </th>
                      <th className="px-4 py-3 text-center text-xs font-semibold text-gray-900 dark:text-white uppercase tracking-wide">
                        Summary
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                    {pageRangesData.map((section, index) => (
                      <tr key={`${section.start_page}-${section.end_page}-${index}`}>
                        <td className="px-4 py-4 text-center">
                          <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 text-sm font-bold text-white">
                            {index + 1}
                          </span>
                        </td>
                        <td className="px-4 py-4 w-32 text-center whitespace-nowrap">
                          <span className="inline-flex items-center gap-1 rounded-full border border-indigo-200 dark:border-indigo-800 bg-indigo-50 dark:bg-indigo-900/30 px-3 py-1 text-sm font-medium text-indigo-700 dark:text-indigo-300">
                            {section.start_page}–{section.end_page}
                          </span>
                        </td>
                        <td className="px-4 py-4">
                          {section.title && section.title.trim() ? (
                            <p className="font-semibold text-gray-900 dark:text-white leading-snug">
                              {section.title}
                            </p>
                          ) : (
                            <p className="italic text-gray-500 dark:text-gray-400 leading-snug">
                              Untitled section
                            </p>
                          )}
                        </td>
                        <td className="px-4 py-4">
                          <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                            {section.summary}
                          </p>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {!pageRangesFetching && !pageRangesError && Array.isArray(pageRangesData) && pageRangesData.length === 0 && (
            <div className="flex flex-col items-center gap-4 py-16 text-center">
              <div className="h-16 w-16 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                <svg className="h-8 w-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <div>
                <p className="text-lg font-medium text-gray-900 dark:text-white">No index available</p>
                <p className="text-gray-600 dark:text-gray-400 mt-1 max-w-sm">This document has no section summaries yet. Process it to generate an index.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}