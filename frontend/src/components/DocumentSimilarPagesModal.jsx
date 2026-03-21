import { useEffect, useMemo } from 'react'

function buildPairKey(pages = []) {
  const [a, b] = [...pages].sort((x, y) => x - y)
  return `${a}-${b}`
}

function formatPercent(score) {
  return `${(score * 100).toFixed(2)}%`
}

export default function DocumentSimilarPagesModal({
  isOpen,
  onClose,
  duplicatesData,
  duplicatesFetching,
  duplicatesError,
  refetchDuplicates,
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

  const filteredDuplicates = useMemo(() => {
    const raw = Array.isArray(duplicatesData?.duplicates) ? duplicatesData.duplicates : []
    if (!raw.length) return []

    const exactByPair = new Map()
    const exactPages = new Set()
    const nearByPair = new Map()

    raw.forEach((item) => {
      const pages = Array.isArray(item?.pages) ? item.pages : []
      if (pages.length !== 2) return

      const [p1, p2] = pages
      const key = buildPairKey(pages)
      const score = Number(item?.score ?? 0)
      const normalized = { pages: [Math.min(p1, p2), Math.max(p1, p2)], score }

      if (score === 1) {
        exactByPair.set(key, normalized)
        exactPages.add(normalized.pages[0])
        exactPages.add(normalized.pages[1])
        return
      }

      const existing = nearByPair.get(key)
      if (!existing || score > existing.score) {
        nearByPair.set(key, normalized)
      }
    })

    const exactRows = Array.from(exactByPair.values()).sort((a, b) => {
      if (a.pages[0] !== b.pages[0]) return a.pages[0] - b.pages[0]
      return a.pages[1] - b.pages[1]
    })

    const nearRows = Array.from(nearByPair.values())
      .filter((item) => !exactPages.has(item.pages[0]) && !exactPages.has(item.pages[1]))
      .sort((a, b) => b.score - a.score)

    return [...exactRows, ...nearRows]
  }, [duplicatesData])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="relative w-full max-w-3xl max-h-[90vh] bg-white dark:bg-slate-800 rounded-2xl shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-slate-800 dark:to-slate-700">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-lg">
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M7 7h10M7 12h10M7 17h6" />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Similar pages</h2>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Exact similar matches are listed first; lower scores for the same pages are hidden.
              </p>
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

        <div className="overflow-auto max-h-[calc(90vh-120px)] p-6">
          {duplicatesFetching && (
            <div className="flex flex-col items-center justify-center gap-4 py-16">
              <div className="h-12 w-12 rounded-full border-4 border-indigo-200 border-t-indigo-600 animate-spin"></div>
              <p className="text-gray-600 dark:text-gray-400">Loading similar pages...</p>
            </div>
          )}

          {duplicatesError && !duplicatesFetching && (
            <div className="flex flex-col items-center gap-4 py-16 px-6 text-center">
              <p className="text-lg font-medium text-red-600 dark:text-red-400">Failed to load similar pages</p>
              <button
                onClick={refetchDuplicates}
                className="mt-2 rounded-lg bg-red-600 px-4 py-2 text-white hover:bg-red-700 transition-colors"
              >
                Retry
              </button>
            </div>
          )}

          {!duplicatesFetching && !duplicatesError && filteredDuplicates.length > 0 && (
            <div className="space-y-3">
              {filteredDuplicates.map((dup, index) => {
                const [p1, p2] = dup.pages
                const isExact = dup.score === 1
                return (
                  <div
                    key={`${p1}-${p2}-${index}`}
                    className="rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 p-4"
                  >
                    <p className="text-sm text-gray-800 dark:text-gray-200">
                      Pages <span className="font-semibold">{p1}</span> &amp; <span className="font-semibold">{p2}</span>{' '}
                      {isExact ? (
                        <span className="text-emerald-700 dark:text-emerald-400 font-medium">are completely similar (100.00%)</span>
                      ) : (
                        <span className="text-indigo-700 dark:text-indigo-400 font-medium">are similar ({formatPercent(dup.score)})</span>
                      )}
                    </p>
                  </div>
                )
              })}
            </div>
          )}

          {!duplicatesFetching && !duplicatesError && filteredDuplicates.length === 0 && (
            <div className="flex flex-col items-center gap-4 py-16 text-center">
              <p className="text-lg font-medium text-gray-900 dark:text-white">No similar pages found</p>
              <p className="text-gray-600 dark:text-gray-400">No similar or near-similar page pairs are available for this document.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
