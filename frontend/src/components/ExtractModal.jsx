import { useState } from 'react'

const EXTRACT_TYPES = [
  { value: 'key_facts', label: 'Key facts' },
  { value: 'entities', label: 'Entities (people, orgs, places)' },
  { value: 'dates', label: 'Dates & deadlines' },
  { value: 'obligations', label: 'Obligations & requirements' },
]

export default function ExtractModal({ documentId, documentName, extractMutation, onClose }) {
  const [extractType, setExtractType] = useState('key_facts')
  const [result, setResult] = useState(null)

  const runExtract = () => {
    setResult(null)
    extractMutation.mutate(extractType, {
      onSuccess: (data) => {
        setResult(data)
      },
    })
  }

  const items = result?.items ?? []
  const error = result?.error

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50" role="dialog" aria-modal="true" aria-labelledby="extract-modal-title">
      <div className="theme-card rounded-2xl shadow-2xl w-full max-w-lg max-h-[85vh] flex flex-col border border-slate-600">
        <div className="flex items-center justify-between px-5 py-4 border-b theme-sidebar">
          <h2 id="extract-modal-title" className="text-lg font-semibold">Extract information</h2>
          <button type="button" onClick={onClose} className="p-1 rounded-lg text-slate-500 hover:bg-slate-700 hover:text-slate-200" aria-label="Close">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="px-5 py-4 space-y-4 flex-1 min-h-0 flex flex-col">
          <p className="text-sm text-slate-400">From: {documentName || documentId}</p>
          <div>
            <label className="block text-sm font-medium mb-2">Type</label>
            <select
              value={extractType}
              onChange={(e) => setExtractType(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-600 text-slate-100 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-400"
            >
              {EXTRACT_TYPES.map((t) => (
                <option key={t.value} value={t.value}>{t.label}</option>
              ))}
            </select>
          </div>
          <button
            type="button"
            onClick={runExtract}
            disabled={extractMutation.isPending}
            className="w-full py-2.5 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {extractMutation.isPending ? 'Extracting…' : 'Extract'}
          </button>
          {error && <p className="text-sm text-rose-400">{error}</p>}
          {result && !error && (
            <div className="flex-1 min-h-0 overflow-y-auto">
              <h3 className="text-sm font-medium mb-2">Results ({items.length})</h3>
              <ul className="space-y-1.5 column-scroll">
                {items.length === 0 ? (
                  <li className="text-sm text-slate-500">No items extracted.</li>
                ) : (
                  items.map((item, i) => (
                    <li key={i} className="text-sm px-3 py-2 rounded-lg bg-slate-800/60 border border-slate-700/50">
                      {item}
                    </li>
                  ))
                )}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
