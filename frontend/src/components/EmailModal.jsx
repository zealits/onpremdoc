import { useState } from 'react'

export default function EmailModal({ documentName, emailMutation, onClose }) {
  const [toEmail, setToEmail] = useState('')
  const [subject, setSubject] = useState('')
  const [sent, setSent] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = (e) => {
    e.preventDefault()
    setError(null)
    const to = toEmail.trim()
    if (!to || !to.includes('@')) {
      setError('Please enter a valid email address.')
      return
    }
    emailMutation.mutate(
      { toEmail: to, subject: subject.trim() || null },
      {
        onSuccess: (data) => {
          setSent(true)
          if (data?.message) setError(null)
        },
        onError: (err) => {
          setError(err?.message || 'Failed to send email.')
        },
      }
    )
  }

  if (sent && !emailMutation.isError) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50" role="dialog" aria-modal="true">
        <div className="theme-card rounded-2xl shadow-2xl w-full max-w-md p-6 border border-slate-600">
          <div className="text-center space-y-4">
            <div className="w-12 h-12 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center mx-auto">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h2 className="text-lg font-semibold">Email sent</h2>
            <p className="text-sm text-slate-400">{emailMutation.data?.message || 'Summary has been sent to the recipient.'}</p>
            <button
              type="button"
              onClick={onClose}
              className="w-full py-2.5 rounded-xl bg-slate-700 text-slate-200 font-medium hover:bg-slate-600"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50" role="dialog" aria-modal="true" aria-labelledby="email-modal-title">
      <div className="theme-card rounded-2xl shadow-2xl w-full max-w-md border border-slate-600">
        <div className="flex items-center justify-between px-5 py-4 border-b theme-sidebar">
          <h2 id="email-modal-title" className="text-lg font-semibold">Email summary</h2>
          <button type="button" onClick={onClose} className="p-1 rounded-lg text-slate-500 hover:bg-slate-700 hover:text-slate-200" aria-label="Close">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          <p className="text-sm text-slate-400">Send a summary of “{documentName}” to:</p>
          <div>
            <label className="block text-sm font-medium mb-1">Recipient email *</label>
            <input
              type="email"
              value={toEmail}
              onChange={(e) => setToEmail(e.target.value)}
              placeholder="name@example.com"
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-600 text-slate-100 placeholder:text-slate-500 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-400"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Subject (optional)</label>
            <input
              type="text"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              placeholder={`Summary: ${documentName || 'Document'}`}
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-600 text-slate-100 placeholder:text-slate-500 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-400"
            />
          </div>
          {error && <p className="text-sm text-rose-400">{error}</p>}
          <div className="flex gap-2">
            <button type="button" onClick={onClose} className="flex-1 py-2.5 rounded-xl border border-slate-600 text-slate-300 font-medium hover:bg-slate-700">
              Cancel
            </button>
            <button type="submit" disabled={emailMutation.isPending} className="flex-1 py-2.5 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-500 disabled:opacity-50">
              {emailMutation.isPending ? 'Sending…' : 'Send'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
