import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useUploadPdf } from '../api/hooks'

export default function HomePage() {
  const inputRef = useRef(null)
  const navigate = useNavigate()
  const upload = useUploadPdf()
  const [drag, setDrag] = useState(false)

  const handleFile = async (file) => {
    if (!file?.name?.toLowerCase().endsWith('.pdf')) {
      return
    }
    try {
      const res = await upload.mutateAsync(file)
      navigate(`/documents/${res.document_id}`)
    } catch (_) {}
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDrag(false)
    const file = e.dataTransfer?.files?.[0]
    if (file) handleFile(file)
  }

  const onDragOver = (e) => {
    e.preventDefault()
    setDrag(true)
  }

  const onDragLeave = () => setDrag(false)

  const onChoose = () => inputRef.current?.click()

  const onInputChange = (e) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
    e.target.value = ''
  }

  return (
    <div className="flex-1 flex items-center justify-center px-6 py-10">
      <div className="w-full max-w-5xl grid gap-10 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] items-center">
        <div className="space-y-6">
          <div className="inline-flex items-center gap-2 rounded-full border theme-card px-3 py-1 text-[11px] font-medium text-slate-300 shadow-sm">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
            AI answers grounded in your own PDFs
          </div>
          <div>
            <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight mb-3">
              Chat with your{' '}
              <span className="bg-gradient-to-r from-indigo-400 to-violet-400 bg-clip-text text-transparent">
                policy, contract, or report
              </span>
            </h1>
            <p className="text-sm sm:text-base max-w-xl">
              Upload a PDF and get precise, citation-backed answers in seconds. Follow citations back into the original
              pages whenever you need to double‑check.
            </p>
          </div>
          <dl className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs sm:text-sm">
            <div className="rounded-xl border theme-card px-3 py-3">
              <dt className="mb-1 font-medium">Legal & policy docs</dt>
              <dd className="theme-sidebar-muted">Terms, policies, contracts, and disclosures.</dd>
            </div>
            <div className="rounded-xl border theme-card px-3 py-3">
              <dt className="mb-1 font-medium">Instant answers</dt>
              <dd className="theme-sidebar-muted">Ask questions in plain language, no setup.</dd>
            </div>
            <div className="rounded-xl border theme-card px-3 py-3">
              <dt className="mb-1 font-medium">Cited sources</dt>
              <dd className="theme-sidebar-muted">Every answer links back to your document.</dd>
            </div>
          </dl>
        </div>
        <div
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && onChoose()}
          onClick={onChoose}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`relative border border-dashed rounded-2xl p-8 sm:p-10 theme-card transition-colors ${
            drag ? 'border-indigo-400/80' : 'border-slate-300'
          } ${upload.isPending ? 'pointer-events-none opacity-70' : 'cursor-pointer'}`}
          aria-label="Drop PDF or click to upload"
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,application/pdf"
            onChange={onInputChange}
            className="hidden"
            aria-hidden
          />
          {upload.isPending ? (
            <div className="flex flex-col items-center justify-center gap-3 text-slate-300 text-sm">
              <div className="w-9 h-9 rounded-full border-2 border-indigo-400/60 border-t-transparent animate-spin" />
              <p>Uploading your PDF…</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center gap-4 text-center">
              <div className="upload-icon-box inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-900 border border-slate-700 text-indigo-300">
                <svg
                  className="h-6 w-6"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M12 16V4M12 4L8 8M12 4L16 8M5 16V18.5C5 19.8807 6.11929 21 7.5 21H16.5C17.8807 21 19 19.8807 19 18.5V16"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </div>
              <div className="space-y-1">
                <p className="upload-title text-sm sm:text-base font-medium">Drop a PDF to start chatting</p>
                <p className="upload-subtitle text-xs sm:text-sm">We’ll extract text locally and keep citations aligned.</p>
              </div>
              <div className="flex flex-col sm:flex-row items-center gap-3 w-full sm:w-auto">
                <span className="upload-helper text-xs sm:text-sm">Drop file here or</span>
                <span className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 text-sm font-medium text-white shadow-sm hover:shadow-md hover:brightness-105">
                  Upload PDF
                  <span className="text-[11px] text-indigo-100/90">.pdf up to 20MB</span>
                </span>
              </div>
            </div>
          )}
          {upload.isError && (
            <p className="mt-4 text-sm text-rose-400 text-center" role="alert">
              {upload.error?.message || 'Upload failed'}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
