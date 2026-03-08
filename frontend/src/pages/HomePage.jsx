import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useUploadPdf } from '../api/hooks'

const FEATURES = [
  {
    id: 'index',
    title: 'Index document',
    description: 'Upload PDFs to extract text and build a searchable index.',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
      </svg>
    ),
  },
  {
    id: 'search',
    title: 'Search in document',
    description: 'Semantic search across your PDF—find relevant passages instantly.',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
  },
  {
    id: 'summarize',
    title: 'Summarize document',
    description: 'Get a concise overview and key points from any document.',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
  },
  {
    id: 'extract',
    title: 'Extract information',
    description: 'Pull out key facts, entities, dates, and obligations automatically.',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
      </svg>
    ),
  },
  {
    id: 'chat',
    title: 'Chat with document',
    description: 'Ask questions in plain language and get citation-backed answers.',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
      </svg>
    ),
  },
  {
    id: 'email',
    title: 'Email with summary',
    description: 'Send a document summary to any email address.',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
  },
]

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
    <div className="flex-1 flex flex-col items-center px-4 sm:px-6 py-8 sm:py-12 overflow-y-auto">
      <div className="w-full max-w-5xl space-y-10">
        {/* Hero */}
        <div className="text-center space-y-4">
          <div className="inline-flex items-center gap-2 rounded-full border theme-card px-3 py-1.5 text-xs font-medium text-slate-400 shadow-sm">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            AI answers grounded in your own PDFs
          </div>
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-balance">
            Chat with your{' '}
            <span className="bg-gradient-to-r from-indigo-400 via-violet-400 to-purple-500 bg-clip-text text-transparent">
              policy, contract, or report
            </span>
          </h1>
          <p className="text-sm sm:text-base max-w-2xl mx-auto text-slate-400 leading-relaxed">
            Upload a PDF and get precise, citation-backed answers in seconds. Search, summarize, extract key information, and follow citations back into the original pages.
          </p>
        </div>

        {/* Upload zone */}
        <div
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && onChoose()}
          onClick={onChoose}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`relative border-2 border-dashed rounded-2xl p-8 sm:p-12 theme-card transition-all duration-200 ${
            drag ? 'border-indigo-400/80 bg-indigo-500/5' : 'border-slate-400/40'
          } ${upload.isPending ? 'pointer-events-none opacity-70' : 'cursor-pointer hover:border-indigo-400/50'}`}
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
            <div className="flex flex-col items-center justify-center gap-4 text-slate-400">
              <div className="w-10 h-10 rounded-xl border-2 border-indigo-400/60 border-t-transparent animate-spin" />
              <p className="text-sm font-medium">Uploading and indexing your PDF…</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center gap-5 text-center">
              <div className="upload-icon-box inline-flex h-14 w-14 items-center justify-center rounded-2xl border-2 theme-card text-indigo-400">
                <svg className="h-7 w-7" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 16V4M12 4L8 8M12 4L16 8M5 16V18.5C5 19.8807 6.11929 21 7.5 21H16.5C17.8807 21 19 19.8807 19 18.5V16" />
                </svg>
              </div>
              <div className="space-y-1">
                <p className="upload-title text-base sm:text-lg font-semibold">Drop a PDF to start</p>
                <p className="upload-subtitle text-xs sm:text-sm">We extract text locally and keep citations aligned. Then search, summarize, extract, and chat.</p>
              </div>
              <div className="flex flex-wrap items-center justify-center gap-2">
                <span className="upload-helper text-sm">Drop file here or</span>
                <span className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 text-sm font-medium text-white shadow-md hover:shadow-lg hover:brightness-105 transition-all">
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

        {/* Feature grid */}
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-widest text-slate-500 mb-4">What you can do</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {FEATURES.map((f) => (
              <div
                key={f.id}
                className="rounded-xl border theme-card p-4 hover:border-indigo-400/30 transition-colors"
              >
                <div className="flex gap-3">
                  <span className="flex-shrink-0 w-10 h-10 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center">
                    {f.icon}
                  </span>
                  <div className="min-w-0">
                    <h3 className="font-semibold text-sm text-inherit">{f.title}</h3>
                    <p className="text-xs theme-sidebar-muted mt-0.5">{f.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
