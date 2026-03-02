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
      <div className="max-w-lg w-full">
        <h1 className="text-3xl font-semibold tracking-tight text-slate-50 mb-3">
          Chat with your <span className="bg-gradient-to-r from-indigo-400 to-violet-400 bg-clip-text text-transparent">policy, contract, or report</span>
        </h1>
        <p className="text-sm text-slate-300 mb-6 max-w-md">
          Upload a PDF and get precise, citation-backed answers in seconds. No copy‑paste, no manual searching.
        </p>
        <div
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && onChoose()}
          onClick={onChoose}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`relative border border-dashed rounded-2xl p-10 bg-slate-900/70 transition-all ${
            drag ? 'border-indigo-400/80 bg-slate-900/90 shadow-[0_0_0_1px_rgba(129,140,248,0.6),0_18px_45px_rgba(15,23,42,0.7)]' : 'border-slate-700/80 shadow-[0_16px_40px_rgba(15,23,42,0.7)]'
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
            <p className="text-slate-300">Uploading…</p>
          ) : (
            <>
              <p className="text-slate-400 mb-3 text-sm">Drop a PDF here or</p>
              <span className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 text-sm font-medium text-white shadow-sm hover:shadow-md hover:brightness-105">
                Upload
                <span className="text-xs text-indigo-100/90">.pdf up to 20MB</span>
              </span>
            </>
          )}
        </div>
        {upload.isError && (
          <p className="mt-4 text-sm text-rose-400" role="alert">
            {upload.error?.message || 'Upload failed'}
          </p>
        )}
      </div>
    </div>
  )
}
