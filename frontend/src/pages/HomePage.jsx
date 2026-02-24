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
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="text-center max-w-lg w-full">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">
          Chat with your PDF
        </h1>
        <p className="text-gray-600 mb-6">
          Drop your PDF here or upload to get started.
        </p>
        <div
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && onChoose()}
          onClick={onChoose}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`border-2 border-dashed rounded-xl p-12 bg-white transition-colors ${
            drag ? 'border-indigo-400 bg-indigo-50' : 'border-gray-300'
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
            <p className="text-gray-600">Uploading…</p>
          ) : (
            <>
              <p className="text-gray-500 mb-2">Drop a file or</p>
              <span className="inline-block px-4 py-2 rounded-lg bg-indigo-600 text-white font-medium">
                Upload
              </span>
            </>
          )}
        </div>
        {upload.isError && (
          <p className="mt-4 text-sm text-red-600" role="alert">
            {upload.error?.message || 'Upload failed'}
          </p>
        )}
      </div>
    </div>
  )
}
