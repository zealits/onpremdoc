import { Outlet, Link, useParams } from 'react-router-dom'
import { useDocuments } from '../api/hooks'

function getDocumentDisplayName(doc) {
  if (!doc) return ''
  // If backend already sends a nice name, use it
  if (doc.name && doc.name !== doc.document_id) return doc.name
  // Otherwise, try to derive from markdown or pdf path
  const path =
    doc.markdown_path ||
    doc.page_mapping_path ||
    doc.confidence_path ||
    ''
  if (path) {
    const parts = String(path).split(/[\\/]/)
    const file = parts[parts.length - 1] || ''
    const withoutExt = file.replace(/\.(md|pdf)$/i, '')
    if (withoutExt) return withoutExt
  }
  return doc.document_id
}

export default function Layout() {
  return (
    <div className="h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 overflow-hidden">
      <div className="flex h-full w-full">
        <aside className="w-64 flex-shrink-0 border-r border-slate-800 bg-slate-900/85 backdrop-blur-xl flex flex-col">
          <div className="px-4 py-4 border-b border-slate-800">
            <Link to="/" className="flex items-center gap-3 group">
              <span className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 text-xs font-semibold text-white shadow-sm group-hover:shadow-md transition-shadow">
                CD
              </span>
              <span className="flex flex-col">
                <span className="text-sm font-semibold tracking-tight text-slate-50 group-hover:text-white">
                  Chat Document
                </span>
                <span className="text-[11px] text-slate-400">
                  Ask questions over any PDF
                </span>
              </span>
            </Link>
          </div>
          <nav className="px-3 pt-3 pb-1">
            <Link
              to="/"
              className="inline-flex items-center justify-center gap-2 w-full rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 px-3 py-2.5 text-sm font-medium text-white shadow-sm hover:shadow-md hover:brightness-105 transition-all"
            >
              <span className="text-base leading-none">+</span>
              <span>New chat</span>
            </Link>
          </nav>
          <div className="flex-1 overflow-auto px-3 pb-4">
            <p className="text-[11px] font-medium text-slate-400 uppercase tracking-[0.18em] mt-3 mb-2 px-1">
              Chats
            </p>
            <DocumentList />
          </div>
        </aside>
        <main className="flex-1 overflow-hidden flex flex-col min-w-0 bg-slate-900/40">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

function DocumentList() {
  const { documentId } = useParams()
  const { data: documents, isLoading, error } = useDocuments()

  if (isLoading) return <div className="text-sm text-gray-400 px-2 py-1">Loading…</div>
  if (error) return <div className="text-sm text-red-500 px-2 py-1">Failed to load</div>
  if (!documents?.length) {
    return (
      <div className="text-sm text-slate-400 px-2 py-1">
        <p>No documents yet</p>
      </div>
    )
  }
  return (
    <ul className="space-y-0.5">
      {documents.map((doc) => (
        <li key={doc.document_id}>
          <Link
            to={`/documents/${doc.document_id}`}
            className={`block px-3 py-2.5 rounded-xl text-sm transition-colors ${
              doc.document_id === documentId
                ? 'bg-slate-800 text-slate-50 font-medium shadow-sm'
                : 'text-slate-200 hover:bg-slate-800/70'
            }`}
          >
            <div className="flex flex-col">
              <span className="truncate">{getDocumentDisplayName(doc)}</span>
            </div>
          </Link>
        </li>
      ))}
    </ul>
  )
}
