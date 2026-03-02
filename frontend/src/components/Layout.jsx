import { useEffect, useState } from 'react'
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

function getInitialTheme() {
  if (typeof window === 'undefined') return 'dark'
  try {
    const stored = window.localStorage.getItem('onpremdoc-theme')
    if (stored === 'light' || stored === 'dark') return stored
    const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches
    return prefersDark ? 'dark' : 'light'
  } catch {
    return 'dark'
  }
}

export default function Layout() {
  const [theme, setTheme] = useState(getInitialTheme)

  useEffect(() => {
    if (typeof document === 'undefined') return
    document.documentElement.dataset.theme = theme
    try {
      window.localStorage.setItem('onpremdoc-theme', theme)
    } catch {
      // ignore
    }
  }, [theme])

  const toggleTheme = () => {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
  }

  return (
    <div className="h-screen theme-shell overflow-hidden">
      <div className="flex h-full w-full">
        <aside className="w-64 flex-shrink-0 border-r theme-sidebar backdrop-blur-xl flex flex-col">
          <div className="px-4 py-4 border-b theme-sidebar">
            <div className="flex items-center justify-between gap-3">
              <Link to="/" className="flex items-center gap-3 group">
                <span className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 text-xs font-semibold text-white shadow-sm group-hover:shadow-md transition-shadow">
                  CD
                </span>
                <span className="flex flex-col">
                  <span className="text-sm font-semibold tracking-tight group-hover:text-indigo-500">
                    Chat Document
                  </span>
                  <span className="text-[11px] theme-sidebar-muted">
                    Ask questions over any PDF
                  </span>
                </span>
              </Link>
              <button
                type="button"
                onClick={toggleTheme}
                className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-slate-700/40 bg-slate-950/40 text-slate-300 hover:border-indigo-400 hover:text-indigo-300 transition-colors text-xs"
                aria-label="Toggle light / dark theme"
              >
                {theme === 'dark' ? '☾' : '☼'}
              </button>
            </div>
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
            <p className="text-[11px] font-medium theme-sidebar-muted uppercase tracking-[0.18em] mt-3 mb-2 px-1">
              Chats
            </p>
            <DocumentList />
          </div>
        </aside>
        <main className="flex-1 overflow-hidden flex flex-col min-w-0 theme-main">
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
            className={`block px-3 py-2.5 rounded-xl text-sm transition-colors chat-list-item ${
              doc.document_id === documentId ? 'chat-list-item-active font-medium shadow-sm' : ''
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
