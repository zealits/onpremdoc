import { useEffect, useState } from 'react'
import { Outlet, Link, useParams } from 'react-router-dom'
import { useDocuments } from '../api/hooks'
import { useAuth } from '../auth/AuthContext'

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
    const stored = window.localStorage.getItem('doconprem-theme')
    if (stored === 'light' || stored === 'dark') return stored
    const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches
    return prefersDark ? 'dark' : 'light'
  } catch {
    return 'dark'
  }
}

function getInitialSidebarState() {
  if (typeof window === 'undefined') return false
  try {
    const stored = window.localStorage.getItem('doconprem-sidebar-collapsed')
    return stored === 'true'
  } catch {
    return false
  }
}

export default function Layout() {
  const [theme, setTheme] = useState(getInitialTheme)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(getInitialSidebarState)
  const { user, logout } = useAuth()

  useEffect(() => {
    if (typeof document === 'undefined') return
    document.documentElement.dataset.theme = theme
    try {
      window.localStorage.setItem('doconprem-theme', theme)
    } catch {
      // ignore
    }
  }, [theme])

  useEffect(() => {
    try {
      window.localStorage.setItem('doconprem-sidebar-collapsed', String(sidebarCollapsed))
    } catch {
      // ignore
    }
  }, [sidebarCollapsed])

  const toggleTheme = () => {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'))
  }

  const toggleSidebar = () => {
    setSidebarCollapsed((prev) => !prev)
  }

  return (
    <div className="h-screen theme-shell overflow-hidden">
      <div className="flex h-full w-full">
        {/* Sidebar */}
        <aside className={`${
          sidebarCollapsed ? 'w-16' : 'w-80'
        } flex-shrink-0 border-r theme-sidebar backdrop-blur-xl flex flex-col transition-all duration-300 ease-in-out`}>
          <div className="px-4 py-4 border-b theme-sidebar">
            <div className="flex flex-col gap-3">
              <div className="flex items-center justify-between gap-2 min-w-0">
                {!sidebarCollapsed ? (
                  <Link to="/" className="flex items-center gap-3 group min-w-0 flex-shrink-0">
                    <span className="inline-flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 text-xs font-semibold text-white shadow-sm group-hover:shadow-md transition-shadow">
                      DoP
                    </span>
                    <span className="flex flex-col min-w-0">
                      <span className="text-sm font-semibold tracking-tight group-hover:text-indigo-500 truncate">
                        DocOnPrem
                      </span>
                      <span className="text-[11px] theme-sidebar-muted truncate">
                        Ask questions over any PDF
                      </span>
                    </span>
                  </Link>
                ) : (
                  <div className="w-full flex justify-center">
                    <Link to="/" className="flex items-center justify-center group flex-shrink-0">
                      <span className="inline-flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 text-xs font-semibold text-white shadow-sm group-hover:shadow-md transition-shadow">
                        DoP
                      </span>
                    </Link>
                  </div>
                )}
                {!sidebarCollapsed && (
                  <button
                    type="button"
                    onClick={toggleSidebar}
                    className={`inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg border transition-all duration-200 ${
                      theme === 'dark'
                        ? 'border-slate-600 bg-slate-800/60 text-slate-300 hover:bg-slate-700/80 hover:text-slate-200'
                        : 'border-slate-300 bg-white text-slate-600 hover:bg-slate-100 hover:text-slate-800'
                    }`}
                    aria-label="Collapse sidebar"
                    title="Collapse sidebar"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                  </button>
                )}
              </div>
              {user && !sidebarCollapsed && (
                <p
                  className="text-[11px] theme-sidebar-muted truncate min-w-0 w-full"
                  title={user.email}
                >
                  {user.email}
                </p>
              )}
              {sidebarCollapsed && (
                <div className="w-full flex justify-center">
                  <button
                    type="button"
                    onClick={toggleSidebar}
                    className={`inline-flex h-6 w-6 items-center justify-center rounded border transition-all duration-200 ${
                      theme === 'dark'
                        ? 'border-slate-600 bg-slate-800/60 text-slate-400 hover:bg-slate-700/80 hover:text-slate-200'
                        : 'border-slate-300 bg-white text-slate-500 hover:bg-slate-100 hover:text-slate-700'
                    }`}
                    aria-label="Expand sidebar"
                    title="Expand sidebar"
                  >
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </button>
                </div>
              )}
            </div>
          </div>
          {!sidebarCollapsed ? (
            <>
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
                <DocumentList collapsed={false} theme={theme} />
              </div>
              <div className="px-3 py-3 border-t theme-sidebar flex-shrink-0">
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={toggleTheme}
                    className={`inline-flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg border transition-all duration-200 ${
                      theme === 'dark'
                        ? 'border-slate-600 bg-slate-800/60 text-amber-300 hover:bg-slate-700/80 hover:text-amber-200'
                        : 'border-slate-300 bg-white text-amber-600 hover:bg-slate-100 hover:text-amber-700'
                    }`}
                    aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                    title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
                  >
                    {theme === 'dark' ? (
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20" aria-hidden>
                        <path d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20" aria-hidden>
                        <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                      </svg>
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={logout}
                    className={`sidebar-logout-btn flex-1 inline-flex items-center justify-center gap-2 h-9 rounded-lg border transition-colors ${
                      theme === 'dark'
                        ? 'border-slate-600 text-slate-300 hover:border-red-400/60 hover:bg-red-500/10 hover:text-red-200'
                        : 'border-slate-300 text-slate-600 hover:border-red-400/60 hover:bg-red-500/10 hover:text-red-600'
                    }`}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                    </svg>
                    Logout
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex flex-col py-4 gap-3">
              <div className="flex justify-center">
                <Link
                  to="/"
                  className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 text-white shadow-sm hover:shadow-md hover:brightness-105 transition-all"
                  title="New chat"
                >
                  <span className="text-lg leading-none">+</span>
                </Link>
              </div>
              <div className="flex-1 overflow-auto">
                <DocumentList collapsed={true} theme={theme} />
              </div>
              <div className="px-2 flex justify-center gap-1">
                <button
                  type="button"
                  onClick={toggleTheme}
                  className={`inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg border transition-all duration-200 ${
                    theme === 'dark'
                      ? 'border-slate-600 bg-slate-800/60 text-amber-300 hover:bg-slate-700/80 hover:text-amber-200'
                      : 'border-slate-300 bg-white text-amber-600 hover:bg-slate-100 hover:text-amber-700'
                  }`}
                  aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                  title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
                >
                  {theme === 'dark' ? (
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20" aria-hidden>
                      <path d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20" aria-hidden>
                      <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                    </svg>
                  )}
                </button>
                <button
                  type="button"
                  onClick={logout}
                  className={`inline-flex h-8 w-8 items-center justify-center rounded-lg border transition-colors ${
                    theme === 'dark'
                      ? 'border-slate-600 text-slate-300 hover:border-red-400/60 hover:bg-red-500/10 hover:text-red-200'
                      : 'border-slate-300 text-slate-600 hover:border-red-400/60 hover:bg-red-500/10 hover:text-red-600'
                  }`}
                  title="Logout"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                  </svg>
                </button>
              </div>
            </div>
          )}
        </aside>
        <main className="flex-1 overflow-hidden flex flex-col min-w-0 theme-main">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

function DocumentList({ collapsed = false, theme = 'dark' }) {
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
  
  if (collapsed) {
    // Show only indicators for collapsed sidebar with center alignment
    return (
      <div className="flex flex-col items-center">
        <ul className="space-y-2">
          {documents.slice(0, 5).map((doc) => (
            <li key={doc.document_id} className="flex justify-center">
              <Link
                to={`/documents/${doc.document_id}`}
                className={`w-8 h-8 rounded-lg transition-colors flex items-center justify-center ${
                  doc.document_id === documentId 
                    ? 'bg-indigo-500 text-white shadow-sm' 
                    : theme === 'dark'
                      ? 'bg-slate-700/50 text-slate-400 hover:bg-slate-600/50 hover:text-slate-300'
                      : 'bg-slate-200/70 text-slate-600 hover:bg-slate-300/80 hover:text-slate-800'
                }`}
                title={getDocumentDisplayName(doc)}
              >
                <span className="text-xs font-medium">
                  {getDocumentDisplayName(doc).charAt(0).toUpperCase()}
                </span>
              </Link>
            </li>
          ))}
          {documents.length > 5 && (
            <li className="flex justify-center">
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-xs ${
                theme === 'dark'
                  ? 'bg-slate-700/30 text-slate-500'
                  : 'bg-slate-200/50 text-slate-500'
              }`}>
                +{documents.length - 5}
              </div>
            </li>
          )}
        </ul>
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
