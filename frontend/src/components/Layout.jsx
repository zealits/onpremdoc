import { Outlet, Link, useParams } from 'react-router-dom'
import { useDocuments } from '../api/hooks'

export default function Layout() {
  return (
    <div className="flex h-screen bg-gray-50">
      <aside className="w-56 flex-shrink-0 border-r border-gray-200 bg-white flex flex-col">
        <div className="p-4 border-b border-gray-100">
          <Link to="/" className="font-semibold text-gray-800 hover:text-indigo-600">
            Chat Document
          </Link>
        </div>
        <nav className="p-2">
          <Link
            to="/"
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-indigo-600 bg-indigo-50 font-medium"
          >
            + New
          </Link>
        </nav>
        <div className="flex-1 overflow-auto px-2">
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mt-4 mb-2 px-2">
            Chats
          </p>
          <DocumentList />
        </div>
      </aside>
      <main className="flex-1 overflow-hidden flex flex-col min-w-0">
        <Outlet />
      </main>
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
      <div className="text-sm text-gray-500 px-2 py-1">
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
            className={`block px-3 py-2 rounded-lg text-sm truncate ${
              doc.document_id === documentId
                ? 'bg-indigo-100 text-indigo-800 font-medium'
                : 'text-gray-700 hover:bg-gray-100'
            }`}
          >
            {doc.name || doc.document_id}
          </Link>
        </li>
      ))}
    </ul>
  )
}
