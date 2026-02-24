/**
 * API client for FastAPI backend.
 * Base URL: VITE_API_URL or /api (Vite proxy in dev).
 */

const getBaseUrl = () => {
  const env = import.meta.env.VITE_API_URL
  if (env) return env.replace(/\/$/, '')
  return '/api' // Vite proxy in dev: /api -> http://127.0.0.1:8000
}

const base = () => getBaseUrl()

async function request(path, options = {}) {
  const url = `${base()}${path}`
  const res = await fetch(url, {
    ...options,
    headers: {
      ...(options.headers || {}),
    },
  })
  if (!res.ok) {
    const text = await res.text()
    let detail = text
    try {
      const j = JSON.parse(text)
      detail = j.detail || (typeof j.detail === 'string' ? j.detail : text)
    } catch (_) {}
    throw new Error(detail || `HTTP ${res.status}`)
  }
  const contentType = res.headers.get('content-type') || ''
  if (contentType.includes('application/json')) return res.json()
  return res
}

export function getPdfUrl(documentId) {
  return `${base()}/documents/${documentId}/pdf`
}

export async function listDocuments() {
  return request('/documents')
}

export async function getDocument(documentId) {
  return request(`/documents/${documentId}`)
}

export async function uploadPdf(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await request('/upload', {
    method: 'POST',
    body: form,
  })
  return res
}

export async function vectorize(documentId) {
  return request(`/vectorize/${documentId}`, { method: 'POST' })
}

export async function queryDocument(documentId, query, includeChunks = true) {
  return request('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_id: documentId, query, include_chunks: includeChunks }),
  })
}
