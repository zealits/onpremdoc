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

const AUTH_TOKEN_KEY = 'doconprem-auth-token'

export function setAuthToken(token) {
  try {
    if (token) {
      window.localStorage.setItem(AUTH_TOKEN_KEY, token)
    } else {
      window.localStorage.removeItem(AUTH_TOKEN_KEY)
    }
  } catch {
    // ignore
  }
}

export function getAuthToken() {
  try {
    return window.localStorage.getItem(AUTH_TOKEN_KEY) || null
  } catch {
    return null
  }
}

export function clearAuthToken() {
  setAuthToken(null)
}

async function request(path, options = {}) {
  const url = `${base()}${path}`
  const token = getAuthToken()
  const res = await fetch(url, {
    ...options,
    headers: {
      ...(options.headers || {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
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

export function getMarkdownUrl(documentId) {
  return `${base()}/documents/${documentId}/markdown`
}

export async function getMarkdown(documentId) {
  const url = getMarkdownUrl(documentId)
  const token = getAuthToken()
  const res = await fetch(url, {
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  })
  if (!res.ok) {
    const text = await res.text()
    let message = text
    try {
      const j = JSON.parse(text)
      message = j.detail || j.error || text
    } catch (_) {}
    throw new Error(message || `HTTP ${res.status}`)
  }
  return res.text()
}

export async function listDocuments() {
  return request('/documents')
}

export async function getDocument(documentId) {
  return request(`/documents/${documentId}`)
}

export async function getDocumentSummary(documentId) {
  return request(`/documents/${documentId}/summary`)
}

export async function getDocumentChunks(documentId, chunkIndices) {
  return request(`/documents/${documentId}/chunks`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chunk_indices: chunkIndices || [] }),
  })
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

export async function queryDocument(documentId, query, sessionId = null, includeChunks = true) {
  return request('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_id: documentId, query, include_chunks: includeChunks, session_id: sessionId }),
  })
}

export async function queryDocumentStream(documentId, query, sessionId = null, includeChunks = true, onChunk) {
  const url = `${base()}/query`
  const token = getAuthToken()
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ document_id: documentId, query, include_chunks: includeChunks, session_id: sessionId }),
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

  if (!res.body) {
    // Fallback: no streaming support, just parse JSON once
    const full = await res.json()
    if (typeof onChunk === 'function') {
      onChunk({ type: 'meta', ...full })
      onChunk({ type: 'answer_chunk', delta: full.answer || '' })
      onChunk({ type: 'done' })
    }
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let newlineIndex
    // Process complete JSON lines
    while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, newlineIndex).trim()
      buffer = buffer.slice(newlineIndex + 1)
      if (!line) continue
      try {
        const data = JSON.parse(line)
        // Use warn level so it still shows when "Info" logs are hidden in DevTools.
        // eslint-disable-next-line no-console
        console.warn('query stream chunk:', data)
        if (typeof onChunk === 'function') {
          onChunk(data)
        }
      } catch (e) {
        // Skip malformed lines but do not break the whole stream
        // eslint-disable-next-line no-console
        console.warn('Failed to parse streaming chunk', e)
      }
    }
  }

  // Flush any remaining buffered line
  const remaining = buffer.trim()
  if (remaining) {
    try {
      const data = JSON.parse(remaining)
      // eslint-disable-next-line no-console
      console.warn('query stream final chunk:', data)
      if (typeof onChunk === 'function') {
        onChunk(data)
      }
    } catch (_) {
      // ignore
    }
  }
}

export async function signup(email, password) {
  return request('/auth/signup', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
}

export async function login(email, password) {
  return request('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
}

export async function getCurrentUser() {
  return request('/auth/me')
}

export async function listChatSessions(documentId) {
  const query = documentId ? `?document_id=${encodeURIComponent(documentId)}` : ''
  const sessions = await request(`/chat/sessions${query}`)
  const sectionIds = Array.isArray(sessions) ? sessions.map((session) => session?.id).filter((id) => id != null) : []
  // eslint-disable-next-line no-console
  console.log('Chat session ids:', sectionIds)
  return sessions
}

export async function createChatSession(documentId, title = null) {
  return request('/chat/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_id: documentId, title }),
  })
}

export async function getSessionMessages(sessionId) {
  return request(`/chat/sessions/${sessionId}/messages`)
}

export async function deleteDocument(documentId) {
  return request(`/documents/${documentId}`, {
    method: 'DELETE',
  })
}

export async function searchDocument(documentId, query, limit = 15) {
  return request(`/documents/${documentId}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: query.trim(), limit }),
  })
}

export async function extractFromDocument(documentId, extractType = 'key_facts') {
  return request(`/documents/${documentId}/extract`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ extract_type: extractType }),
  })
}

export async function emailDocumentSummary(documentId, toEmail, subject = null) {
  return request(`/documents/${documentId}/email`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ to_email: toEmail, subject: subject || undefined }),
  })
}

export async function getEconomicsPipeline(documentId) {
  return request(`/economics/pipeline/${documentId}`)
}

export async function getDocumentPageRanges(documentId) {
  return request(`/documents/${documentId}/page_ranges`)
}

export async function getDocumentDuplicates(documentId) {
  return request(`/documents/${documentId}/duplicates`)
}

export async function getQueryEconomics(documentId, sessionId = null) {
  const params = new URLSearchParams()
  if (sessionId != null) {
    params.set('session_id', String(sessionId))
  }
  if (documentId) {
    params.set('document_id', documentId)
  }
  const qs = params.toString() ? `?${params.toString()}` : ''
  return request(`/economics/queries${qs}`)
}
