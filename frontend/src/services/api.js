const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

/**
 * API service for communicating with the FastAPI backend
 */

/**
 * List all documents
 * @returns {Promise<Array>} List of document info objects
 */
export async function listDocuments() {
  const response = await fetch(`${API_BASE_URL}/documents`)
  if (!response.ok) {
    throw new Error(`Failed to fetch documents: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Get document information by ID
 * @param {string} documentId - Document ID
 * @returns {Promise<Object>} Document info object
 */
export async function getDocument(documentId) {
  const response = await fetch(`${API_BASE_URL}/documents/${documentId}`)
  if (!response.ok) {
    throw new Error(`Failed to fetch document: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Upload a PDF file
 * @param {File} file - PDF file to upload
 * @returns {Promise<Object>} Upload response with document_id
 */
export async function uploadPDF(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: response.statusText }))
    throw new Error(error.error || error.detail || `Upload failed: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Trigger vectorization for a document
 * @param {string} documentId - Document ID
 * @returns {Promise<Object>} Vectorization response
 */
export async function vectorizeDocument(documentId) {
  const response = await fetch(`${API_BASE_URL}/vectorize/${documentId}`, {
    method: 'POST',
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: response.statusText }))
    throw new Error(error.error || error.detail || `Vectorization failed: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Query a document using the agentic chat system
 * @param {string} documentId - Document ID
 * @param {string} query - Query text
 * @returns {Promise<Object>} Query response with answer and chunks
 */
export async function queryDocument(documentId, query) {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      document_id: documentId,
      query: query,
      include_chunks: false, // Set to true if you want chunk details
    }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: response.statusText }))
    throw new Error(error.error || error.detail || `Query failed: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Get Docling confidence / accuracy scores for a document
 * @param {string} documentId - Document ID
 * @returns {Promise<Object>} Confidence response with document- and page-level scores
 */
export async function getConfidence(documentId) {
  const response = await fetch(`${API_BASE_URL}/documents/${documentId}/confidence`)

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: response.statusText }))
    throw new Error(error.error || error.detail || `Failed to fetch confidence: ${response.statusText}`)
  }

  return response.json()
}
