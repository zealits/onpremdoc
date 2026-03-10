import {
  useQuery,
  useMutation,
  useQueryClient,
} from '@tanstack/react-query'
import {
  listDocuments,
  getDocument,
  getDocumentSummary,
  getMarkdown,
  uploadPdf,
  vectorize,
  queryDocument,
  listChatSessions,
  createChatSession,
  getSessionMessages,
  deleteDocument as deleteDocumentApi,
  searchDocument,
  extractFromDocument,
  emailDocumentSummary,
  getEconomicsPipeline,
  getDocumentPageRanges,
} from './client'

export const documentKeys = {
  all: ['documents'],
  list: () => [...documentKeys.all, 'list'],
  detail: (id) => [...documentKeys.all, id],
  markdown: (id) => [...documentKeys.all, id, 'markdown'],
}

export function useDocuments() {
  return useQuery({
    queryKey: documentKeys.list(),
    queryFn: listDocuments,
  })
}

export function useMarkdown(documentId) {
  return useQuery({
    queryKey: documentKeys.markdown(documentId),
    queryFn: () => getMarkdown(documentId),
    enabled: !!documentId,
  })
}

export function useDocument(documentId, options = {}) {
  return useQuery({
    queryKey: documentKeys.detail(documentId),
    queryFn: () => getDocument(documentId),
    enabled: !!documentId,
    refetchInterval: (data) => {
      if (!data) return 2000
      if (data.status === 'ready' || data.status === 'uploaded') return false
      return 2000
    },
    ...options,
  })
}

export function useDocumentSummary(documentId, options = {}) {
  return useQuery({
    queryKey: [...documentKeys.detail(documentId), 'summary'],
    queryFn: () => getDocumentSummary(documentId),
    enabled: !!documentId && !!(options.enabled !== false),
    staleTime: 5 * 60 * 1000,
    retry: 2,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 5000),
    ...options,
  })
}

export function useUploadPdf() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (file) => uploadPdf(file),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: documentKeys.list() })
    },
  })
}

export function useVectorize(documentId) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => vectorize(documentId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: documentKeys.detail(documentId) })
      qc.invalidateQueries({ queryKey: documentKeys.list() })
    },
  })
}

export function useQueryDocument(documentId) {
  return useMutation({
    mutationFn: ({ query, sessionId }) => queryDocument(documentId, query, sessionId),
  })
}

export function useChatSessions(documentId) {
  return useQuery({
    queryKey: ['chat', 'sessions', documentId],
    queryFn: () => listChatSessions(documentId),
    enabled: !!documentId,
  })
}

export function useCreateChatSession(documentId) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (title) => createChatSession(documentId, title),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['chat', 'sessions', documentId] })
    },
  })
}

export function useSessionMessages(sessionId) {
  return useQuery({
    queryKey: ['chat', 'sessions', sessionId, 'messages'],
    queryFn: () => getSessionMessages(sessionId),
    enabled: !!sessionId,
  })
}

export function useDeleteDocument(documentId) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => deleteDocumentApi(documentId),
    onSuccess: () => {
      // Optimistically remove the deleted document from the cached list
      qc.setQueryData(documentKeys.list(), (old) => {
        if (!Array.isArray(old)) return old
        return old.filter((doc) => doc.document_id !== documentId)
      })
      qc.invalidateQueries({ queryKey: documentKeys.list() })
      qc.removeQueries({ queryKey: documentKeys.detail(documentId) })
    },
  })
}

export function useSearchDocument(documentId) {
  return useMutation({
    mutationFn: ({ query, limit }) => searchDocument(documentId, query, limit),
  })
}

export function useExtractFromDocument(documentId) {
  return useMutation({
    mutationFn: (extractType) => extractFromDocument(documentId, extractType),
  })
}

export function useEmailDocumentSummary(documentId) {
  return useMutation({
    mutationFn: ({ toEmail, subject }) => emailDocumentSummary(documentId, toEmail, subject),
  })
}

export function useEconomicsPipeline(documentId) {
  return useQuery({
    queryKey: [...documentKeys.detail(documentId), 'economics', 'pipeline'],
    queryFn: () => getEconomicsPipeline(documentId),
    enabled: !!documentId,
    staleTime: 5 * 60 * 1000,
  })
}

export function useDocumentPageRanges(documentId, options = {}) {
  return useQuery({
    queryKey: [...documentKeys.detail(documentId), 'page_ranges'],
    queryFn: () => getDocumentPageRanges(documentId),
    enabled: !!documentId && !!(options.enabled !== false),
    staleTime: 5 * 60 * 1000,
    retry: 2,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 5000),
    ...options,
  })
}