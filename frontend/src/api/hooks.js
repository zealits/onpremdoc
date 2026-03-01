import {
  useQuery,
  useMutation,
  useQueryClient,
} from '@tanstack/react-query'
import {
  listDocuments,
  getDocument,
  getMarkdown,
  uploadPdf,
  vectorize,
  queryDocument,
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
    mutationFn: ({ query }) => queryDocument(documentId, query),
  })
}

export function useMarkdown(documentId) {
  return useQuery({
    queryKey: documentKeys.markdown(documentId),
    queryFn: () => getMarkdown(documentId),
    enabled: !!documentId,
  })
}
