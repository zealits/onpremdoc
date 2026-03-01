import { useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { useMarkdown } from '../api/hooks'
import { getMarkdownUrl } from '../api/client'

function findTextAndScroll(container, searchText, sectionTitle, heading) {
  if (!container) return
  const candidates = []
  if (typeof searchText === 'string' && searchText.trim().length >= 4) {
    const normalized = searchText.replace(/\s+/g, ' ').trim()
    for (const len of [80, 50, 30]) {
      const s = normalized.slice(0, len).trim()
      if (s && !candidates.includes(s)) candidates.push(s)
    }
  }
  const section = typeof sectionTitle === 'string' && sectionTitle.trim() && sectionTitle !== 'No heading' ? sectionTitle.trim() : ''
  const head = typeof heading === 'string' && heading.trim() && heading !== 'No heading' ? heading.trim() : ''
  if (section && !candidates.includes(section)) candidates.push(section)
  if (head && !candidates.includes(head)) candidates.push(head)
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT)
  let node
  while ((node = walker.nextNode())) {
    const text = node.textContent || ''
    for (const candidate of candidates) {
      if (candidate && text.includes(candidate)) {
        node.parentElement?.scrollIntoView({ behavior: 'smooth', block: 'center' })
        return
      }
    }
  }
  const toMatch = (section || head).toLowerCase()
  if (toMatch) {
    const allHeadings = container.querySelectorAll('h1, h2, h3, h4, h5, h6')
    for (const el of allHeadings) {
      const t = (el.textContent || '').trim().toLowerCase()
      if (t && (t.includes(toMatch.slice(0, 25)) || toMatch.includes(t.slice(0, 25)))) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
        return
      }
    }
  }
}

export default function MarkdownViewer({ documentId, activeHighlight }) {
  const contentRef = useRef(null)
  const { data: markdown, isLoading, error } = useMarkdown(documentId)

  useEffect(() => {
    if (!activeHighlight || !contentRef.current) return
    findTextAndScroll(
      contentRef.current,
      activeHighlight.text,
      activeHighlight.sectionTitle,
      activeHighlight.heading
    )
  }, [activeHighlight])

  if (!documentId) return null

  if (isLoading) {
    return (
      <div className="flex flex-col h-full min-h-0 bg-gray-100 items-center justify-center p-4">
        <div className="text-gray-500 text-sm">Loading markdown…</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex flex-col h-full min-h-0 bg-gray-100 items-center justify-center p-4">
        <div className="text-red-600 text-sm">{error?.message || 'Failed to load markdown.'}</div>
      </div>
    )
  }

  const markdownUrl = getMarkdownUrl(documentId)

  return (
    <div className="flex flex-col h-full min-h-0 bg-gray-100">
      <div className="flex items-center gap-2 px-2 py-1.5 bg-white border-b border-gray-200 shrink-0">
        <a
          href={markdownUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="ml-auto text-sm text-indigo-600 hover:underline"
        >
          Open in new tab
        </a>
      </div>
      <div
        ref={contentRef}
        className="flex-1 min-h-0 overflow-auto p-4 text-gray-800 [&_h1]:text-2xl [&_h1]:font-semibold [&_h1]:mt-4 [&_h2]:text-xl [&_h2]:font-semibold [&_h2]:mt-3 [&_h3]:text-lg [&_h3]:font-medium [&_h3]:mt-2 [&_p]:my-2 [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-6 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-6 [&_table]:border [&_table]:border-gray-300 [&_th]:border [&_th]:border-gray-300 [&_th]:px-2 [&_th]:py-1 [&_td]:border [&_td]:border-gray-300 [&_td]:px-2 [&_td]:py-1"
      >
        <ReactMarkdown>{markdown ?? ''}</ReactMarkdown>
      </div>
    </div>
  )
}
