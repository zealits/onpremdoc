import { useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useMarkdown } from '../api/hooks'
import { getMarkdownUrl } from '../api/client'

/** Remove HTML comments for page break and image so they are not shown in the UI. */
function stripPageBreakAndImageComments(md) {
  if (typeof md !== 'string') return ''
  return md.replace(/^\s*<!--\s*(?:page break|image)\s*-->\s*$/gim, '')
}

function highlightElementAndScroll(el) {
  if (!el) return
  el.scrollIntoView({ behavior: 'smooth', block: 'center' })

  const root = el.closest('[data-markdown-root]') || el.parentElement || el
  if (root) {
    const prev = root.querySelectorAll('.markdown-highlight')
    prev.forEach((node) => {
      // Add removing class for fade-out animation
      node.classList.add('removing')
      // Clear any existing timeouts for previous highlights
      if (node._markdownHighlightTimeout) {
        window.clearTimeout(node._markdownHighlightTimeout)
      }
      if (node._markdownRemovalTimeout) {
        window.clearTimeout(node._markdownRemovalTimeout)
      }
      // Remove highlight after fade-out animation completes
      node._markdownRemovalTimeout = window.setTimeout(() => {
        node.classList.remove('markdown-highlight', 'removing')
      }, 600) // Match the fade-out animation duration
    })
  }

  el.classList.add('markdown-highlight')
  el.classList.remove('removing') // Ensure no removing class on new highlight
  
  // Clear any existing timeouts
  window.clearTimeout(el._markdownHighlightTimeout)
  window.clearTimeout(el._markdownRemovalTimeout)
  
  // Set timeout for automatic removal after 10 seconds
  el._markdownHighlightTimeout = window.setTimeout(() => {
    el.classList.add('removing')
    el._markdownRemovalTimeout = window.setTimeout(() => {
      el.classList.remove('markdown-highlight', 'removing')
    }, 600) // Match the fade-out animation duration
  }, 10000) // 10 seconds before starting fade-out
}

/** Normalize string the same way rendered DOM text is normalized for matching. */
function norm(s) {
  return (s || '').replace(/\s+/g, ' ').replace(/\u00A0/g, ' ').trim()
}

/**
 * Build searchable candidates from chunk content lines.
 * Each line: skip empty/comments/images; strip markdown syntax (# headings, - list); decode entities; normalize.
 * Table rows become one candidate per cell (excluding separator cells).
 */
function buildCandidates(lines) {
  const out = []
  for (const raw of lines) {
    if (typeof raw !== 'string') continue
    let trimmed = raw.trim()
    if (!trimmed) continue
    if (trimmed.startsWith('<!--')) continue
    if (trimmed.startsWith('![')) continue

    trimmed = trimmed.replace(/^#{1,6}\s*/, '').replace(/^[-*+]\s+/, '')
    trimmed = trimmed.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&')
    const normalized = norm(trimmed)
    if (!normalized) continue

    if (trimmed.includes('|')) {
      const cells = trimmed.split('|').map((c) => c.trim())
      for (const cell of cells) {
        if (!cell || /^:?-{2,}:?$/.test(cell)) continue
        const n = norm(cell)
        if (n) out.push(n)
      }
    } else {
      out.push(normalized)
      // Numbered list items (e.g. "8) On receipt...") may render without the number in the DOM.
      const withoutNumber = normalized.replace(/^\d+\)\s+/, '').trim()
      if (withoutNumber.length > 20 && withoutNumber !== normalized) out.push(withoutNumber)
      // Roman/letter sub-points (e.g. "i. either...", "ii. the complainant...") may render without the marker.
      const withoutSubMarker = normalized.replace(/^(?:[ivxlcdm]+\.|[a-z]\.)\s+/i, '').trim()
      if (withoutSubMarker.length > 15 && withoutSubMarker !== normalized && withoutSubMarker !== withoutNumber) out.push(withoutSubMarker)
    }
  }
  return out
}

/**
 * Build full text of container and segments (text node -> range in that text).
 * Segments are in document order: { node, start, end }.
 */
function buildFullTextAndSegments(container) {
  const segments = []
  let fullText = ''
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT)
  let node
  while ((node = walker.nextNode())) {
    const t = (node.textContent || '').replace(/\s+/g, ' ').replace(/\u00A0/g, ' ').trim()
    if (!t) continue
    if (fullText.length) fullText += ' '
    const start = fullText.length
    fullText += t
    segments.push({ node, start, end: fullText.length })
  }
  return { fullText, segments }
}

/**
 * Collapse every run of whitespace in fullText to a single space.
 * Returns { normalized, oldToNew } where oldToNew[i] = index in normalized string for character i in fullText.
 * This makes search robust when DOM splits list items (e.g. "8)" and " On receipt..." => " 8)  On receipt...").
 */
function normalizeFullTextWithMap(fullText) {
  let normalized = ''
  const oldToNew = []
  for (let i = 0; i < fullText.length; i++) {
    const c = fullText[i]
    if (/\s/.test(c)) {
      if (normalized.length === 0 || normalized[normalized.length - 1] !== ' ') {
        normalized += ' '
      }
      oldToNew[i] = normalized.length - 1
    } else {
      normalized += c
      oldToNew[i] = normalized.length - 1
    }
  }
  return { normalized, oldToNew }
}

/**
 * Find character ranges [start, end) in normalized fullText where each candidate appears.
 * Match is case-insensitive. If anchorPos is set, for each candidate we pick the occurrence
 * whose start is closest to anchorPos (so we highlight the block containing the anchor, not an earlier duplicate).
 */
function findRangesInFullText(normalizedFullText, candidates, anchorPos) {
  const ranges = []
  const lower = normalizedFullText.toLowerCase()
  for (const cand of candidates) {
    if (!cand) continue
    const cLower = cand.toLowerCase()
    let idx = lower.indexOf(cLower)
    if (idx < 0) continue
    if (anchorPos != null) {
      let best = idx
      let bestDist = Math.abs(idx - anchorPos)
      while (idx >= 0) {
        const d = Math.abs(idx - anchorPos)
        if (d < bestDist) {
          bestDist = d
          best = idx
        }
        idx = lower.indexOf(cLower, idx + 1)
      }
      idx = best
    }
    ranges.push({ start: idx, end: idx + cand.length })
  }
  return ranges
}

/**
 * Map character ranges (in normalized space) to DOM elements.
 * Segments must have normStart, normEnd. We collect unique parent elements in document order.
 */
function rangesToElements(segments, ranges) {
  const elementSet = new Set()
  const order = []
  for (const { start, end } of ranges) {
    for (const seg of segments) {
      const segStart = seg.normStart
      const segEnd = seg.normEnd
      if (segEnd <= start || segStart >= end) continue
      const parent = seg.node.parentElement
      if (parent && !elementSet.has(parent)) {
        elementSet.add(parent)
        order.push(parent)
      }
    }
  }
  return order
}

function findTextAndScroll(container, _searchText, _sectionTitle, _heading, lines) {
  if (!container || !Array.isArray(lines) || !lines.length) return

  const candidates = buildCandidates(lines)
  if (!candidates.length) return

  const { fullText, segments } = buildFullTextAndSegments(container)
  if (!fullText) return

  const { normalized, oldToNew } = normalizeFullTextWithMap(fullText)
  for (const seg of segments) {
    seg.normStart = seg.start < oldToNew.length ? oldToNew[seg.start] : 0
    seg.normEnd = seg.end > 0 && seg.end - 1 < oldToNew.length ? oldToNew[seg.end - 1] + 1 : seg.normStart
  }

  const anchorPhrases = [
    'to the policyholder of having registered a nomination',
    'die after the policyholder but before his share',
    'If nominee(s) die after the policyholder but before his share',
  ]
  let anchorPos = null
  const nLower = normalized.toLowerCase()
  for (const phrase of anchorPhrases) {
    if (candidates.some((c) => c && c.toLowerCase().includes(phrase))) {
      const pos = nLower.indexOf(phrase.toLowerCase())
      if (pos >= 0) {
        anchorPos = pos
        break
      }
    }
  }
  let ranges = findRangesInFullText(normalized, candidates, anchorPos)
  if (anchorPos != null && ranges.length > 1) {
    ranges = [...ranges].sort((a, b) => Math.abs(a.start - anchorPos) - Math.abs(b.start - anchorPos))
  }
  const matchElements = rangesToElements(segments, ranges)

  if (matchElements.length > 0) {
    highlightElementAndScroll(matchElements[0])
    for (let i = 1; i < matchElements.length; i++) {
      const el = matchElements[i]
      el.classList.add('markdown-highlight')
      window.clearTimeout(el._markdownHighlightTimeout)
      el._markdownHighlightTimeout = window.setTimeout(() => {
        el.classList.remove('markdown-highlight')
      }, 2000)
    }
  }
}

export default function MarkdownViewer({ documentId, activeHighlight, onClose }) {
  const contentRef = useRef(null)
  const { data: markdown, isLoading, error } = useMarkdown(documentId)

  useEffect(() => {
    if (!activeHighlight || !contentRef.current) return
    findTextAndScroll(
      contentRef.current,
      activeHighlight.text,
      activeHighlight.sectionTitle,
      activeHighlight.heading,
      activeHighlight.lines
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
    <div className="flex flex-col h-full min-h-0 theme-main border-r border-slate-800">
      <div className="flex items-center gap-2 px-3 py-2 bg-slate-900/80 border-b border-slate-800 shrink-0">
        {typeof onClose === 'function' && (
          <button
            type="button"
            onClick={onClose}
            className="inline-flex items-center rounded-md px-2 py-1 text-xs font-medium text-slate-400 hover:bg-slate-800 hover:text-slate-100 transition-colors"
          >
            <span className="text-base leading-none mr-1">&times;</span>
            <span className="hidden sm:inline">Close</span>
          </button>
        )}
        <span className="text-xs text-slate-400">Document preview</span>
        <a
          href={markdownUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="ml-auto text-xs sm:text-sm text-indigo-300 hover:text-indigo-200 hover:underline"
        >
          Open in new tab
        </a>
      </div>
      <div
        ref={contentRef}
        data-markdown-root
        className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden p-4 column-scroll markdown-body [&_h1]:text-2xl [&_h1]:font-semibold [&_h1]:mt-4 [&_h2]:text-xl [&_h2]:font-semibold [&_h2]:mt-3 [&_h3]:text-lg [&_h3]:font-medium [&_h3]:mt-2 [&_p]:my-2 [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-6 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-6 [&_table]:border [&_table]:border-slate-300 [&_th]:border [&_th]:border-slate-300 [&_th]:px-2 [&_th]:py-1 [&_td]:border [&_td]:border-slate-200 [&_td]:px-2 [&_td]:py-1"
      >
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {stripPageBreakAndImageComments(markdown)}
        </ReactMarkdown>
      </div>
    </div>
  )
}
