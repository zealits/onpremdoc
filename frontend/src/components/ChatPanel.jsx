import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { queryDocumentStream } from '../api/client'

const DEFAULT_SUGGESTED = [
  'What is this document about?',
  'Summarize the main points.',
]

const CITATION_REGEX = /\[C(\d+)\]/g
const CHAT_STORAGE_KEY = 'onpremdoc-chat'
const SESSION_STORAGE_KEY_PREFIX = 'onpremdoc-session-'

function getStoredSessionId(documentId) {
  if (!documentId) return null
  try {
    const raw = localStorage.getItem(SESSION_STORAGE_KEY_PREFIX + documentId)
    if (raw == null) return null
    const n = parseInt(raw, 10)
    return Number.isNaN(n) ? null : n
  } catch {
    return null
  }
}

function setStoredSessionId(documentId, sessionId) {
  if (!documentId) return
  try {
    if (sessionId == null) {
      localStorage.removeItem(SESSION_STORAGE_KEY_PREFIX + documentId)
    } else {
      localStorage.setItem(SESSION_STORAGE_KEY_PREFIX + documentId, String(sessionId))
    }
  } catch {
    // ignore
  }
}

function getStoredMessages(documentId) {
  if (!documentId) return []
  try {
    const raw = localStorage.getItem(`${CHAT_STORAGE_KEY}-${documentId}`)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function saveMessages(documentId, messages) {
  if (!documentId || !messages?.length) return
  try {
    localStorage.setItem(`${CHAT_STORAGE_KEY}-${documentId}`, JSON.stringify(messages))
  } catch {
    // quota exceeded or other
  }
}

/** Rehype plugin: replace [C9], [C2] etc. in text nodes with inline <span dataCitation="9"> so they render as clickable. */
function rehypeCitationSpans() {
  return (tree) => {
    function visit(node, parent, idx) {
      if (parent?.type === 'element' && (parent.properties?.['data-citation'] || parent.properties?.dataCitation)) return
      if (node.type === 'text' && node.value && /\[C\d+\]/.test(node.value)) {
        const parts = node.value.split(/(\[C(\d+)\])/g)
        const newNodes = []
        for (let i = 0; i < parts.length; i++) {
          if (i % 3 === 0 && parts[i]) newNodes.push({ type: 'text', value: parts[i] })
          else if (i % 3 === 2)
            newNodes.push({
              type: 'element',
              tagName: 'span',
              properties: {
                'data-citation': parts[i],
                className: 'chunk-citation',
              },
              children: [{ type: 'text', value: `[C${parts[i]}]` }],
            })
        }
        if (parent && typeof idx === 'number' && newNodes.length) {
          parent.children.splice(idx, 1, ...newNodes)
        }
        return
      }
      if (node.children) for (let i = 0; i < node.children.length; i++) visit(node.children[i], node, i)
    }
    visit(tree, null, 0)
  }
}

function CitationButton({ chunkId, chunks, onHighlight }) {
  const chunk = (chunks || []).find((c) => c.chunk_index === chunkId)
  const handleClick = () => {
    if (!onHighlight || !chunk) return
    const contentStr = Array.isArray(chunk.content)
      ? chunk.content.join('\n')
      : (chunk.content ?? '')
    const lines = Array.isArray(chunk.content)
      ? chunk.content.filter((s) => typeof s === 'string' && s.trim().length > 0)
      : []
    onHighlight({
      pageNumber: chunk.page_number,
      text: contentStr,
      sectionTitle: chunk.section_title,
      heading: chunk.heading,
      lines,
    })
  }
  return (
    <button
      type="button"
      onClick={handleClick}
      className="inline align-baseline mx-0.5 px-0 py-0 rounded-none border-0 bg-transparent text-blue-600 hover:text-blue-800 hover:underline cursor-pointer font-inherit"
      title={chunk ? `Go to source (${chunk.section_title || 'Page ' + chunk.page_number})` : 'Go to source'}
    >
      [C{chunkId}]
    </button>
  )
}

function ChatPanel({ documentId, documentReady, documentSummary, suggestedQueries, documentName, isSummaryLoading, showSummaryBlock, onHighlightChunk }) {
  const suggested = (Array.isArray(suggestedQueries) && suggestedQueries.length > 0) ? suggestedQueries : DEFAULT_SUGGESTED
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const activeAssistantIndexRef = useRef(null)
  const abortControllerRef = useRef(null)

  // Load chat history from localStorage when document changes (or on mount)
  useEffect(() => {
    setMessages(getStoredMessages(documentId))
  }, [documentId])

  // Persist chat history whenever messages change
  useEffect(() => {
    saveMessages(documentId, messages)
  }, [documentId, messages])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (text) => {
    const q = (text || input).trim()
    if (!q || !documentId) return
    setInput('')

    // Cancel any in-flight stream
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }

    setMessages((m) => [...m, { role: 'user', content: q }])

    // Create a placeholder assistant message we will stream into
    let assistantIndex = null
    setMessages((m) => {
      const next = [
        ...m,
        {
          role: 'assistant',
          content: '',
          chunks: [],
          next_questions: [],
        },
      ]
      assistantIndex = next.length - 1
      activeAssistantIndexRef.current = assistantIndex
      return next
    })

    const controller = new AbortController()
    abortControllerRef.current = controller
    setIsStreaming(true)

    // Use stored session for this document so all messages in this chat belong to one session
    const sessionIdToSend = getStoredSessionId(documentId)

    try {
      await queryDocumentStream(documentId, q, sessionIdToSend, true, (chunk) => {
        setMessages((prev) => {
          const idx = activeAssistantIndexRef.current
          if (idx == null || idx < 0 || idx >= prev.length) return prev
          const copy = [...prev]
          const msg = { ...copy[idx] }

          if (chunk.type === 'meta') {
            if (chunk.session_id != null) setStoredSessionId(documentId, chunk.session_id)
            msg.chunks = chunk.chunks || []
            msg.next_questions = chunk.next_questions || []
            msg.is_page_summary = chunk.is_page_summary === true
            msg.page_number = chunk.page_number
          } else if (chunk.type === 'answer_chunk') {
            msg.content = (msg.content || '') + (chunk.delta || '')
          } else if (chunk.type === 'error') {
            msg.content = `Error: ${chunk.message || 'Request failed'}`
          }

          copy[idx] = msg
          return copy
        })
      })
    } catch (err) {
      if (err?.message?.toLowerCase().includes('session not found')) {
        setStoredSessionId(documentId, null)
      }
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: `Error: ${err?.message || 'Request failed'}` },
      ])
    } finally {
      setIsStreaming(false)
      abortControllerRef.current = null
    }
  }

  const onSubmit = (e) => {
    e.preventDefault()
    send()
  }

  if (!documentId) {
    return (
      <div className="flex flex-col h-full items-center justify-center p-6 text-center text-gray-500">
        <p>Select a document to chat.</p>
      </div>
    )
  }

  if (!documentReady) {
    return (
      <div className="flex flex-col h-full items-center justify-center p-8 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 border-l border-slate-800">
        <div className="animate-processing-fade-in text-center max-w-xs">
          <div className="relative inline-flex justify-center mb-5">
            <div className="absolute inset-0 rounded-full bg-indigo-500/35 animate-ping" style={{ animationDuration: '2.5s' }} />
            <div className="relative w-14 h-14 rounded-xl bg-slate-900 border border-indigo-400/50 shadow-xl flex items-center justify-center animate-processing-float">
              <svg className="w-7 h-7 text-indigo-200 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
              </svg>
            </div>
          </div>
          <h3 className="text-slate-100 font-semibold mb-1">Preparing chat</h3>
          <p className="text-slate-400 text-sm">Chat will be available when your document is ready.</p>
          <div className="mt-5 flex justify-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-indigo-300 animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-2 h-2 rounded-full bg-indigo-500 animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="chat-shell flex flex-col h-full min-h-0 px-3 py-3 sm:px-4 sm:py-4 border-l border-slate-800">
      <div className="flex flex-col h-full min-h-0 rounded-2xl theme-card shadow-[0_18px_60px_rgba(15,23,42,0.25)]">
        <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden px-4 pt-4 pb-2 space-y-4 column-scroll">
        {((showSummaryBlock == null ? (documentSummary || (documentReady && isSummaryLoading)) : showSummaryBlock)) && (
          <div className="document-summary mb-4 pb-4 border-b border-slate-600/60">
            <h2 className="text-base font-semibold mb-1 text-inherit">
              {documentName || 'Document'} summary
            </h2>
            <p className="text-xs theme-sidebar-muted mb-2">1 source</p>
            {isSummaryLoading ? (
              <div className="text-sm theme-sidebar-muted flex items-center gap-2">
                <span className="inline-block w-4 h-4 rounded-full border-2 border-indigo-400/60 border-t-indigo-300 animate-spin" />
                Loading summary…
              </div>
            ) : (
              <div className="text-sm leading-relaxed whitespace-pre-line max-w-none text-inherit">
                {documentSummary}
              </div>
            )}
          </div>
        )}
        {messages.length === 0 && (
          <div className="text-sm space-y-3">
            <p className="font-medium">Ask anything about this document.</p>
            <div className="flex flex-wrap gap-2">
              {suggested.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => send(q)}
                  className="px-3 py-2 rounded-full border border-indigo-200 bg-indigo-50 text-indigo-700 text-xs sm:text-sm hover:bg-indigo-100 hover:border-indigo-300 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm shadow-sm ${
                msg.role === 'user'
                  ? 'bg-gradient-to-r from-indigo-500 to-violet-500 text-white'
                  : 'bg-white text-slate-900 border border-slate-200'
              }`}
            >
              {msg.role === 'assistant' ? (
                <div className="space-y-2 prose-citations">
                  {msg.is_page_summary && (msg.chunks?.length ?? 0) > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5 mb-2 pb-2 border-b border-slate-200">
                      <span className="text-xs font-medium text-slate-500 mr-1">Chunks used:</span>
                      {msg.chunks.map((c) => (
                        <CitationButton
                          key={c.chunk_index}
                          chunkId={c.chunk_index}
                          chunks={msg.chunks}
                          onHighlight={onHighlightChunk}
                        />
                      ))}
                    </div>
                  )}
                  <ReactMarkdown
                    rehypePlugins={[rehypeCitationSpans]}
                    components={{
                      span: (props) => {
                        let citationId = props['data-citation'] ?? props.dataCitation ?? props.node?.properties?.['data-citation'] ?? props.node?.properties?.dataCitation
                        if (citationId == null || citationId === '') {
                          const child = props.children
                          const str = typeof child === 'string' ? child : (Array.isArray(child) && child.length === 1 && typeof child[0] === 'string' ? child[0] : null)
                          const m = str && str.match(/^\[C(\d+)\]$/)
                          if (m) citationId = m[1]
                        }
                        if (citationId != null && citationId !== '') {
                          return (
                            <CitationButton
                              chunkId={parseInt(String(citationId), 10)}
                              chunks={msg.chunks}
                              onHighlight={onHighlightChunk}
                            />
                          )
                        }
                        const { node, ...spanProps } = props
                        return <span {...spanProps} />
                      },
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                  {(msg.next_questions?.length ?? 0) > 0 && (
                    <div className="mt-3 pt-2 border-t border-slate-800/80">
                      <p className="text-xs font-medium text-slate-400 mb-2">Suggested follow-up questions</p>
                      <div className="flex flex-wrap gap-2">
                        {msg.next_questions.map((question, j) => (
                          <button
                            key={j}
                            type="button"
                            onClick={() => send(question)}
                            className="px-3 py-1.5 rounded-full border border-indigo-200 bg-indigo-50 text-indigo-700 text-xs sm:text-sm hover:bg-indigo-100 hover:border-indigo-300 disabled:opacity-50 disabled:cursor-not-allowed"
                            disabled={isStreaming}
                          >
                            {question}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                msg.content
              )}
            </div>
          </div>
        ))}
        {isStreaming && (
          <div className="flex justify-start">
            <div className="bg-slate-900/80 border border-slate-700/80 rounded-2xl px-3 py-2 text-sm text-slate-400">
              …
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <form onSubmit={onSubmit} className="px-4 pt-2 pb-4 border-t border-slate-800/80 shrink-0">
        <div className="flex gap-2 items-end">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask any question…"
            className="chat-input flex-1 px-4 py-2.5 rounded-xl bg-slate-900/70 border border-slate-700/80 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/80 focus:border-transparent"
            disabled={isStreaming}
            aria-label="Ask a question"
          />
          <button
            type="submit"
            disabled={!input.trim() || isStreaming}
            className="inline-flex items-center justify-center px-4 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 text-white font-medium shadow-sm hover:shadow-md hover:brightness-105 disabled:opacity-60 disabled:cursor-not-allowed"
            aria-label="Send"
          >
            <span className="hidden sm:inline mr-1.5">Send</span>
            <svg
              className="h-4 w-4"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M5 12H19M19 12L13 6M19 12L13 18"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
      </form>
      </div>
    </div>
  )
}

export default ChatPanel
