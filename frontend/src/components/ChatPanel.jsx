import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { useQueryDocument } from '../api/hooks'

const SUGGESTED = [
  'What is this document about?',
  'Summarize the main points.',
]

const CITATION_REGEX = /\[C(\d+)\]/g

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
    onHighlight({
      pageNumber: chunk.page_number,
      text: contentStr,
      sectionTitle: chunk.section_title,
      heading: chunk.heading,
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

function ChatPanel({ documentId, documentReady, onHighlightChunk }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)
  const queryMutation = useQueryDocument(documentId)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (text) => {
    const q = (text || input).trim()
    if (!q || !documentId) return
    setInput('')
    setMessages((m) => [...m, { role: 'user', content: q }])
    try {
      const res = await queryMutation.mutateAsync({ query: q })
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          content: res.answer,
          chunks: res.chunks || [],
        },
      ])
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: `Error: ${err?.message || 'Request failed'}` },
      ])
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
      <div className="flex flex-col h-full items-center justify-center p-6 text-center text-gray-500">
        <p>Processing document…</p>
        <p className="text-sm mt-2">Chat will be available when ready.</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full min-h-0 bg-white border-l border-gray-200">
      <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-gray-500 text-sm space-y-3">
            <p>Ask any question about this document.</p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTED.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => send(q)}
                  className="px-3 py-2 rounded-lg border border-indigo-200 text-indigo-700 text-sm hover:bg-indigo-50"
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
              className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${
                msg.role === 'user'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              {msg.role === 'assistant' ? (
                <div className="space-y-2 prose-citations">
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
                </div>
              ) : (
                msg.content
              )}
            </div>
          </div>
        ))}
        {queryMutation.isPending && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg px-3 py-2 text-sm text-gray-500">
              …
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <form onSubmit={onSubmit} className="p-4 border-t border-gray-100 shrink-0">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask any question…"
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            disabled={queryMutation.isPending}
            aria-label="Ask a question"
          />
          <button
            type="submit"
            disabled={!input.trim() || queryMutation.isPending}
            className="px-4 py-2 rounded-lg bg-indigo-600 text-white font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label="Send"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  )
}

export default ChatPanel
