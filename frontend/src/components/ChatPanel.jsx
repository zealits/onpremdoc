import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { queryDocumentStream } from "../api/client";

const DEFAULT_SUGGESTED = ["What is this document about?", "Summarize the main points."];

function ProcessingInterface({ documentName }) {
  const [currentStep, setCurrentStep] = useState(0);
  
  const steps = [
    {
      id: "upload",
      title: "Processing Document",
      description: "Analyzing file structure and content",
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-4.5B4.875 8.25A2.625 2.625 0 0 0 2.25 10.875v8.25a2.625 2.625 0 0 0 2.625 2.625h6.75a2.625 2.625 0 0 0 2.625-2.625V18a.75.75 0 0 1 .75-.75ZM19.5 14.25h-2.625a1.125 1.125 0 0 1-1.125-1.125v-2.625a1.125 1.125 0 0 1 1.125-1.125H19.5m-12.75 3v-3m6 3v-3m-6 3h6m-6 0v3m6-3v3m-6-3h6" />
        </svg>
      )
    },
    {
      id: "extract",
      title: "Extracting Content",
      description: "Reading text and extracting key information",
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-4.5A2.625 2.625 0 0 0 9 8.25v8.25a2.625 2.625 0 0 0 2.625 2.625h6.75A2.625 2.625 0 0 0 21 18.375v-4.125ZM4.5 19.5l15-15m0 0H8.25m11.25 0v11.25" />
        </svg>
      )
    },
    {
      id: "vectorize",
      title: "Creating Embeddings",
      description: "Converting content into searchable vectors",
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423L16.5 15.75l.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z" />
        </svg>
      )
    },
    {
      id: "prepare",
      title: "Preparing Chat",
      description: "Setting up AI assistant for conversations",
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H8.25m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H12m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 0 1-2.555-.337A5.972 5.972 0 0 1 5.41 20.97a5.969 5.969 0 0 1-.474-.065 4.48 4.48 0 0 0 .978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25Z" />
        </svg>
      )
    }
  ];

  // Cycle through steps automatically
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev + 1) % steps.length);
    }, 2500);
    
    return () => clearInterval(interval);
  }, [steps.length]);

  return (
    <div className="flex flex-col h-full items-center justify-center p-8 theme-main border-l theme-card">
      <div className="animate-processing-fade-in text-center max-w-md w-full">
        {/* Main Processing Icon */}
        <div className="relative inline-flex justify-center mb-8">
          <div 
            className="absolute inset-0 rounded-full bg-indigo-500/20 animate-ping" 
            style={{ animationDuration: "3s" }}
          />
          <div className="relative w-20 h-20 rounded-2xl theme-card border-2 border-indigo-400/30 shadow-xl flex items-center justify-center animate-processing-float processing-main-icon">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center animate-processing-glow text-white">
              {steps[currentStep].icon}
            </div>
          </div>
        </div>

        {/* Current Step Info */}
        <div className="mb-8">
          <h3 className="text-xl font-semibold mb-2 theme-card text-inherit">{steps[currentStep].title}</h3>
          <p className="theme-sidebar-muted text-sm leading-relaxed">{steps[currentStep].description}</p>
          {documentName && (
            <p className="theme-sidebar-muted text-xs mt-2 opacity-75">Processing: {documentName}</p>
          )}
        </div>

        {/* Progress Steps */}
        <div className="space-y-4 mb-8">
          {steps.map((step, index) => {
            const isActive = index === currentStep;
            const isCompleted = index < currentStep;
            const isPending = index > currentStep;
            
            return (
              <div key={step.id} className="flex items-center justify-between p-3 rounded-xl theme-card border processing-step-card">
                <div className="flex items-center gap-3">
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-500 ${
                    isActive 
                      ? 'bg-gradient-to-br from-indigo-500 to-violet-600 text-white animate-processing-pulse-soft' 
                      : isCompleted 
                        ? 'bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400'
                        : 'bg-gray-100 text-gray-400 dark:bg-gray-800 dark:text-gray-500'
                  }`}>
                    {isCompleted ? (
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    ) : (
                      React.cloneElement(step.icon, { className: "w-4 h-4" })
                    )}
                  </div>
                  <div className="text-left">
                    <p className={`text-sm font-medium transition-colors duration-500 ${
                      isActive ? 'text-indigo-600 dark:text-indigo-400' : 'text-inherit'
                    }`}>
                      {step.title}
                    </p>
                    <p className="text-xs theme-sidebar-muted">{step.description}</p>
                  </div>
                </div>
                
                {isActive && (
                  <div className="flex gap-1">
                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-bounce" style={{ animationDelay: "150ms" }} />
                    <div className="w-1.5 h-1.5 rounded-full bg-indigo-600 animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                )}
                
                {isPending && (
                  <div className="w-8 h-1 bg-gray-200 dark:bg-gray-700 rounded-full">
                    <div className="w-0 h-full bg-indigo-500 rounded-full transition-all duration-500"></div>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Loading Progress Bar */}
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-indigo-500 to-violet-600 rounded-full transition-all duration-500 animate-processing-shimmer"
            style={{ 
              width: `${((currentStep + 1) / steps.length) * 100}%`,
            }}
          />
        </div>
        
        <p className="theme-sidebar-muted text-xs mt-4 opacity-75">
          This usually takes 30-60 seconds depending on document size
        </p>
      </div>
    </div>
  );
}

const CITATION_REGEX = /\[C(\d+)\]/g;
const CHAT_STORAGE_KEY = "doconprem-chat";
const SESSION_STORAGE_KEY_PREFIX = "doconprem-session-";

function getStoredSessionId(documentId) {
  if (!documentId) return null;
  try {
    const raw = localStorage.getItem(SESSION_STORAGE_KEY_PREFIX + documentId);
    if (raw == null) return null;
    const n = parseInt(raw, 10);
    return Number.isNaN(n) ? null : n;
  } catch {
    return null;
  }
}

function setStoredSessionId(documentId, sessionId) {
  if (!documentId) return;
  try {
    if (sessionId == null) {
      localStorage.removeItem(SESSION_STORAGE_KEY_PREFIX + documentId);
    } else {
      localStorage.setItem(SESSION_STORAGE_KEY_PREFIX + documentId, String(sessionId));
    }
  } catch {
    // ignore
  }
}

function getStoredMessages(documentId) {
  if (!documentId) return [];
  try {
    const raw = localStorage.getItem(`${CHAT_STORAGE_KEY}-${documentId}`);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveMessages(documentId, messages) {
  if (!documentId || !messages?.length) return;
  try {
    localStorage.setItem(`${CHAT_STORAGE_KEY}-${documentId}`, JSON.stringify(messages));
  } catch {
    // quota exceeded or other
  }
}

/** Rehype plugin: replace [C9], [C2] etc. in text nodes with inline <span dataCitation="9"> so they render as clickable. */
function rehypeCitationSpans() {
  return (tree) => {
    function visit(node, parent, idx) {
      if (parent?.type === "element" && (parent.properties?.["data-citation"] || parent.properties?.dataCitation))
        return;
      if (node.type === "text" && node.value && /\[C\d+\]/.test(node.value)) {
        const parts = node.value.split(/(\[C(\d+)\])/g);
        const newNodes = [];
        for (let i = 0; i < parts.length; i++) {
          if (i % 3 === 0 && parts[i]) newNodes.push({ type: "text", value: parts[i] });
          else if (i % 3 === 2)
            newNodes.push({
              type: "element",
              tagName: "span",
              properties: {
                "data-citation": parts[i],
                className: "chunk-citation",
              },
              children: [{ type: "text", value: `[C${parts[i]}]` }],
            });
        }
        if (parent && typeof idx === "number" && newNodes.length) {
          parent.children.splice(idx, 1, ...newNodes);
        }
        return;
      }
      if (node.children) for (let i = 0; i < node.children.length; i++) visit(node.children[i], node, i);
    }
    visit(tree, null, 0);
  };
}

function CitationButton({ chunkId, chunks, onHighlight }) {
  const chunk = (chunks || []).find((c) => c.chunk_index === chunkId);
  const handleClick = () => {
    if (!onHighlight || !chunk) return;
    const contentStr = Array.isArray(chunk.content) ? chunk.content.join("\n") : (chunk.content ?? "");
    const lines = Array.isArray(chunk.content)
      ? chunk.content.filter((s) => typeof s === "string" && s.trim().length > 0)
      : [];
    onHighlight({
      pageNumber: chunk.page_number,
      text: contentStr,
      sectionTitle: chunk.section_title,
      heading: chunk.heading,
      lines,
    });
  };
  return (
    <button
      type="button"
      onClick={handleClick}
      className="inline align-baseline mx-0.5 px-0 py-0 rounded-none border-0 bg-transparent text-blue-600 hover:text-blue-800 hover:underline cursor-pointer font-inherit"
      title={chunk ? `Go to source (${chunk.section_title || "Page " + chunk.page_number})` : "Go to source"}
    >
      [C{chunkId}]
    </button>
  );
}

function ChatPanel({
  documentId,
  documentReady,
  documentSummary,
  suggestedQueries,
  documentName,
  isSummaryLoading,
  showSummaryBlock,
  onHighlightChunk,
}) {
  const suggested =
    Array.isArray(suggestedQueries) && suggestedQueries.length > 0 ? suggestedQueries : DEFAULT_SUGGESTED;
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const activeAssistantIndexRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Load chat history from localStorage when document changes (or on mount)
  useEffect(() => {
    setMessages(getStoredMessages(documentId));
  }, [documentId]);

  // Persist chat history whenever messages change
  useEffect(() => {
    saveMessages(documentId, messages);
  }, [documentId, messages]);

  useEffect(() => {
    const el = bottomRef.current;
    if (!el) return;
    // Prefer scrolling only the chat column, not the whole page
    const scrollContainer =
      el.closest(".column-scroll") ||
      el.parentElement;
    if (scrollContainer && "scrollTo" in scrollContainer) {
      scrollContainer.scrollTo({
        top: scrollContainer.scrollHeight,
        behavior: "smooth",
      });
    } else {
      // Fallback for older browsers
      el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [messages]);

  const send = async (text) => {
    const q = (text || input).trim();
    if (!q || !documentId) return;
    setInput("");

    // Cancel any in-flight stream
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    setMessages((m) => [...m, { role: "user", content: q }]);

    // Create a placeholder assistant message we will stream into
    let assistantIndex = null;
    setMessages((m) => {
      const next = [
        ...m,
        {
          role: "assistant",
          content: "",
          chunks: [],
          next_questions: [],
        },
      ];
      assistantIndex = next.length - 1;
      activeAssistantIndexRef.current = assistantIndex;
      return next;
    });

    const controller = new AbortController();
    abortControllerRef.current = controller;
    setIsStreaming(true);

    // Use stored session for this document so all messages in this chat belong to one session
    const sessionIdToSend = getStoredSessionId(documentId);

    try {
      await queryDocumentStream(documentId, q, sessionIdToSend, true, (chunk) => {
        setMessages((prev) => {
          const idx = activeAssistantIndexRef.current;
          if (idx == null || idx < 0 || idx >= prev.length) return prev;
          const copy = [...prev];
          const msg = { ...copy[idx] };

          if (chunk.type === "meta") {
            if (chunk.session_id != null) setStoredSessionId(documentId, chunk.session_id);
            msg.chunks = chunk.chunks || [];
            msg.next_questions = chunk.next_questions || [];
            msg.is_page_summary = chunk.is_page_summary === true;
            msg.page_number = chunk.page_number;
          } else if (chunk.type === "next_questions") {
            // Final chunk containing suggested follow-up questions once the answer is complete
            msg.next_questions = chunk.next_questions || [];
          } else if (chunk.type === "answer_chunk") {
            msg.content = (msg.content || "") + (chunk.delta || "");
          } else if (chunk.type === "error") {
            msg.content = `Error: ${chunk.message || "Request failed"}`;
          }

          copy[idx] = msg;
          return copy;
        });
      });
    } catch (err) {
      if (err?.message?.toLowerCase().includes("session not found")) {
        setStoredSessionId(documentId, null);
      }
      setMessages((m) => [...m, { role: "assistant", content: `Error: ${err?.message || "Request failed"}` }]);
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  };

  const onSubmit = (e) => {
    e.preventDefault();
    send();
  };

  if (!documentId) {
    return (
      <div className="flex flex-col h-full items-center justify-center p-6 text-center text-gray-500">
        <p>Select a document to chat.</p>
      </div>
    );
  }

  if (!documentReady) {
    return (
      <ProcessingInterface documentName={documentName} />
    );
  }

  return (
    <div className="chat-shell flex flex-col h-full min-h-0 px-3 py-3 sm:px-4 sm:py-4 border-l border-slate-800">
      <div className="flex flex-col h-full min-h-0 rounded-2xl theme-card shadow-[0_18px_60px_rgba(15,23,42,0.25)]">
        <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden px-4 pt-4 pb-2 space-y-4 column-scroll">
          {(showSummaryBlock == null ? documentSummary || (documentReady && isSummaryLoading) : showSummaryBlock) && (
            <div className="document-summary mb-6 pb-6 border-b border-slate-600/40">
              <div className="flex items-center gap-2 mb-4">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-sm">
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <h2 className="text-lg font-semibold text-inherit">{documentName || "Document"} Summary</h2>
              </div>
              {isSummaryLoading ? (
                <div className="flex items-center gap-3 py-8 theme-sidebar-muted">
                  <span className="inline-block w-5 h-5 rounded-full border-2 border-indigo-400/60 border-t-indigo-300 animate-spin" />
                  <span className="text-sm">Generating summary…</span>
                </div>
              ) : (
                <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 rounded-xl p-5 border border-slate-600/30">
                  <div className="text-sm leading-7 space-y-4 text-inherit">
                    {documentSummary?.split('\n\n').map((paragraph, index) => (
                      <p key={index} className="text-gray-200 dark:text-gray-200 last:mb-0">
                        {paragraph.trim()}
                      </p>
                    )) || documentSummary}
                  </div>
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
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm shadow-sm ${
                  msg.role === "user"
                    ? "bg-gradient-to-r from-indigo-500 to-violet-500 text-white"
                    : "chat-assistant-bubble border"
                }`}
              >
                {msg.role === "assistant" ? (
                  <div className="space-y-2 prose-citations">
                    {msg.is_page_summary && (msg.chunks?.length ?? 0) > 0 && (
                      <div className="flex flex-wrap items-center gap-1.5 mb-2 pb-2 border-b chat-assistant-divider">
                        <span className="text-xs font-medium chat-assistant-muted mr-1">Chunks used:</span>
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
                          let citationId =
                            props["data-citation"] ??
                            props.dataCitation ??
                            props.node?.properties?.["data-citation"] ??
                            props.node?.properties?.dataCitation;
                          if (citationId == null || citationId === "") {
                            const child = props.children;
                            const str =
                              typeof child === "string"
                                ? child
                                : Array.isArray(child) && child.length === 1 && typeof child[0] === "string"
                                  ? child[0]
                                  : null;
                            const m = str && str.match(/^\[C(\d+)\]$/);
                            if (m) citationId = m[1];
                          }
                          if (citationId != null && citationId !== "") {
                            return (
                              <CitationButton
                                chunkId={parseInt(String(citationId), 10)}
                                chunks={msg.chunks}
                                onHighlight={onHighlightChunk}
                              />
                            );
                          }
                          const { node, ...spanProps } = props;
                          return <span {...spanProps} />;
                        },
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                    {(msg.next_questions?.length ?? 0) > 0 && (
                      <div className="mt-4 pt-3 border-t chat-assistant-divider">
                        <div className="flex items-center gap-2 mb-3">
                          <div className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-indigo-500/10 text-indigo-500">
                            <svg
                              className="h-3.5 w-3.5"
                              viewBox="0 0 24 24"
                              fill="none"
                              xmlns="http://www.w3.org/2000/svg"
                            >
                              <path
                                d="M12 3L4 9V21H10V15H14V21H20V9L12 3Z"
                                stroke="currentColor"
                                strokeWidth="1.6"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                            </svg>
                          </div>
                          <div>
                            <p className="text-xs font-semibold uppercase tracking-wide text-indigo-700 dark:text-indigo-400">
                              Suggested follow-up questions
                            </p>
                            <p className="text-[11px] leading-snug theme-sidebar-muted">
                              Jump into a deeper, more focused analysis.
                            </p>
                          </div>
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                          {msg.next_questions.map((question, j) => (
                            <button
                              key={j}
                              type="button"
                              onClick={() => send(question)}
                              className="group flex w-full items-start gap-2 rounded-xl border border-indigo-200 bg-white/80 px-3 py-2 text-left text-xs sm:text-sm text-slate-800 hover:bg-indigo-50 hover:border-indigo-400/70 dark:border-indigo-500/20 dark:bg-indigo-500/5 dark:text-indigo-50 dark:hover:bg-indigo-500/15 transition-all disabled:opacity-60 disabled:cursor-not-allowed"
                              disabled={isStreaming}
                            >
                              <span className="mt-1 h-1.5 w-1.5 rounded-full bg-indigo-400 group-hover:bg-indigo-500 dark:group-hover:bg-indigo-300 flex-shrink-0" />
                              <span className="whitespace-normal break-words leading-snug">
                                {question}
                              </span>
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
              <div className="chat-assistant-bubble border rounded-2xl px-3 py-2 text-sm flex items-center gap-1.5">
                <span className="sr-only">Assistant is typing</span>
                <span className="w-1.5 h-1.5 rounded-full bg-slate-400 dark:bg-slate-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-slate-400 dark:bg-slate-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-slate-400 dark:bg-slate-500 animate-bounce" style={{ animationDelay: '300ms' }} />
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
              <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
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
  );
}

export default ChatPanel;
