import React, { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { queryDocumentStream, getDocumentChunks, listChatSessions } from "../api/client";
import { useDocument } from "../api/hooks";

const DEFAULT_SUGGESTED = ["What is this document about?", "Summarize the main points."];

function ProcessingInterface({ documentName, documentStatus }) {
  const steps = [
    {
      id: "uploaded",
      title: "Processing Document",
      description: "Analyzing file structure and content",
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-4.5B4.875 8.25A2.625 2.625 0 0 0 2.25 10.875v8.25a2.625 2.625 0 0 0 2.625 2.625h6.75a2.625 2.625 0 0 0 2.625-2.625V18a.75.75 0 0 1 .75-.75ZM19.5 14.25h-2.625a1.125 1.125 0 0 1-1.125-1.125v-2.625a1.125 1.125 0 0 1 1.125-1.125H19.5m-12.75 3v-3m6 3v-3m-6 3h6m-6 0v3m6-3v3m-6-3h6" />
        </svg>
      )
    },
    {
      id: "processing",
      title: "Extracting Content",
      description: "Reading text and extracting key information",
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-4.5A2.625 2.625 0 0 0 9 8.25v8.25a2.625 2.625 0 0 0 2.625 2.625h6.75A2.625 2.625 0 0 0 21 18.375v-4.125ZM4.5 19.5l15-15m0 0H8.25m11.25 0v11.25" />
        </svg>
      )
    },
    {
      id: "vectorized",
      title: "Creating Embeddings",
      description: "Converting content into searchable vectors",
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423L16.5 15.75l.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z" />
        </svg>
      )
    },
    {
      id: "ready",
      title: "Preparing Chat",
      description: "Setting up AI assistant for conversations",
      icon: (
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H8.25m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H12m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 0 1-2.555-.337A5.972 5.972 0 0 1 5.41 20.97a5.969 5.969 0 0 1-.474-.065 4.48 4.48 0 0 0 .978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25Z" />
        </svg>
      )
    }
  ];

  // Map status to step index
  const currentStepIndex = steps.findIndex(step => step.id === documentStatus);
  const currentStep = currentStepIndex >= 0 ? currentStepIndex : 0;

  return (
    <div className="flex flex-col h-full items-center justify-center p-4 sm:p-6 md:p-8 theme-main md:border-l theme-card">
      <div className="animate-processing-fade-in text-center max-w-sm sm:max-w-md w-full">
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
        <div className="mb-6 sm:mb-8">
          <h3 className="text-lg sm:text-xl font-semibold mb-2 theme-card text-inherit">{steps[currentStep].title}</h3>
          <p className="theme-sidebar-muted text-xs sm:text-sm leading-relaxed">{steps[currentStep].description}</p>
          {documentName && (
            <p className="theme-sidebar-muted text-[10px] sm:text-xs mt-2 opacity-75 truncate">
              <span className="hidden sm:inline">Processing: </span>{documentName}
            </p>
          )}
        </div>

        {/* Progress Steps */}
        <div className="space-y-3 sm:space-y-4 mb-6 sm:mb-8">
          {steps.map((step, index) => {
            const isActive = index === currentStep;
            const isCompleted = index < currentStep;
            const isPending = index > currentStep;
            
            return (
              <div key={step.id} className="flex items-center justify-between p-2.5 sm:p-3 rounded-lg sm:rounded-xl theme-card border processing-step-card">
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
                  <div className="text-left min-w-0 flex-1">
                    <p className={`text-xs sm:text-sm font-medium transition-colors duration-500 truncate ${
                      isActive ? 'text-indigo-600 dark:text-indigo-400' : 'text-inherit'
                    }`}>
                      {step.title}
                    </p>
                    <p className="text-[10px] sm:text-xs theme-sidebar-muted line-clamp-2 sm:line-clamp-1">{step.description}</p>
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
        
        {/* <p className="theme-sidebar-muted text-xs mt-4 opacity-75">
          This usually takes 30-60 seconds depending on document size
        </p> */}
      </div>
    </div>
  );
}

const CITATION_REGEX = /\[C(\d+)\]/g;
const CHAT_STORAGE_KEY = "doconprem-chat";
const SESSION_STORAGE_KEY_PREFIX = "doconprem-session-";
const MAX_VISIBLE_QUESTIONS = 4;
const MAX_VISIBLE_SUMMARY_CITATIONS = 7;

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
  if (!documentId) return;
  try {
    const safeMessages = Array.isArray(messages) ? messages : [];
    localStorage.setItem(`${CHAT_STORAGE_KEY}-${documentId}`, JSON.stringify(safeMessages));
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

function expandChunkReferenceGroups(text) {
  if (typeof text !== "string") return "";
  let out = text;

  // Convert footnote-style `^[C1,C5,C12]` into individual citation tokens: `[C1] [C5] [C12]`
  out = out.replace(/\^\[\s*([^\]]+?)\s*\]/g, (match, inner) => {
    const ids = Array.from(String(inner).matchAll(/C(\d+)/gi)).map((m) => m[1]);
    if (!ids.length) return match;
    return ids.map((id) => `[C${id}]`).join(" ");
  });

  // Convert bracket-group style `[C1,C5,C12]` into individual citation tokens.
  // (We keep it strict so we don't transform normal markdown link text.)
  out = out.replace(/\[\s*(C\d+(?:\s*,\s*C\d+)*)\s*\]/gi, (match, inner) => {
    const ids = Array.from(String(inner).matchAll(/C(\d+)/gi)).map((m) => m[1]);
    if (!ids.length) return match;
    return ids.map((id) => `[C${id}]`).join(" ");
  });

  return out;
}

function SummaryCitationButton({ chunkId, fetchChunkById, onHighlightChunk }) {
  const [isLoading, setIsLoading] = useState(false);

  const handleClick = async () => {
    if (isLoading) return;
    if (!fetchChunkById || typeof fetchChunkById !== "function") return;
    if (!onHighlightChunk || typeof onHighlightChunk !== "function") return;

    setIsLoading(true);
    try {
      const chunk = await fetchChunkById(chunkId);
      if (chunk) {
        const contentStr = Array.isArray(chunk.content) ? chunk.content.join("\n") : chunk.content ?? "";
        const lines = Array.isArray(chunk.content)
          ? chunk.content.filter((s) => typeof s === "string" && s.trim().length > 0)
          : [];

        onHighlightChunk({
          pageNumber: chunk.page_number,
          text: contentStr,
          sectionTitle: chunk.section_title,
          heading: chunk.heading,
          lines,
        });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const baseClasses =
    "inline-flex align-middle px-1.5 py-0.5 rounded text-xs font-medium cursor-pointer transition-all duration-200 touch-manipulation whitespace-nowrap";
  const sourceClasses =
    "text-indigo-600 hover:text-indigo-800 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 hover:border-indigo-300 dark:text-indigo-300 dark:hover:text-indigo-200 dark:bg-indigo-900/30 dark:hover:bg-indigo-900/50 dark:border-indigo-800 dark:hover:border-indigo-600 hover:scale-105 active:scale-95 hover:shadow-sm";

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={isLoading}
      className={`${baseClasses} ${sourceClasses}`}
      title="Go to this source (scroll + highlight)"
    >
      {isLoading ? "…" : `${chunkId}`}
    </button>
  );
}

function CitationButton({ chunkId, chunks, onHighlight, variant = "citation" }) {
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

  const baseClasses = "inline align-baseline mx-0.5 px-1.5 py-0.5 rounded text-xs font-medium cursor-pointer transition-all duration-200 touch-manipulation";
  const citationClasses = "text-blue-600 hover:text-blue-800 hover:underline bg-transparent border-0 hover:scale-105 active:scale-95";
  const sourceClasses = "text-indigo-600 hover:text-indigo-800 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 hover:border-indigo-300 dark:text-indigo-300 dark:hover:text-indigo-200 dark:bg-indigo-900/30 dark:hover:bg-indigo-900/50 dark:border-indigo-800 dark:hover:border-indigo-600 hover:scale-105 active:scale-95 hover:shadow-sm min-h-[28px] min-w-[28px] flex items-center justify-center";

  const classes = variant === "source" ? `${baseClasses} ${sourceClasses}` : `${baseClasses} ${citationClasses}`;

  return (
    <button
      type="button"
      onClick={handleClick}
      className={classes}
      title={chunk ? `Go to source (${chunk.section_title || "Page " + chunk.page_number})` : "Go to source"}
    >
      {variant === "source" ? `${chunkId}` : `[C${chunkId}]`}
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
  // Get document data to track processing status
  const { data: documentData } = useDocument(documentId);
  const suggested =
    Array.isArray(suggestedQueries) && suggestedQueries.length > 0 ? suggestedQueries : DEFAULT_SUGGESTED;
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const activeAssistantIndexRef = useRef(null);
  const abortControllerRef = useRef(null);
  
  // Typing effect state
  const [typingBuffer, setTypingBuffer] = useState("");
  const [displayedContent, setDisplayedContent] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const typingIntervalRef = useRef(null);
  const currentTypingIndexRef = useRef(0);
  const [typingPhase, setTypingPhase] = useState("chunks"); // "chunks", "content", "questions"
  const skipNextPersistRef = useRef(true);
  
  // Mobile-specific states
  const [isMobileDevice, setIsMobileDevice] = useState(false);
  const [isKeyboardVisible, setIsKeyboardVisible] = useState(false);
  
  // Detect mobile device and keyboard visibility
  useEffect(() => {
    const checkMobile = () => {
      setIsMobileDevice(window.innerWidth < 768);
    };
    
    const handleResize = () => {
      checkMobile();
      // Detect virtual keyboard on mobile
      if (window.innerWidth < 768) {
        const heightDiff = window.screen.height - window.innerHeight;
        setIsKeyboardVisible(heightDiff > 150);
      }
    };
    
    checkMobile();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  // Mobile touch scroll optimization
  const chatContainerRef = useRef(null);
  
  const optimizeScrollForMobile = useCallback(() => {
    if (isMobileDevice && chatContainerRef.current) {
      const container = chatContainerRef.current;
      container.style.webkitOverflowScrolling = 'touch';
      container.style.scrollBehavior = 'smooth';
    }
  }, [isMobileDevice]);
  
  useEffect(() => {
    optimizeScrollForMobile();
  }, [optimizeScrollForMobile]);

  // Cache chunk payloads so repeated clicks on `[C#]` are fast.
  const chunkLookupCacheRef = useRef(new Map());
  const fetchChunkById = useCallback(
    async (chunkId) => {
      const id = Number(chunkId);
      if (!Number.isFinite(id)) return null;
      if (chunkLookupCacheRef.current.has(id)) return chunkLookupCacheRef.current.get(id);

      try {
        const chunks = await getDocumentChunks(documentId, [id]);
        const chunk = Array.isArray(chunks) ? chunks.find((c) => c.chunk_index === id) : null;
        if (chunk) chunkLookupCacheRef.current.set(id, chunk);
        return chunk;
      } catch {
        return null;
      }
    },
    [documentId]
  );

  // Sequential typing effect function
  const startSequentialTypingEffect = (fullText, messageIndex, messageMetadata) => {
    setIsTyping(true);
    setTypingPhase("chunks");
    
    // Clear any existing typing animation
    if (typingIntervalRef.current) {
      clearInterval(typingIntervalRef.current);
    }

    // Phase 1: Show source chunks immediately
    setMessages((prev) => {
      if (messageIndex == null || messageIndex < 0 || messageIndex >= prev.length) return prev;
      const copy = [...prev];
      copy[messageIndex] = { 
        ...copy[messageIndex], 
        showSourceChunks: true,
        content: "",
        isTyping: true,
        typingPhase: "chunks"
      };
      return copy;
    });

    // Phase 2: Start typing main content after a brief delay
    setTimeout(() => {
      setTypingPhase("content");
      currentTypingIndexRef.current = 0;
      
      typingIntervalRef.current = setInterval(() => {
        const currentIndex = currentTypingIndexRef.current;
        
        if (currentIndex >= fullText.length) {
          clearInterval(typingIntervalRef.current);
          
          // Phase 3: Show follow-up questions after main content is done
          setTimeout(() => {
            setTypingPhase("questions");
            
            setMessages((prev) => {
              if (messageIndex == null || messageIndex < 0 || messageIndex >= prev.length) return prev;
              const copy = [...prev];
              copy[messageIndex] = { 
                ...copy[messageIndex], 
                content: fullText,
                isTyping: false,
                showFollowUpQuestions: true,
                typingPhase: "complete"
              };
              return copy;
            });
            
            setIsTyping(false);
          }, 500);
          return;
        }
        
        currentTypingIndexRef.current = currentIndex + 1;
        
        // Update message with current typed content
        setMessages((prev) => {
          if (messageIndex == null || messageIndex < 0 || messageIndex >= prev.length) return prev;
          const copy = [...prev];
          copy[messageIndex] = { 
            ...copy[messageIndex], 
            content: fullText.substring(0, currentIndex + 1),
            isTyping: true,
            typingPhase: "content"
          };
          return copy;
        });
        
      }, 25); // Adjust speed: lower = faster typing
    }, 800); // Delay before starting main content typing
  };

  // Clean up typing effect on unmount
  useEffect(() => {
    return () => {
      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
      }
    };
  }, []);

  // Load chat history from localStorage when document changes (or on mount)
  useEffect(() => {
    const loaded = getStoredMessages(documentId);
    // Skip one persist cycle immediately after loading a document's storage snapshot.
    // This prevents old in-memory messages from being written under the new document key.
    skipNextPersistRef.current = true;
    // eslint-disable-next-line no-console
    console.warn("Chat load from storage:", {
      documentId,
      storageKey: `${CHAT_STORAGE_KEY}-${documentId}`,
      count: Array.isArray(loaded) ? loaded.length : 0,
      previewRoles: Array.isArray(loaded) ? loaded.slice(0, 3).map((m) => m?.role) : [],
    });
    setMessages(loaded);
  }, [documentId]);

  // Log session ids as soon as a chat/document is opened.
  useEffect(() => {
    let active = true;
    const loadSessionsOnOpen = async () => {
      if (!documentId) return;
      try {
        const sessions = await listChatSessions(documentId);
        if (!active) return;
        const sectionIds = Array.isArray(sessions)
          ? sessions.map((s) => s?.id).filter((id) => id != null)
          : [];
        // eslint-disable-next-line no-console
        console.warn("Chat open session ids:", sectionIds);
      } catch (err) {
        if (!active) return;
        // eslint-disable-next-line no-console
        console.warn("Failed to load chat sessions on open:", err);
      }
    };
    loadSessionsOnOpen();
    return () => {
      active = false;
    };
  }, [documentId]);

  // Persist chat history whenever messages change
  useEffect(() => {
    if (skipNextPersistRef.current) {
      skipNextPersistRef.current = false;
      // eslint-disable-next-line no-console
      console.warn("Skipping persist during storage hydration:", {
        documentId,
        messageCount: Array.isArray(messages) ? messages.length : 0,
      });
      return;
    }
    // eslint-disable-next-line no-console
    console.warn("Chat save to storage:", {
      documentId,
      storageKey: `${CHAT_STORAGE_KEY}-${documentId}`,
      count: Array.isArray(messages) ? messages.length : 0,
      lastRole: Array.isArray(messages) && messages.length ? messages[messages.length - 1]?.role : null,
    });
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
    // eslint-disable-next-line no-console
    console.warn("Chat send called:", {
      documentId,
      questionPreview: q.slice(0, 80),
      existingMessageCount: messages.length,
      storedSessionId: getStoredSessionId(documentId),
    });
    setInput("");

    // Cancel any in-flight stream and typing effect
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    
    // Stop any active typing animation
    if (typingIntervalRef.current) {
      clearInterval(typingIntervalRef.current);
      setIsTyping(false);
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
          chunk_indices_used_for_answer: [],
          isTyping: false,
          showSourceChunks: false,
          showFollowUpQuestions: false,
          typingPhase: "chunks"
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

    let completeResponse = "";
    let responseMetadata = null;

    try {
      await queryDocumentStream(documentId, q, sessionIdToSend, true, (chunk) => {
        if (chunk.type === "meta") {
          responseMetadata = {
            session_id: chunk.session_id,
            chunks: chunk.chunks || [],
            next_questions: chunk.next_questions || [],
            is_page_summary: chunk.is_page_summary === true,
            page_number: chunk.page_number,
            chunk_indices_used_for_answer: chunk.chunk_indices_used_for_answer || []
          };
          // eslint-disable-next-line no-console
          console.log("Chat stream ids:", {
            session_id: responseMetadata.session_id,
            source_chunk_ids: responseMetadata.chunk_indices_used_for_answer,
            chunk_ids: responseMetadata.chunks.map((c) => c?.chunk_index).filter((id) => id != null),
          });
          
          if (chunk.session_id != null) setStoredSessionId(documentId, chunk.session_id);
          
          // Update message metadata immediately
          setMessages((prev) => {
            const idx = activeAssistantIndexRef.current;
            if (idx == null || idx < 0 || idx >= prev.length) return prev;
            const copy = [...prev];
            copy[idx] = { 
              ...copy[idx], 
              chunks: responseMetadata.chunks,
              next_questions: responseMetadata.next_questions,
              is_page_summary: responseMetadata.is_page_summary,
              page_number: responseMetadata.page_number,
              chunk_indices_used_for_answer: responseMetadata.chunk_indices_used_for_answer
            };
            return copy;
          });
          
        } else if (chunk.type === "next_questions") {
          // Final chunk containing suggested follow-up questions once the answer is complete
          if (responseMetadata) {
            responseMetadata.next_questions = chunk.next_questions || [];
          }
        } else if (chunk.type === "answer_chunk") {
          completeResponse += (chunk.delta || "");
        } else if (chunk.type === "error") {
          completeResponse = `Error: ${chunk.message || "Request failed"}`;
        }
      });

      // Start sequential typing effect with the complete response
      if (completeResponse && activeAssistantIndexRef.current != null) {
        startSequentialTypingEffect(completeResponse, activeAssistantIndexRef.current, responseMetadata);
      } else if (activeAssistantIndexRef.current != null) {
        // No response received, clear the placeholder message
        setMessages((prev) => {
          const idx = activeAssistantIndexRef.current;
          if (idx == null || idx < 0 || idx >= prev.length) return prev;
          const copy = [...prev];
          copy[idx] = { 
            ...copy[idx], 
            content: "No response received.",
            isTyping: false 
          };
          return copy;
        });
      }
    } catch (err) {
      if (err?.message?.toLowerCase().includes("session not found")) {
        setStoredSessionId(documentId, null);
      }
      
      const errorMessage = `Error: ${err?.message || "Request failed"}`;
      
      // Show error with typing effect if we have an active message
      if (activeAssistantIndexRef.current != null) {
        startSequentialTypingEffect(errorMessage, activeAssistantIndexRef.current, null);
      } else {
        setMessages((m) => [...m, { role: "assistant", content: errorMessage, isTyping: false }]);
      }
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
      <ProcessingInterface 
        documentName={documentName} 
        documentStatus={documentData?.status || "uploaded"} 
      />
    );
  }

  // Limit document summary citation chips to avoid long noisy rows.
  const visibleSummaryCitationIds = new Set();

  return (
    <div className="chat-shell flex flex-col h-full min-h-0 px-2 py-2 sm:px-3 sm:py-3 md:px-4 md:py-4 md:border-l border-slate-800 mobile-safe-area">
      <div className="flex flex-col h-full min-h-0 rounded-xl md:rounded-2xl theme-card shadow-lg md:shadow-[0_18px_60px_rgba(15,23,42,0.25)]">
        <div 
          ref={chatContainerRef}
          className={`flex-1 min-h-0 overflow-y-auto overflow-x-hidden px-3 sm:px-4 pt-3 sm:pt-4 pb-2 space-y-3 sm:space-y-4 column-scroll ${
            isKeyboardVisible ? 'pb-4' : ''
          }`}
        >
          {(showSummaryBlock == null ? documentSummary || (documentReady && isSummaryLoading) : showSummaryBlock) && (
            <div className="document-summary mb-4 sm:mb-6 pb-4 sm:pb-6 border-b border-slate-600/40">
              <div className="flex items-center gap-2 mb-3 sm:mb-4">
                <div className="flex h-7 w-7 sm:h-8 sm:w-8 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-sm">
                  <svg className="h-3.5 w-3.5 sm:h-4 sm:w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <h2 className="text-base sm:text-lg font-semibold text-inherit truncate">
                  <span className="hidden sm:inline">{documentName || "Document"} Summary</span>
                  <span className="sm:hidden">Summary</span>
                </h2>
              </div>
              {isSummaryLoading ? (
                <div className="flex items-center gap-3 py-8 theme-sidebar-muted">
                  <span className="inline-block w-5 h-5 rounded-full border-2 border-indigo-400/60 border-t-indigo-300 animate-spin" />
                  <span className="text-sm">Generating summary…</span>
                </div>
              ) : (
                <div className="summary-surface rounded-xl p-5 border">
                  <div className="text-sm leading-7 space-y-4 text-inherit">
                    <ReactMarkdown
                      rehypePlugins={[rehypeCitationSpans]}
                      components={{
                        p: ({ node, ...props }) => <p {...props} className="summary-text last:mb-0" />,
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
                            const id = parseInt(String(citationId), 10);
                            if (!Number.isNaN(id)) {
                              if (
                                !visibleSummaryCitationIds.has(id) &&
                                visibleSummaryCitationIds.size >= MAX_VISIBLE_SUMMARY_CITATIONS
                              ) {
                                return null;
                              }
                              visibleSummaryCitationIds.add(id);
                              return (
                                <SummaryCitationButton
                                  chunkId={id}
                                  fetchChunkById={fetchChunkById}
                                  onHighlightChunk={onHighlightChunk}
                                />
                              );
                            }
                          }

                          const { node, ...spanProps } = props;
                          return <span {...spanProps} />;
                        },
                      }}
                    >
                      {expandChunkReferenceGroups(documentSummary || "")}
                    </ReactMarkdown>
                  </div>
                </div>
              )}
            </div>
          )}
          {messages.length === 0 && (
            <div className="space-y-3 sm:space-y-4 mt-2">
              <p className="ask-prompt font-medium text-sm sm:text-base">Ask anything about this document.</p>
              <div className="flex flex-col items-start gap-2.5 suggested-questions">
                {suggested.slice(0, MAX_VISIBLE_QUESTIONS).map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => send(q)}
                    className="question-chip group inline-flex w-auto max-w-full self-start items-center px-3.5 sm:px-4 py-2.5 sm:py-2.5 rounded-xl sm:rounded-2xl border-0 text-xs sm:text-sm transition-all duration-200 font-medium touch-manipulation active:scale-[0.98] text-left"
                  >
                    <span className="question-chip-label block max-w-full whitespace-nowrap overflow-hidden text-ellipsis leading-snug transition-colors duration-200">
                      {q}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[90%] sm:max-w-[85%] rounded-xl sm:rounded-2xl px-3 sm:px-3.5 py-2 sm:py-2.5 text-sm shadow-sm ${
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
                            variant="citation"
                          />
                        ))}
                      </div>
                    )}
                    {/* Show chunk indices used for answer when available - Phase 1 */}
                    {!msg.is_page_summary && 
                     (msg.chunk_indices_used_for_answer?.length ?? 0) > 0 && 
                     (msg.showSourceChunks || msg.typingPhase === "complete" || !msg.isTyping) && (
                      <div className="flex flex-wrap items-center gap-1.5 mb-2 pb-2 border-b chat-assistant-divider typing-phase-enter">
                        <span className="text-xs font-medium chat-assistant-muted mr-1">Sources:</span>
                        {msg.chunk_indices_used_for_answer.map((chunkIndex) => (
                          <CitationButton
                            key={chunkIndex}
                            chunkId={chunkIndex}
                            chunks={msg.chunks}
                            onHighlight={onHighlightChunk}
                            variant="source"
                          />
                        ))}
                      </div>
                    )}
                    {/* Main content - Phase 2 */}
                    {(msg.typingPhase === "content" || msg.typingPhase === "complete" || !msg.isTyping) && (
                      <div className="relative">
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
                                    variant="citation"
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
                        {msg.isTyping && msg.typingPhase === "content" && (
                          <span className="typing-cursor inline-block w-0.5 h-5 bg-current ml-1"></span>
                        )}
                      </div>
                    )}
                    {/* Follow-up questions - Phase 3 */}
                    {(msg.next_questions?.length ?? 0) > 0 && 
                     (msg.showFollowUpQuestions || msg.typingPhase === "complete" || !msg.isTyping) && (
                      <div className="mt-4 pt-3 border-t chat-assistant-divider typing-phase-enter">
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
                        <div className="flex flex-col items-start gap-2.5 sm:gap-3">
                          {msg.next_questions.slice(0, MAX_VISIBLE_QUESTIONS).map((question, j) => (
                            <button
                              key={j}
                              type="button"
                              onClick={() => send(question)}
                              className="question-chip group inline-flex w-auto max-w-full self-start items-center gap-2 rounded-xl sm:rounded-2xl border-0 px-3.5 py-2.5 text-left text-xs sm:text-sm transition-all duration-200 disabled:opacity-60 disabled:cursor-not-allowed touch-manipulation active:scale-[0.98]"
                              disabled={isStreaming}
                            >
                              <span className="question-chip-dot mt-1 h-1.5 w-1.5 rounded-full flex-shrink-0 transition-colors duration-200" />
                              <span className="question-chip-label block max-w-full whitespace-nowrap overflow-hidden text-ellipsis leading-snug transition-colors duration-200">
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
        <form 
          onSubmit={onSubmit} 
          className={`px-3 sm:px-4 pt-2 pb-3 sm:pb-4 border-t border-slate-800/80 shrink-0 ${
            isKeyboardVisible ? 'pb-2' : ''
          }`}
        >
          <div className="flex gap-2 items-end">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask any question…"
              className="chat-input flex-1 px-3 sm:px-4 py-2.5 sm:py-3 rounded-lg sm:rounded-xl bg-slate-900/70 border border-slate-700/80 text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/80 focus:border-transparent text-sm sm:text-base min-h-[44px]"
              disabled={isStreaming}
              aria-label="Ask a question"
            />
            <button
              type="submit"
              disabled={!input.trim() || isStreaming}
              className="inline-flex items-center justify-center px-3 sm:px-4 py-2.5 sm:py-3 rounded-lg sm:rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 text-white font-medium shadow-sm hover:shadow-md hover:brightness-105 disabled:opacity-60 disabled:cursor-not-allowed touch-manipulation active:scale-95 min-h-[44px] min-w-[44px] sm:min-w-auto"
              aria-label="Send"
            >
              <span className="hidden sm:inline mr-1.5">Send</span>
              <svg className="h-4 w-4 sm:h-5 sm:w-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
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
