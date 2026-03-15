import { useEffect, useRef, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useDocument, useDocumentSummary, useVectorize, useDeleteDocument } from "../api/hooks";
import MarkdownViewer from "../components/MarkdownViewer";
import ChatPanel from "../components/ChatPanel";
import DocumentToolbar from "../components/DocumentToolbar";
import ConfirmModal from "../components/ConfirmModal";

function getDocumentDisplayName(doc, documentId) {
  if (!doc) return documentId || "";
  if (doc.name && doc.name !== documentId) return doc.name;
  const path = doc.markdown_path || doc.page_mapping_path || doc.confidence_path || "";
  if (path) {
    const parts = String(path).split(/[\\/]/);
    const file = parts[parts.length - 1] || "";
    const withoutExt = file.replace(/\.(md|pdf)$/i, "");
    if (withoutExt) return withoutExt;
  }
  return documentId || "";
}

export default function DocumentPage() {
  const { documentId } = useParams();
  const navigate = useNavigate();
  const { data: doc, isLoading, error } = useDocument(documentId);
  const vectorizeMutation = useVectorize(documentId);
  const deleteMutation = useDeleteDocument(documentId);
  const vectorizeTriggered = useRef(false);
  const [activeHighlight, setActiveHighlight] = useState(null);
  const [isMarkdownOpen, setIsMarkdownOpen] = useState(false);
  const [quickSummary, setQuickSummary] = useState(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  useEffect(() => {
    vectorizeTriggered.current = false;
    setActiveHighlight(null);
    setIsMarkdownOpen(false);
    setQuickSummary(null);
  }, [documentId]);

  useEffect(() => {
    if (!doc || vectorizeTriggered.current) return;
    if (doc.status === "processing" && doc.markdown_path) {
      vectorizeTriggered.current = true;
      vectorizeMutation.mutate(undefined, {
        onSuccess: (data) => {
          if (data?.summary) {
            setQuickSummary(data.summary);
          }
        },
      });
    }
  }, [doc?.status, doc?.markdown_path]);

  // If the document no longer exists (e.g. deleted in another tab),
  // redirect back to the landing page instead of showing a plain error.
  useEffect(() => {
    if (!error) return;
    const msg = String(error?.message || "").toLowerCase();
    if (msg.includes("not found") || msg.includes("no such document")) {
      navigate("/", { replace: true });
    }
  }, [error, navigate]);

  const ready = doc?.status === "ready";
  const {
    data: summaryData,
    isLoading: isSummaryLoading,
    isError: isSummaryError,
  } = useDocumentSummary(documentId, {
    enabled: !!documentId && ready,
  });
  const displaySummary =
    summaryData?.summary ??
    doc?.doc_summary ??
    (isSummaryError ? "Summary could not be loaded. Ask a question below." : null);
  const displaySuggestedQueries =
    (summaryData?.suggested_queries?.length ? summaryData.suggested_queries : doc?.suggested_queries) ?? null;
  const showSummaryLoading = ready && isSummaryLoading;
  const showSummaryBlock = ready && (displaySummary || showSummaryLoading);

  if (isLoading && !doc) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-4 bg-gradient-to-br from-slate-50 via-indigo-50/20 to-slate-50">
        <div className="w-12 h-12 rounded-xl border-2 border-indigo-200 border-t-indigo-500 animate-spin" />
        <p className="text-gray-600 font-medium animate-pulse">Loading document…</p>
      </div>
    );
  }

  if (!documentId) return null;

  const handleDelete = () => {
    if (!documentId) return;
    setShowDeleteConfirm(true);
  };

  const handleDeleteConfirm = () => {
    setShowDeleteConfirm(false);
    deleteMutation.mutate(undefined, {
      onSuccess: () => {
        navigate("/", { replace: true });
      },
    });
  };

  const handleDeleteCancel = () => {
    setShowDeleteConfirm(false);
  };

  return (
    <div className="flex-1 flex flex-col h-full min-h-0">
      <header className="doc-header shrink-0 border-b border-slate-800 bg-slate-900/70 backdrop-blur flex items-center justify-between gap-3 px-5 py-3">
        <div className="min-w-0">
          <div className="font-medium truncate text-sm sm:text-base">{getDocumentDisplayName(doc, documentId)}</div>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleDelete}
            disabled={deleteMutation.isPending}
            className="inline-flex items-center justify-center rounded-full border border-red-500/60 px-3 py-1 text-[11px] font-medium text-red-200 hover:bg-red-500/10 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            Delete
          </button>
        </div>
      </header>
      <DocumentToolbar
        documentId={documentId}
        documentReady={ready}
        documentName={getDocumentDisplayName(doc, documentId)}
        summaryText={displaySummary}
        isSummaryLoading={showSummaryLoading}
        onHighlightChunk={(chunk) => {
          setActiveHighlight(chunk);
          setIsMarkdownOpen(true);
        }}
        onOpenDocument={() => setIsMarkdownOpen(true)}
      />
      <div className="flex-1 flex min-h-0 overflow-hidden">
        <div
          className={`min-w-0 min-h-0 flex flex-col overflow-hidden transition-all duration-300 ease-out ${
            ready && isMarkdownOpen ? "w-1/2" : "w-full"
          }`}
        >
          {quickSummary && !ready && (
            <div className="px-4 pt-4 pb-2 border-b border-slate-800/80 bg-slate-950/70">
              <h2 className="text-sm font-semibold text-slate-100 mb-1">Quick document overview</h2>
              <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-line">{quickSummary}</p>
            </div>
          )}
          <ChatPanel
            documentId={documentId}
            documentReady={ready}
            documentSummary={displaySummary}
            suggestedQueries={displaySuggestedQueries}
            documentName={getDocumentDisplayName(doc, documentId)}
            isSummaryLoading={showSummaryLoading}
            showSummaryBlock={showSummaryBlock}
            onHighlightChunk={(chunk) => {
              setActiveHighlight(chunk);
              setIsMarkdownOpen(true);
            }}
          />
        </div>
        <div
          className={`relative min-w-0 min-h-0 flex flex-col overflow-hidden transition-all duration-300 ease-out ${
            ready ? (isMarkdownOpen ? "w-1/2 opacity-100" : "w-0 opacity-0 pointer-events-none") : "w-0"
          }`}
        >
          {ready ? (
            isMarkdownOpen && (
              <MarkdownViewer
                documentId={documentId}
                activeHighlight={activeHighlight}
                onClose={() => setIsMarkdownOpen(false)}
              />
            )
          ) : (
            <div className="hidden">
              <div className="animate-processing-fade-in max-w-sm w-full text-center">
                <div className="relative inline-flex justify-center mb-6">
                  <div
                    className="absolute inset-0 rounded-full bg-indigo-500/35 animate-ping"
                    style={{ animationDuration: "2s" }}
                  />
                  <div className="relative w-16 h-16 rounded-2xl bg-slate-900 border border-indigo-400/40 shadow-xl animate-processing-float flex items-center justify-center animate-processing-glow">
                    <svg
                      className="w-8 h-8 text-indigo-300 animate-pulse"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={1.5}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5v-7.5H8.25v7.5z"
                      />
                    </svg>
                  </div>
                </div>
                <h3 className="text-slate-100 font-semibold text-lg mb-1">
                  {(doc?.status === "processing" || doc?.status === "vectorized") && "Preparing your document"}
                </h3>
                <p className="text-slate-400 text-sm mb-6">
                  {doc?.status === "uploaded" && "Your file is in the queue. We’ll start shortly."}
                  {(doc?.status === "processing" || doc?.status === "vectorized") &&
                    "Markdown will appear here when ready."}
                </p>
                <div className="flex justify-center gap-2 mb-4">
                  {["uploaded", "processing", "vectorized", "ready"].map((step, i) => {
                    const isActive = step === doc?.status;
                    const isPast =
                      (step === "uploaded" &&
                        (doc?.status === "processing" || doc?.status === "vectorized" || doc?.status === "ready")) ||
                      (step === "processing" && (doc?.status === "vectorized" || doc?.status === "ready")) ||
                      (step === "vectorized" && doc?.status === "ready");
                    return (
                      <div
                        key={step}
                        className={`h-1.5 flex-1 max-w-12 rounded-full transition-all duration-500 ${
                          isPast
                            ? "bg-indigo-500"
                            : isActive
                              ? "animate-processing-shimmer bg-indigo-400/80"
                              : "bg-slate-700"
                        }`}
                        title={step}
                      />
                    );
                  })}
                </div>
                <div className="h-1 w-full rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all duration-700 ease-out"
                    style={{
                      width:
                        doc?.status === "uploaded"
                          ? "25%"
                          : doc?.status === "processing" || doc?.status === "vectorized"
                            ? "75%"
                            : "100%",
                    }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <ConfirmModal
        isOpen={showDeleteConfirm}
        title="Delete Document"
        message="Delete this chat and all associated data for this document? This cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
        confirmVariant="danger"
        onConfirm={handleDeleteConfirm}
        onCancel={handleDeleteCancel}
        onClose={handleDeleteCancel}
      />
    </div>
  );
}
