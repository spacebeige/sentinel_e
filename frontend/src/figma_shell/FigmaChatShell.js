/**
 * ============================================================
 * FigmaChatShell — Figma Visual Authority (Controlled Component)
 * ============================================================
 *
 * ARCHITECTURE ROLE:
 *   sentinel_e/frontend/src/App.js        → Logic Authority
 *   sentinel_e/frontend/src/figma_shell/  → Visual Authority
 *
 * This component is a direct port of figma_ui/src/app/components/ChatPage.tsx
 * converted to a controlled React component. ALL API logic has been removed.
 * All data flows through props from the sentinel_e state management layer.
 *
 * REQUIRED BINDINGS (per integration spec):
 *   input, setInput           — Controlled text input state
 *   selectedModel, setSelectedModel — Active model selection
 *   loading                   — Processing state indicator
 *   response                  — Latest backend response (for omega metadata)
 *   handleSubmit              — Delegates send action to sentinel_e handler
 *
 * LOCAL STATE (visual concerns only):
 *   showModelPicker, expandedMeta, showHistory, showSessionPanel,
 *   attachedFile, localFeedback — UI interactions that don't affect data flow
 *
 * ============================================================
 */

import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send, Sparkles, ChevronDown, Plus, Paperclip,
  Swords, Gem, FileSearch, X,
  Wifi, WifiOff, AlertCircle, ThumbsUp, ThumbsDown,
  History, ChevronRight,
  Activity, Brain, Shield, BarChart3, Zap,
  Skull, Loader2,
  MessageSquare, PanelRightOpen,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import StructuredOutput from '../components/structured/StructuredOutput';
import ThinkingAnimation from '../components/structured/ThinkingAnimation';
import AdvancedCopyMenu from '../components/AdvancedCopyMenu';
import { normalizeResponseText } from '../engines/responseNormalizer';
import memoryManager from '../engines/memoryManager';
import { getVisibility, hasAnyVisibleAnalytics } from '../engines/analyticsVisibilityController';

// ============================================================
// Visual Constants (from Figma design system)
// ============================================================

/** Legacy static fallback — only used when chatModels prop is absent */
export const MODELS = [
  { id: 'sentinel-std', name: 'Sentinel-E Standard', provider: 'Standard', color: '#3b82f6', category: 'standard', enabled: true },
  { id: 'sentinel-exp', name: 'Sentinel-E Pro', provider: 'Experimental', color: '#8b5cf6', category: 'experimental', enabled: true },
];

const PRO_SUB_MODES = [
  { id: 'debate', label: 'Debate Mode', iconKey: 'swords', color: '#ef4444', description: 'Argues both sides of a topic so you can decide', placeholder: 'Give me a topic to debate...' },
  { id: 'glass', label: 'Glass Mode', iconKey: 'gem', color: '#8b5cf6', description: 'Shows its full reasoning chain — nothing hidden', placeholder: "Ask something and I'll show my thinking..." },
  { id: 'evidence', label: 'Evidence Mode', iconKey: 'filesearch', color: '#06b6d4', description: 'Every claim backed by a cited source', placeholder: 'What do you need evidence for...' },
];

const SUB_MODE_ICONS = {
  swords: (cls) => <Swords className={cls} />,
  gem: (cls) => <Gem className={cls} />,
  filesearch: (cls) => <FileSearch className={cls} />,
};

/** Font stack matching Figma design system */
const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

// ============================================================
// FigmaChatShell — Main Visual Shell
// ============================================================

export default function FigmaChatShell({
  // === Required bindings (per integration spec) ===
  input,
  setInput,
  selectedModel,
  setSelectedModel,
  loading,
  response,
  handleSubmit,

  // === Chat state (from sentinel_e App.js) ===
  messages,
  serverStatus,
  activeChatId,
  history,
  sessionState,

  // === Mode state ===
  mode,
  setMode,
  subMode,
  setSubMode,
  killActive,
  setKillActive,

  // === Handlers ===
  onNewChat,
  onSelectRun,

  // === Engine pipeline (v4) ===
  pipelineSteps,

  // === Memory context (vNext) ===
  lastQueryText,
  lastResponseText,

  // === Cognitive governance (Section XII) ===
  governanceVerdict,

  // === Dynamic model registry ===
  chatModels: chatModelsProp,
  mcoModels,
}) {
  // ============================================================
  // RESOLVED MODELS — Dynamic from props, fallback to static
  // ============================================================
  const resolvedModels = chatModelsProp && chatModelsProp.length > 0 ? chatModelsProp : MODELS;

  // ============================================================
  // LOCAL UI STATE (visual concerns only — no data flow impact)
  // ============================================================
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showSessionPanel, setShowSessionPanel] = useState(false);
  const [expandedMeta, setExpandedMeta] = useState(null);
  const [attachedFile, setAttachedFile] = useState(null);
  const [attachedPreview, setAttachedPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [localFeedback, setLocalFeedback] = useState({});

  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);

  const isImageFile = (file) => file?.type?.startsWith('image/');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const backendOnline = serverStatus === 'online';
  // Ensemble is always active — sub_mode enrichment applies to all modes
  const activeSubMode = subMode || (response?.omega_metadata?.sub_mode) || null;

  const getSubModeIcon = (iconKey, cls) => {
    const iconFn = SUB_MODE_ICONS[iconKey];
    return iconFn ? iconFn(cls) : null;
  };

  // ============================================================
  // MESSAGE TRANSFORMATION
  // Adapts sentinel_e flat message format → Figma enriched format
  // sentinel_e: { role, content, timestamp }
  // Figma:      { id, role, content, timestamp: Date, mode?, confidence?, omegaMetadata?, ... }
  //
  // MODE ISOLATION: Standard mode messages are NEVER enriched with
  // debate/evidence/glass metadata. Only experimental mode with active
  // subMode gets engine-specific enrichment.
  // ============================================================
  const enhancedMessages = useMemo(() => {
    const welcome = {
      id: 'welcome',
      role: 'assistant',
      content: "Hello! I'm Sentinel-E, your AI assistant powered by multi-model intelligence. How can I help you today?",
      timestamp: new Date(),
    };

    if (!messages || messages.length === 0) return [welcome];

    return messages.map((msg, i) => {
      const base = {
        id: `msg-${i}`,
        role: msg.role,
        content: msg.role === 'assistant' ? normalizeResponseText(msg.content) : msg.content,
        timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date(),
        feedbackGiven: localFeedback[`msg-${i}`] || null,
      };

      // Enrich the LAST assistant message with omega metadata from response
      // Ensemble is always active — always attach metadata
      if (msg.role === 'assistant' && i === messages.length - 1 && response) {
        const resolvedMode = response.omega_metadata?.sub_mode || response.sub_mode || response.mode || response.omega_metadata?.mode || subMode || 'standard';
        return {
          ...base,
          mode: resolvedMode,
          chatId: response.chat_id || activeChatId,
          confidence: response.confidence,
          boundaryResult: response.boundary_result || response.omega_metadata?.boundary_result,
          reasoningTrace: response.reasoning_trace || response.omega_metadata?.reasoning_trace,
          confidenceEvolution: response.omega_metadata?.confidence_evolution,
          omegaMetadata: response.omega_metadata,
        };
      }

      return base;
    });
  }, [messages, response, subMode, activeChatId, localFeedback]);

  // Auto-scroll only on new message count (not metadata enrichment)
  const prevMsgCount = useRef(0);
  useEffect(() => {
    const count = enhancedMessages.length;
    if (count !== prevMsgCount.current || loading) {
      prevMsgCount.current = count;
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [enhancedMessages.length, loading]);

  // ============================================================
  // LOCAL HANDLERS (delegate to sentinel_e logic authority)
  // ============================================================

  const handleSendLocal = useCallback(() => {
    if (!input.trim() && !attachedFile) return;
    handleSubmit({ text: input.trim(), file: attachedFile });
    setInput('');
    setAttachedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [input, attachedFile, handleSubmit, setInput]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendLocal();
    }
  }, [handleSendLocal]);

  const handleModelSelect = useCallback((model) => {
    setSelectedModel(model);
    setShowModelPicker(false);
    if (model.category === 'standard') {
      setMode('standard');
      setSubMode(null);
    } else if (model.category === 'experimental') {
      setMode('experimental');
    }
  }, [setSelectedModel, setMode, setSubMode]);

  const handleSubModeToggle = useCallback((modeId) => {
    setSubMode(activeSubMode === modeId ? null : modeId);
  }, [setSubMode, activeSubMode]);

  const handleNewChatLocal = useCallback(() => {
    setExpandedMeta(null);
    setShowSessionPanel(false);
    setAttachedFile(null);
    setLocalFeedback({});
    onNewChat(); // ChatEngine handles memoryManager.newSession()
  }, [onNewChat]);

  const handleSelectRunLocal = useCallback((chat) => {
    onSelectRun(chat);
    setShowHistory(false);
    setLocalFeedback({});
  }, [onSelectRun]);

  const handleFeedback = useCallback((messageId, vote) => {
    setLocalFeedback(prev => ({ ...prev, [messageId]: vote }));

    // Wire to memory manager for adaptive learning
    const targetMsg = enhancedMessages.find(m => m.id === messageId);
    memoryManager.recordFeedback(vote, {
      responseLength: targetMsg?.content?.length || 0,
      hadAnalytics: !!(targetMsg?.omegaMetadata),
      hadCitations: !!(targetMsg?.omegaMetadata?.evidence_result),
      mode: mode,
      subMode: subMode,
    });

    // POST to backend /feedback endpoint (fire-and-forget)
    const API_BASE = require('../config').API_BASE;
    if (activeChatId) {
      const feedbackForm = new FormData();
      feedbackForm.append('run_id', activeChatId);
      feedbackForm.append('feedback', vote === 'up' ? 'positive' : 'negative');
      feedbackForm.append('rating', vote === 'up' ? '5' : '1');
      feedbackForm.append('mode', mode || 'standard');
      if (subMode) feedbackForm.append('sub_mode', subMode);
      fetch(`${API_BASE}/feedback`, { method: 'POST', body: feedbackForm }).catch(() => {});
    }
  }, [enhancedMessages, mode, subMode, activeChatId]);

  const attachFile = useCallback((file) => {
    if (!file) return;
    setAttachedFile(file);
    if (file.type?.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (ev) => setAttachedPreview(ev.target.result);
      reader.readAsDataURL(file);
    } else {
      setAttachedPreview(null);
    }
  }, []);

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files?.[0];
    if (file) attachFile(file);
  }, [attachFile]);

  const removeFile = useCallback(() => {
    setAttachedFile(null);
    setAttachedPreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, []);

  // -- Drag-and-drop handlers --
  const handleDragOver = useCallback((e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); }, []);
  const handleDragLeave = useCallback((e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); }, []);
  const handleDrop = useCallback((e) => {
    e.preventDefault(); e.stopPropagation(); setIsDragging(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) attachFile(file);
  }, [attachFile]);

  // ============================================================
  // RENDER HELPERS
  // ============================================================

  /**
   * renderCleanContent — Markdown-aware text renderer
   * Handles headings, bold, italic, lists, code blocks, links, and tables
   * via react-markdown with remark-gfm.
   */
  const renderCleanContent = (text) => {
    if (!text) return null;

    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Headings
          h1: ({ children }) => (
            <h1 style={{ fontFamily: FONT, fontSize: '20px', fontWeight: 700, lineHeight: 1.3, marginTop: '16px', marginBottom: '8px', color: 'inherit' }}>{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 style={{ fontFamily: FONT, fontSize: '17px', fontWeight: 700, lineHeight: 1.3, marginTop: '14px', marginBottom: '6px', color: 'inherit' }}>{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 style={{ fontFamily: FONT, fontSize: '15px', fontWeight: 700, lineHeight: 1.3, marginTop: '12px', marginBottom: '4px', color: 'inherit' }}>{children}</h3>
          ),
          // Paragraphs
          p: ({ children }) => (
            <p style={{ fontFamily: FONT, fontSize: '15px', lineHeight: 1.6, marginTop: '6px', marginBottom: '6px', color: 'inherit' }}>{children}</p>
          ),
          // Lists
          ul: ({ children }) => (
            <ul style={{ fontFamily: FONT, fontSize: '15px', lineHeight: 1.6, paddingLeft: '20px', marginTop: '4px', marginBottom: '4px', listStyleType: 'disc', color: 'inherit' }}>{children}</ul>
          ),
          ol: ({ children }) => (
            <ol style={{ fontFamily: FONT, fontSize: '15px', lineHeight: 1.6, paddingLeft: '20px', marginTop: '4px', marginBottom: '4px', listStyleType: 'decimal', color: 'inherit' }}>{children}</ol>
          ),
          li: ({ children }) => (
            <li style={{ marginBottom: '2px', color: 'inherit' }}>{children}</li>
          ),
          // Inline code
          code: ({ inline, className, children }) => {
            if (inline) {
              return (
                <code style={{
                  fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
                  fontSize: '13px', backgroundColor: 'rgba(0,0,0,0.05)',
                  padding: '1px 5px', borderRadius: '4px', color: 'inherit',
                }}>{children}</code>
              );
            }
            const lang = (className || '').replace('language-', '');
            return (
              <div className="my-3 rounded-xl overflow-hidden border border-black/5 dark:border-white/10">
                {lang && (
                  <div className="px-4 py-1.5 bg-[#f5f5f7] dark:bg-[#2a2a2e] border-b border-black/5 dark:border-white/10">
                    <span className="text-[#6e6e73] dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.03em' }}>
                      {lang}
                    </span>
                  </div>
                )}
                <pre className="p-4 bg-[#fafafa] dark:bg-[#1a1a1e] overflow-x-auto" style={{ margin: 0 }}>
                  <code className="text-[#1d1d1f] dark:text-[#e2e8f0]" style={{ fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace", fontSize: '13px', lineHeight: 1.5 }}>
                    {children}
                  </code>
                </pre>
              </div>
            );
          },
          pre: ({ children }) => <>{children}</>,
          // Bold / italic
          strong: ({ children }) => <strong style={{ fontWeight: 700, color: 'inherit' }}>{children}</strong>,
          em: ({ children }) => <em style={{ fontStyle: 'italic', color: 'inherit' }}>{children}</em>,
          // Links
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer" style={{ color: '#3b82f6', textDecoration: 'underline' }}>{children}</a>
          ),
          // Blockquote
          blockquote: ({ children }) => (
            <blockquote style={{ borderLeft: '3px solid #d1d5db', paddingLeft: '12px', margin: '8px 0', color: '#6e6e73', fontStyle: 'italic' }}>{children}</blockquote>
          ),
          // Table
          table: ({ children }) => (
            <div className="overflow-x-auto my-3">
              <table style={{ fontFamily: FONT, fontSize: '13px', borderCollapse: 'collapse', width: '100%' }}>{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th style={{ textAlign: 'left', padding: '6px 10px', borderBottom: '2px solid #e5e7eb', fontWeight: 600, fontSize: '12px', color: '#6e6e73' }}>{children}</th>
          ),
          td: ({ children }) => (
            <td style={{ padding: '6px 10px', borderBottom: '1px solid #f3f4f6', color: 'inherit' }}>{children}</td>
          ),
          // Horizontal rule
          hr: () => <hr style={{ border: 'none', borderTop: '1px solid #e5e7eb', margin: '12px 0' }} />,
        }}
      >
        {text}
      </ReactMarkdown>
    );
  };

  const renderConfidenceBar = (value, label, color) => (
    <div className="flex items-center gap-2">
      <span className="text-[#6e6e73] dark:text-[#94a3b8] w-20 flex-shrink-0"
        style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 500 }}>
        {label}
      </span>
      <div className="flex-1 h-1.5 bg-black/5 dark:bg-white/10 rounded-full overflow-hidden">
        <div className="h-full rounded-full"
          style={{ width: `${Math.round(value * 100)}%`, backgroundColor: color }} />
      </div>
      <span className="text-[#1d1d1f] dark:text-white w-10 text-right flex-shrink-0"
        style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600 }}>
        {Math.round(value * 100)}%
      </span>
    </div>
  );

  // ============================================================
  // OMEGA INSIGHTS PANEL
  // Renders expandable metadata for assistant messages
  // ============================================================
  const renderOmegaInsights = (message) => {
    const isExpanded = expandedMeta === message.id;
    const hasData = message.omegaMetadata || message.reasoningTrace ||
      message.confidenceEvolution || message.boundaryResult;
    if (!hasData || message.role !== 'assistant') return null;

    // MODE ISOLATION: Don't show Omega Insights for standard mode
    const isStandardMode = selectedModel?.category === 'standard';
    if (isStandardMode) return null;

    // ANALYTICS VISIBILITY: Check if insights should be shown based on
    // query complexity + user preferences + metric significance
    const visFlags = getVisibility({
      userQuery: lastQueryText || '',
      mode: mode,
      subMode: subMode,
      responseData: { omega_metadata: message.omegaMetadata, boundary_result: message.boundaryResult },
    });
    if (!hasAnyVisibleAnalytics(visFlags)) return null;

    return (
      <div className="mt-2">
        <button
          onClick={() => setExpandedMeta(isExpanded ? null : message.id)}
          className="flex items-center gap-1 px-2 py-1 rounded-lg hover:bg-black/5 transition-colors"
        >
          <Brain className="w-3 h-3 text-[#8b5cf6]" />
          <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#8b5cf6' }}>
            Analysis Details
          </span>
          <ChevronRight
            className="w-3 h-3 text-[#8b5cf6] transition-transform"
            style={{ transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}
          />
        </button>

        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
            >
              <div className="mt-2 p-3 rounded-xl bg-[#f5f5f7]/80 dark:bg-[#1c1c1e]/80 space-y-3">

                {/* Confidence Evolution */}
                {message.confidenceEvolution && (
                  <div>
                    <div className="flex items-center gap-1 mb-2">
                      <BarChart3 className="w-3 h-3 text-[#3b82f6]" />
                      <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#3b82f6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Confidence Evolution
                      </span>
                    </div>
                    <div className="space-y-1.5">
                      {renderConfidenceBar(message.confidenceEvolution.initial, 'Initial', '#aeaeb2')}
                      {message.confidenceEvolution.post_debate != null &&
                        renderConfidenceBar(message.confidenceEvolution.post_debate, 'Post-debate', '#ef4444')}
                      {message.confidenceEvolution.post_boundary != null &&
                        renderConfidenceBar(message.confidenceEvolution.post_boundary, 'Post-bound.', '#f59e0b')}
                      {message.confidenceEvolution.post_evidence != null &&
                        renderConfidenceBar(message.confidenceEvolution.post_evidence, 'Post-evid.', '#06b6d4')}
                      {message.confidenceEvolution.post_stress != null &&
                        renderConfidenceBar(message.confidenceEvolution.post_stress, 'Post-stress', '#8b5cf6')}
                      {renderConfidenceBar(message.confidenceEvolution.final, 'Final', '#10b981')}
                    </div>
                  </div>
                )}

                {/* Reasoning Trace */}
                {message.reasoningTrace && (
                  <div>
                    <div className="flex items-center gap-1 mb-2">
                      <Activity className="w-3 h-3 text-[#f59e0b]" />
                      <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#f59e0b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Reasoning Trace
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                      {[
                        { label: 'Passes', value: message.reasoningTrace.passes_executed },
                        { label: 'Assumptions', value: message.reasoningTrace.assumptions_extracted },
                        { label: 'Logic gaps', value: message.reasoningTrace.logical_gaps_detected },
                        { label: 'Boundary sev.', value: message.reasoningTrace.boundary_severity },
                      ].map((item) => (
                        <div key={item.label} className="flex items-center justify-between">
                          <span className="text-[#6e6e73] dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 400 }}>{item.label}</span>
                          <span className="text-[#1d1d1f] dark:text-white" style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600 }}>{item.value}</span>
                        </div>
                      ))}
                    </div>
                    <div className="flex gap-2 mt-1.5">
                      {message.reasoningTrace.self_critique_applied && (
                        <span className="px-1.5 py-0.5 rounded-md bg-[#f59e0b]/10 text-[#f59e0b]"
                          style={{ fontSize: '9px', fontWeight: 600 }}>Self-critique</span>
                      )}
                      {message.reasoningTrace.refinement_applied && (
                        <span className="px-1.5 py-0.5 rounded-md bg-[#10b981]/10 text-[#10b981]"
                          style={{ fontSize: '9px', fontWeight: 600 }}>Refined</span>
                      )}
                    </div>
                  </div>
                )}

                {/* Boundary Result */}
                {message.boundaryResult && message.boundaryResult.severity_score > 0 && (
                  <div>
                    <div className="flex items-center gap-1 mb-2">
                      <Shield className="w-3 h-3 text-[#ef4444]" />
                      <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#ef4444', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Boundary Evaluation
                      </span>
                    </div>
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span style={{ fontFamily: FONT, fontSize: '10px', color: '#6e6e73' }}>Risk Level</span>
                        <span className="px-1.5 py-0.5 rounded-md" style={{
                          fontSize: '9px', fontWeight: 600,
                          backgroundColor: message.boundaryResult.risk_level === 'HIGH' ? '#fef2f2'
                            : message.boundaryResult.risk_level === 'MEDIUM' ? '#fffbeb' : '#f0fdf4',
                          color: message.boundaryResult.risk_level === 'HIGH' ? '#ef4444'
                            : message.boundaryResult.risk_level === 'MEDIUM' ? '#f59e0b' : '#10b981',
                        }}>
                          {message.boundaryResult.risk_level}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-[#6e6e73] dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '10px' }}>Severity</span>
                        <span className="text-[#1d1d1f] dark:text-white" style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600 }}>
                          {message.boundaryResult.severity_score}/100
                        </span>
                      </div>
                      {message.boundaryResult.explanation && (
                        <p style={{ fontFamily: FONT, fontSize: '10px', color: '#6e6e73', lineHeight: 1.4 }}>
                          {message.boundaryResult.explanation}
                        </p>
                      )}
                      {message.boundaryResult.human_review_required && (
                        <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-[#fef2f2]">
                          <AlertCircle className="w-3 h-3 text-[#ef4444]" />
                          <span style={{ fontSize: '9px', fontWeight: 600, color: '#ef4444' }}>Human review required</span>
                        </div>
                      )}
                      {/* Risk dimensions */}
                      {message.boundaryResult.risk_dimensions && Object.keys(message.boundaryResult.risk_dimensions).length > 0 && (
                        <div className="space-y-1 mt-1">
                          {Object.entries(message.boundaryResult.risk_dimensions).map(([dim, val]) => (
                            renderConfidenceBar(val / 100, dim, '#ef4444')
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Fragility Index */}
                {message.omegaMetadata?.fragility_index != null && message.omegaMetadata.fragility_index > 0 && (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1">
                      <Zap className="w-3 h-3 text-[#f59e0b]" />
                      <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 500, color: '#6e6e73' }}>Fragility Index</span>
                    </div>
                    <span className="text-[#1d1d1f] dark:text-white" style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600 }}>
                      {(message.omegaMetadata.fragility_index * 100).toFixed(1)}%
                    </span>
                  </div>
                )}

                {/* Behavioral Risk */}
                {message.omegaMetadata?.behavioral_risk && (
                  <div>
                    <div className="flex items-center gap-1 mb-1">
                      <Activity className="w-3 h-3 text-[#8b5cf6]" />
                      <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#8b5cf6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Behavioral Risk
                      </span>
                      <span className="ml-auto px-1.5 py-0.5 rounded-md" style={{
                        fontSize: '9px', fontWeight: 600,
                        backgroundColor: message.omegaMetadata.behavioral_risk.risk_level === 'HIGH' ? '#fef2f2' : '#f0fdf4',
                        color: message.omegaMetadata.behavioral_risk.risk_level === 'HIGH' ? '#ef4444' : '#10b981',
                      }}>
                        {message.omegaMetadata.behavioral_risk.risk_level}
                      </span>
                    </div>
                    <div className="space-y-1">
                      {renderConfidenceBar(message.omegaMetadata.behavioral_risk.manipulation_probability, 'Manipulation', '#ef4444')}
                      {renderConfidenceBar(message.omegaMetadata.behavioral_risk.evasion_index, 'Evasion', '#f59e0b')}
                      {renderConfidenceBar(message.omegaMetadata.behavioral_risk.confidence_inflation, 'Inflation', '#8b5cf6')}
                    </div>
                  </div>
                )}

                {/* Evidence Sources */}
                {message.omegaMetadata?.evidence_result && message.omegaMetadata.evidence_result.sources?.length > 0 && (
                  <div>
                    <div className="flex items-center gap-1 mb-1">
                      <FileSearch className="w-3 h-3 text-[#06b6d4]" />
                      <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#06b6d4', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Evidence Sources ({message.omegaMetadata.evidence_result.source_count})
                      </span>
                    </div>
                    <div className="space-y-1">
                      {message.omegaMetadata.evidence_result.sources.slice(0, 5).map((src, i) => (
                        <div key={i} className="flex items-start gap-1.5">
                          <span style={{ fontFamily: FONT, fontSize: '9px', fontWeight: 600, color: '#06b6d4' }}>[{i + 1}]</span>
                          <div>
                            <span className="text-[#1d1d1f] dark:text-white" style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 500 }}>{src.title || src.domain}</span>
                            {src.url && (
                              <a href={src.url} target="_blank" rel="noopener noreferrer"
                                className="block text-[#3b82f6] hover:underline" style={{ fontSize: '9px' }}>
                                {src.url.length > 50 ? src.url.slice(0, 50) + '...' : src.url}
                              </a>
                            )}
                            <span style={{ fontFamily: FONT, fontSize: '9px', color: '#6e6e73' }}>
                              Reliability: {Math.round(src.reliability_score * 100)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Engine version tag */}
                {message.omegaMetadata?.omega_version && (
                  <div className="flex items-center justify-between pt-1 border-t border-black/5">
                    <span style={{ fontFamily: FONT, fontSize: '9px', color: '#aeaeb2' }}>
                      Sentinel-E Pro v{message.omegaMetadata.omega_version}
                    </span>
                    {message.omegaMetadata.session_state?.inferred_domain && (
                      <span style={{ fontFamily: FONT, fontSize: '9px', color: '#aeaeb2' }}>
                        {message.omegaMetadata.session_state.inferred_domain}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  };

  // ============================================================
  // SESSION ANALYTICS PANEL (renders from pre-loaded sessionState)
  // ============================================================
  const renderSessionPanel = () => {
    if (!backendOnline) {
      return (
        <div className="px-4 py-12 text-center">
          <WifiOff className="w-8 h-8 text-[#aeaeb2] mx-auto mb-2" />
          <p style={{ fontFamily: FONT, fontSize: '12px', color: '#aeaeb2' }}>Backend offline</p>
        </div>
      );
    }

    if (!activeChatId) {
      return (
        <div className="px-4 py-12 text-center">
          <MessageSquare className="w-8 h-8 text-[#aeaeb2] mx-auto mb-2" />
          <p style={{ fontFamily: FONT, fontSize: '12px', color: '#aeaeb2' }}>
            Start a conversation to see analytics
          </p>
        </div>
      );
    }

    if (!sessionState) {
      return (
        <div className="px-4 py-12 text-center">
          <Loader2 className="w-6 h-6 text-[#aeaeb2] mx-auto mb-2 animate-spin" />
          <p style={{ fontFamily: FONT, fontSize: '12px', color: '#aeaeb2' }}>Loading session data...</p>
        </div>
      );
    }

    return (
      <div className="p-4 space-y-4">
        {/* Session Identity */}
        <div className="p-3 rounded-xl bg-gradient-to-br from-[#f5f5f7] to-[#eeeef0]">
          <span className="text-[#1d1d1f] block truncate"
            style={{ fontFamily: FONT, fontSize: '15px', fontWeight: 600 }}>
            {sessionState.chat_name || 'Active Session'}
          </span>
          {sessionState.primary_goal && (
            <div className="flex items-center gap-2 mt-1.5">
              <Activity className="w-3 h-3 text-[#6e6e73]" />
              <span style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73' }}>
                {String(sessionState.primary_goal).replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
              </span>
            </div>
          )}
          {sessionState.inferred_domain && (
            <div className="flex items-center gap-2 mt-1">
              <Brain className="w-3 h-3 text-[#6e6e73]" />
              <span style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73' }}>
                {sessionState.inferred_domain}
              </span>
            </div>
          )}
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-3 gap-2">
          {[
            { value: sessionState.message_count || 0, label: 'Messages' },
            { value: sessionState.boundary_history_count || 0, label: 'Boundaries' },
            { value: sessionState.reasoning_depth || 'N/A', label: 'Depth' },
          ].map((stat) => (
            <div key={stat.label} className="p-2 rounded-xl bg-[#f5f5f7] dark:bg-[#1c1c1e] text-center">
              <span className="block text-[#1d1d1f] dark:text-[#f1f5f9]"
                style={{ fontFamily: FONT, fontSize: '18px', fontWeight: 700 }}>
                {stat.value}
              </span>
              <span style={{ fontFamily: FONT, fontSize: '9px', color: '#aeaeb2', fontWeight: 500 }}>
                {stat.label}
              </span>
            </div>
          ))}
        </div>

        {/* Confidence & Risk Metrics */}
        {sessionState.session_confidence != null && (
          <div className="space-y-2">
            {renderConfidenceBar(sessionState.session_confidence, 'Confidence', '#3b82f6')}
            {sessionState.fragility_index != null &&
              renderConfidenceBar(sessionState.fragility_index, 'Fragility', '#f59e0b')}
            {sessionState.disagreement_score != null &&
              renderConfidenceBar(sessionState.disagreement_score, 'Disagreement', '#ef4444')}
          </div>
        )}

        {/* Boundary Trend */}
        {sessionState.boundary_trend && (
          <div className="flex items-center justify-between p-2 rounded-xl bg-[#f5f5f7] dark:bg-[#1c1c1e]">
            <span style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73' }}>Boundary Trend</span>
            <span className="px-2 py-0.5 rounded-md" style={{
              fontFamily: FONT, fontSize: '10px', fontWeight: 600,
              backgroundColor: sessionState.boundary_trend === 'increasing' ? '#fef2f2' : '#f0fdf4',
              color: sessionState.boundary_trend === 'increasing' ? '#ef4444' : '#10b981',
            }}>
              {sessionState.boundary_trend}
            </span>
          </div>
        )}

        {/* User Expertise */}
        {sessionState.user_expertise_score != null && (
          <div>
            {renderConfidenceBar(sessionState.user_expertise_score, 'User Expertise', '#8b5cf6')}
          </div>
        )}
      </div>
    );
  };

  // ============================================================
  // MAIN RENDER
  // ============================================================
  return (
    <div className="flex bg-[#f5f5f7] dark:bg-[#0f0f10]" style={{ height: 'calc(100vh - 56px)' }}>

      {/* ==================== HISTORY SIDEBAR ==================== */}
      <AnimatePresence>
        {showHistory && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 280, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
            className="h-full border-r border-black/5 dark:border-white/10 bg-white/80 dark:bg-[#1c1c1e]/80 backdrop-blur-xl overflow-hidden flex-shrink-0"
          >
            <div className="h-full flex flex-col">
              <div className="flex items-center justify-between px-4 py-3 border-b border-black/5 dark:border-white/5">
                <span className="text-[#1d1d1f] dark:text-[#f1f5f9]"
                  style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}>
                  Chat History
                </span>
                <button onClick={() => setShowHistory(false)}
                  className="p-1 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                  <X className="w-4 h-4 text-[#6e6e73] dark:text-[#94a3b8]" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto">
                {!backendOnline ? (
                  <div className="px-4 py-8 text-center">
                    <WifiOff className="w-8 h-8 text-[#aeaeb2] mx-auto mb-2" />
                    <p style={{ fontFamily: FONT, fontSize: '12px', color: '#aeaeb2' }}>
                      Connect to backend to see history
                    </p>
                  </div>
                ) : history.length === 0 ? (
                  <div className="px-4 py-8 text-center">
                    <MessageSquare className="w-8 h-8 text-[#aeaeb2] mx-auto mb-2" />
                    <p style={{ fontFamily: FONT, fontSize: '12px', color: '#aeaeb2' }}>
                      No previous chats
                    </p>
                  </div>
                ) : (
                  <div className="p-2 space-y-0.5">
                    {history.map((chat) => (
                      <button
                        key={chat.id}
                        onClick={() => handleSelectRunLocal(chat)}
                        className={`w-full text-left px-3 py-2.5 rounded-xl transition-colors ${
                          activeChatId === chat.id ? 'bg-[#e8e8ed] dark:bg-white/10' : 'hover:bg-[#f5f5f7] dark:hover:bg-[#1c1c1e]'
                        }`}
                      >
                        <div className="text-[#1d1d1f] dark:text-[#f1f5f9] truncate"
                          style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 500 }}>
                          {chat.summary || chat.name || 'Untitled Chat'}
                        </div>
                        <div className="flex items-center gap-2 mt-0.5">
                          <span className="px-1.5 py-0.5 rounded-md" style={{
                            fontFamily: FONT, fontSize: '9px', fontWeight: 600,
                            backgroundColor: chat.mode === 'experimental' ? '#f3e8ff' : '#e0f2fe',
                            color: chat.mode === 'experimental' ? '#8b5cf6' : '#3b82f6',
                          }}>
                            {chat.mode || 'standard'}
                          </span>
                          <span style={{ fontFamily: FONT, fontSize: '10px', color: '#aeaeb2' }}>
                            {new Date(chat.timestamp).toLocaleDateString()}
                          </span>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ==================== MAIN CHAT AREA ==================== */}
      <div className="flex-1 flex flex-col min-w-0">

        {/* ---------- CHAT HEADER ---------- */}
        <div className="flex items-center justify-between px-4 py-3 bg-white/80 dark:bg-[#1c1c1e]/80 backdrop-blur-xl border-b border-black/5 dark:border-white/10">
          <div className="flex items-center gap-2">
            {/* History toggle */}
            <button
              onClick={() => setShowHistory(!showHistory)}
              className={`p-2 rounded-xl transition-colors ${showHistory ? 'bg-[#e8e8ed] dark:bg-white/10' : 'hover:bg-black/5 dark:hover:bg-white/5'}`}
              title="Chat History"
            >
              <History className="w-4 h-4 text-[#6e6e73] dark:text-[#94a3b8]" />
            </button>

            {/* Model picker trigger */}
            <button
              onClick={() => setShowModelPicker(!showModelPicker)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
            >
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: selectedModel.color }} />
              <span className="text-[#1d1d1f] dark:text-[#f1f5f9]"
                style={{ fontFamily: FONT, fontSize: '15px', fontWeight: 600 }}>
                {selectedModel.name}
              </span>
              <ChevronDown className="w-3.5 h-3.5 text-[#6e6e73] dark:text-[#94a3b8]" />
            </button>

            {/* Mode badge — shows active mode/model */}
            {selectedModel && !selectedModel.isMeta && selectedModel.id !== 'sentinel-std' && selectedModel.id !== 'sentinel-exp' ? (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg"
                style={{ backgroundColor: selectedModel.color + '15' }}>
                <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: selectedModel.color }} />
                <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: selectedModel.color }}>
                  Running: {selectedModel.name}
                </span>
              </div>
            ) : selectedModel.id === 'sentinel-exp' ? (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#8b5cf6]/10">
                <div className="w-1.5 h-1.5 rounded-full bg-[#8b5cf6]" />
                <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: '#8b5cf6' }}>
                  Sentinel-E Pro{activeSubMode ? ` — ${PRO_SUB_MODES.find(m => m.id === activeSubMode)?.label || ''}` : ''}
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#3b82f6]/10">
                <div className="w-1.5 h-1.5 rounded-full bg-[#3b82f6]" />
                <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: '#3b82f6' }}>
                  Sentinel-E Standard
                </span>
              </div>
            )}

            {/* Connection Status */}
            <div className="flex items-center gap-1.5">
              {serverStatus === 'unknown' ? (
                <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-[#f5f5f7] dark:bg-white/10">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#aeaeb2] animate-pulse" />
                  <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 500, color: '#aeaeb2' }}>
                    Connecting...
                  </span>
                </div>
              ) : backendOnline ? (
                <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-[#d1fae5]/60">
                  <Wifi className="w-3 h-3 text-[#10b981]" />
                  <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 500, color: '#10b981' }}>
                    Live
                  </span>
                </div>
              ) : (
                <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-[#fef3c7]/60">
                  <WifiOff className="w-3 h-3 text-[#f59e0b]" />
                  <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 500, color: '#f59e0b' }}>
                    Offline
                  </span>
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-1">
            {/* Session Analytics toggle */}
            {activeChatId && (
              <button
                onClick={() => setShowSessionPanel(!showSessionPanel)}
                className={`p-2 rounded-xl transition-colors ${showSessionPanel ? 'bg-[#f3e8ff]' : 'hover:bg-black/5'}`}
                title="Session Analytics"
              >
                <PanelRightOpen className={`w-4 h-4 ${showSessionPanel ? 'text-[#8b5cf6]' : 'text-[#6e6e73]'}`} />
              </button>
            )}

            {/* Kill switch (Pro mode only) */}
            {selectedModel.id === 'sentinel-exp' && activeChatId && (
              <button
                onClick={() => setKillActive(!killActive)}
                className={`p-2 rounded-xl transition-colors ${killActive ? 'bg-[#fef2f2]' : 'hover:bg-[#fef2f2]'}`}
                title={killActive ? 'Kill Mode Active — next send uses kill endpoint' : 'Activate Kill Diagnostic'}
              >
                <Skull className={`w-4 h-4 ${killActive ? 'text-[#ef4444]' : 'text-[#6e6e73]'}`} />
              </button>
            )}

            {/* New Chat */}
            <button onClick={handleNewChatLocal}
              className="p-2 rounded-xl hover:bg-black/5 transition-colors" title="New Chat">
              <Plus className="w-5 h-5 text-[#6e6e73]" />
            </button>
          </div>
        </div>

        {/* ---------- OFFLINE BANNER ---------- */}
        <AnimatePresence>
          {!backendOnline && serverStatus !== 'unknown' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="bg-[#fffbeb] border-b border-[#f59e0b]/20 px-4 py-2 flex items-center gap-2"
            >
              <AlertCircle className="w-3.5 h-3.5 text-[#f59e0b] flex-shrink-0" />
              <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 500, color: '#92400e' }}>
                Backend offline — start your FastAPI server at localhost:8000 for live AI.
              </span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ---------- MODEL PICKER DROPDOWN ---------- */}
        <AnimatePresence>
          {showModelPicker && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setShowModelPicker(false)} />
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute top-16 left-16 z-50 w-72 p-2 rounded-2xl bg-white/95 dark:bg-[#1c1c1e]/95 backdrop-blur-xl shadow-2xl shadow-black/10 border border-black/5 dark:border-white/10"
              >
                {/* Standard models */}
                <div className="px-3 pt-2 pb-1 text-[#6e6e73] dark:text-[#94a3b8]"
                  style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                  Standard
                </div>
                {resolvedModels.filter(m => m.category === 'standard').map((model) => {
                  const isDisabled = model.enabled === false;
                  return (
                    <div key={model.id} className="relative group">
                      <button
                        onClick={() => !isDisabled && handleModelSelect(model)}
                        disabled={isDisabled}
                        className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors ${
                          isDisabled
                            ? 'opacity-40 cursor-not-allowed'
                            : selectedModel.id === model.id ? 'bg-[#f5f5f7] dark:bg-white/10' : 'hover:bg-[#f5f5f7] dark:hover:bg-white/5'
                        }`}
                      >
                        <div className="w-8 h-8 rounded-xl flex items-center justify-center"
                          style={{ backgroundColor: (isDisabled ? '#9ca3af' : model.color) + '20' }}>
                          <Sparkles className="w-4 h-4" style={{ color: isDisabled ? '#9ca3af' : model.color }} />
                        </div>
                        <div className="text-left flex-1 min-w-0">
                          <div className="text-[#1d1d1f] dark:text-[#f1f5f9] truncate"
                            style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}>
                            {model.name}
                          </div>
                          {model.provider && !model.isMeta && (
                            <div className="text-[#6e6e73] dark:text-[#94a3b8] truncate"
                              style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 400 }}>
                              {model.provider}{model.role ? ` \u00b7 ${model.role}` : ''}
                            </div>
                          )}
                        </div>
                        {isDisabled && (
                          <span className="px-1.5 py-0.5 rounded-md bg-[#fef2f2] dark:bg-red-500/10 text-[#ef4444] flex-shrink-0"
                            style={{ fontFamily: FONT, fontSize: '9px', fontWeight: 600 }}>
                            OFF
                          </span>
                        )}
                        {!isDisabled && selectedModel.id === model.id && (
                          <div className="ml-auto w-5 h-5 rounded-full bg-[#007aff] flex items-center justify-center flex-shrink-0">
                            <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
                              <path d="M1 4L3.5 6.5L9 1" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                          </div>
                        )}
                      </button>
                    </div>
                  );
                })}

                <div className="my-1.5 mx-3 border-t border-black/5 dark:border-white/10" />

                {/* Experimental models */}
                <div className="px-3 pt-2 pb-1 text-[#6e6e73] dark:text-[#94a3b8]"
                  style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                  Experimental
                </div>
                {resolvedModels.filter(m => m.category === 'experimental').map((model) => (
                  <button
                    key={model.id}
                    onClick={() => handleModelSelect(model)}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors ${
                      selectedModel.id === model.id ? 'bg-[#f5f5f7] dark:bg-white/10' : 'hover:bg-[#f5f5f7] dark:hover:bg-white/5'
                    }`}
                  >
                    <div className="w-8 h-8 rounded-xl flex items-center justify-center"
                      style={{ backgroundColor: model.color + '20' }}>
                      <Sparkles className="w-4 h-4" style={{ color: model.color }} />
                    </div>
                    <div className="text-left">
                      <div className="text-[#1d1d1f] dark:text-[#f1f5f9]"
                        style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}>
                        {model.name}
                      </div>
                    </div>
                    {selectedModel.id === model.id && (
                      <div className="ml-auto w-5 h-5 rounded-full bg-[#007aff] flex items-center justify-center">
                        <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
                          <path d="M1 4L3.5 6.5L9 1" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      </div>
                    )}
                  </button>
                ))}
              </motion.div>
            </>
          )}
        </AnimatePresence>

        {/* ---------- MESSAGES ---------- */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-5xl mx-auto space-y-4">
            <AnimatePresence>
              {enhancedMessages.map((message) => {
                const msgMode = message.mode
                  ? PRO_SUB_MODES.find(m => m.id === message.mode) : null;

                return (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.15 }}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`${
                        message.role === 'user'
                          ? 'max-w-[85%] sm:max-w-[70%] rounded-[20px] rounded-br-md bg-blue-600 dark:bg-blue-500 text-white px-4 py-3'
                          : 'max-w-[95%] sm:max-w-[85%] rounded-[20px] rounded-bl-md bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/10 text-[#1d1d1f] dark:text-white shadow-sm overflow-hidden'
                      }`}
                      style={
                        message.role === 'assistant' && msgMode
                          ? { borderLeft: `3px solid ${msgMode.color}` }
                          : undefined
                      }
                    >
                      {/* Mode badge for assistant messages */}
                      {message.role === 'assistant' && msgMode && (
                        <div className="flex items-center gap-1.5 px-4 py-1.5"
                          style={{ backgroundColor: msgMode.color + '0a' }}>
                          <div className="flex items-center justify-center w-4 h-4"
                            style={{ color: msgMode.color }}>
                            {getSubModeIcon(msgMode.iconKey, 'w-3.5 h-3.5')}
                          </div>
                          <span style={{
                            fontFamily: FONT, fontSize: '11px', fontWeight: 600,
                            color: msgMode.color, letterSpacing: '0.03em', textTransform: 'uppercase',
                          }}>
                            {msgMode.label}
                          </span>
                          {message.confidence != null && (
                            <span className="ml-auto"
                              style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 500, color: '#6e6e73' }}>
                              {Math.round(message.confidence * 100)}%{' '}
                              {message.confidence >= 0.85 ? '(High)' :
                               message.confidence >= 0.65 ? '(Moderate)' :
                               message.confidence >= 0.40 ? '(Low)' : '(Unstable)'}
                            </span>
                          )}
                        </div>
                      )}

                      <div className={message.role === 'assistant' ? 'px-4 py-3' : ''}>
                        {/* Clean text rendering — NO markdown, NO raw symbols */}
                        <div style={{ fontFamily: FONT, fontSize: '15px', lineHeight: 1.6, fontWeight: 400 }}>
                          {renderCleanContent(message.content)}
                        </div>

                        {/* Structured Output — renders for all modes (ensemble always active) */}
                        {message.role === 'assistant' && message.omegaMetadata && (
                          (() => {
                            const meta = message.omegaMetadata;
                            const hasStructuredData = meta.aggregation_result || meta.forensic_result || meta.audit_result || meta.debate_result || meta.ensemble_metrics || meta.model_outputs;
                            
                            if (!hasStructuredData) return null;
                            
                            // Determine rendering mode from metadata
                            const messageMode = message.mode || activeSubMode || 'standard';
                            const resolvedSubMode = messageMode === 'debate' ? 'debate' : (meta.ensemble_metrics ? 'ensemble' : messageMode);
                            
                            return (
                              <div className="mt-3 -mx-4 px-4 pt-3 border-t border-black/5">
                                <StructuredOutput
                                  result={{
                                    ...response,
                                    omega_metadata: meta,
                                    sub_mode: resolvedSubMode,
                                  }}
                                  activeSubMode={resolvedSubMode}
                                />
                              </div>
                            );
                          })()
                        )}

                        {/* Boundary warning — professional card style */}
                        {message.role === 'assistant' && message.boundaryResult &&
                          message.boundaryResult.severity_score > 40 &&
                          (() => {
                            const vis = getVisibility({
                              userQuery: lastQueryText || '',
                              mode, subMode,
                              responseData: { omega_metadata: message.omegaMetadata, boundary_result: message.boundaryResult },
                            });
                            return vis.showBoundary;
                          })() && (
                          <div className="flex items-center gap-2 mt-2 px-3 py-2 rounded-xl"
                            style={{
                              backgroundColor: message.boundaryResult.risk_level === 'HIGH' ? '#fef2f2' : '#fffbeb',
                              border: `1px solid ${message.boundaryResult.risk_level === 'HIGH' ? '#fecaca' : '#fde68a'}`,
                            }}>
                            <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" style={{
                              color: message.boundaryResult.risk_level === 'HIGH' ? '#ef4444' : '#f59e0b',
                            }} />
                            <span style={{
                              fontFamily: FONT, fontSize: '12px', fontWeight: 500,
                              color: message.boundaryResult.risk_level === 'HIGH' ? '#991b1b' : '#92400e',
                            }}>
                              {message.boundaryResult.risk_level === 'HIGH'
                                ? 'High divergence detected between models. Review with caution.'
                                : 'Moderate divergence detected. Some model disagreement present.'}
                            </span>
                          </div>
                        )}

                        {/* Omega Insights */}
                        {renderOmegaInsights(message)}

                        {/* Timestamp + Feedback */}
                        <div className="flex items-center justify-between mt-1">
                          <div className={message.role === 'user' ? 'text-white/50' : 'text-[#6e6e73]'}
                            style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 400 }}>
                            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </div>

                          {message.role === 'assistant' && message.id !== 'welcome' && (
                            <div className="flex items-center gap-1 ml-2">
                              <AdvancedCopyMenu message={message} />
                              <button
                                onClick={() => handleFeedback(message.id, 'up')}
                                className={`p-1 rounded-md transition-colors ${
                                  message.feedbackGiven === 'up' ? 'bg-[#d1fae5]' : 'hover:bg-black/5'
                                }`}
                                disabled={!!message.feedbackGiven}
                              >
                                <ThumbsUp className={`w-3 h-3 ${
                                  message.feedbackGiven === 'up' ? 'text-[#10b981]' : 'text-[#aeaeb2]'
                                }`} />
                              </button>
                              <button
                                onClick={() => handleFeedback(message.id, 'down')}
                                className={`p-1 rounded-md transition-colors ${
                                  message.feedbackGiven === 'down' ? 'bg-[#fef2f2]' : 'hover:bg-black/5'
                                }`}
                                disabled={!!message.feedbackGiven}
                              >
                                <ThumbsDown className={`w-3 h-3 ${
                                  message.feedbackGiven === 'down' ? 'text-[#ef4444]' : 'text-[#aeaeb2]'
                                }`} />
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>

            {/* Thinking Animation — pipeline-aware loading indicator */}
            {loading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-start"
              >
                <div
                  className="max-w-[95%] sm:max-w-[85%] rounded-[20px] rounded-bl-md bg-white dark:bg-[#1c1c1e] border shadow-sm overflow-hidden"
                  style={{
                    borderColor: activeSubMode
                      ? (PRO_SUB_MODES.find(m => m.id === activeSubMode)?.color || '#6e6e73') + '30'
                      : 'rgba(0,0,0,0.05)',
                    borderLeftWidth: activeSubMode ? '3px' : '1px',
                    borderLeftColor: activeSubMode
                      ? PRO_SUB_MODES.find(m => m.id === activeSubMode)?.color
                      : 'rgba(0,0,0,0.05)',
                  }}
                >
                  <ThinkingAnimation
                    steps={pipelineSteps || []}
                    activeColor={activeSubMode
                      ? PRO_SUB_MODES.find(m => m.id === activeSubMode)?.color
                      : '#3b82f6'}
                  />
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ---------- INPUT AREA ---------- */}
        <div className="px-4 pb-6 pt-2">
          <div className="max-w-5xl mx-auto">
            <div
              ref={dropZoneRef}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`relative flex flex-col gap-0 p-2 rounded-[28px] bg-white dark:bg-[#1c1c1e] shadow-lg transition-colors duration-200 ${
                isDragging ? 'ring-2 ring-[#3b82f6] ring-offset-2 dark:ring-offset-[#0f0f10]' : ''
              }`}
              style={{
                borderWidth: '1px',
                borderStyle: 'solid',
                borderColor: isDragging
                  ? '#3b82f6'
                  : activeSubMode
                    ? (PRO_SUB_MODES.find(m => m.id === activeSubMode)?.color || '#007aff') + '40'
                    : 'rgba(0,0,0,0.1)',
                boxShadow: activeSubMode
                  ? `0 4px 24px -4px ${PRO_SUB_MODES.find(m => m.id === activeSubMode)?.color}20, 0 0 0 1px ${PRO_SUB_MODES.find(m => m.id === activeSubMode)?.color}15`
                  : '0 4px 6px -1px rgba(0,0,0,0.05)',
              }}
            >
              {/* Drag overlay */}
              <AnimatePresence>
                {isDragging && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 z-10 flex items-center justify-center rounded-[28px] bg-[#3b82f6]/10 dark:bg-[#3b82f6]/20 border-2 border-dashed border-[#3b82f6] pointer-events-none"
                  >
                    <span className="text-[#3b82f6] font-semibold" style={{ fontFamily: FONT, fontSize: '14px' }}>
                      Drop file here
                    </span>
                  </motion.div>
                )}
              </AnimatePresence>
              {/* File attachment preview */}
              <AnimatePresence>
                {attachedFile && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                  >
                    <div className="flex items-center gap-2 px-3 py-2 mx-1 mt-1 rounded-xl bg-[#f5f5f7] dark:bg-white/10">
                      {attachedPreview ? (
                        <img src={attachedPreview} alt="preview" className="w-10 h-10 rounded-lg object-cover flex-shrink-0" />
                      ) : (
                        <Paperclip className="w-3.5 h-3.5 text-[#6e6e73] dark:text-[#94a3b8] flex-shrink-0" />
                      )}
                      <div className="flex-1 min-w-0">
                        <span className="block truncate text-[#1d1d1f] dark:text-[#f1f5f9]"
                          style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 500 }}>
                          {attachedFile.name}
                        </span>
                        <span style={{ fontFamily: FONT, fontSize: '10px', color: '#aeaeb2' }}>
                          {(attachedFile.size / 1024).toFixed(1)} KB
                          {isImageFile(attachedFile) && ' · Image'}
                        </span>
                      </div>
                      <button onClick={removeFile}
                        className="p-0.5 rounded-md hover:bg-black/10 dark:hover:bg-white/10 transition-colors">
                        <X className="w-3 h-3 text-[#6e6e73] dark:text-[#94a3b8]" />
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Sentinel-E Pro Submodes (shown when experimental model selected) */}
              <AnimatePresence>
                {selectedModel.id === 'sentinel-exp' && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                  >
                    <div className="flex flex-col gap-2 px-1 pb-2 pt-1">
                      <div className="flex items-center gap-1.5">
                        {PRO_SUB_MODES.map((sm) => {
                          const isActive = activeSubMode === sm.id;
                          return (
                            <button
                              key={sm.id}
                              onClick={() => handleSubModeToggle(sm.id)}
                              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full transition-all duration-200 ${
                                isActive ? 'text-white shadow-md' : 'bg-[#f5f5f7] dark:bg-white/10 text-[#6e6e73] dark:text-[#94a3b8] hover:bg-[#e8e8ed] dark:hover:bg-white/15'
                              }`}
                              style={isActive ? {
                                backgroundColor: sm.color,
                                boxShadow: `0 2px 8px ${sm.color}40`,
                              } : undefined}
                            >
                              {getSubModeIcon(sm.iconKey, 'w-3.5 h-3.5')}
                              <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600 }}>
                                {sm.label}
                              </span>
                            </button>
                          );
                        })}
                      </div>

                      {/* Active mode description */}
                      <AnimatePresence mode="wait">
                        {activeSubMode && (
                          <motion.div
                            key={activeSubMode}
                            initial={{ opacity: 0, y: -4 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -4 }}
                            transition={{ duration: 0.15 }}
                            className="flex items-center gap-2 px-2.5 py-1.5 rounded-xl"
                            style={{
                              backgroundColor: (PRO_SUB_MODES.find(m => m.id === activeSubMode)?.color || '#007aff') + '08',
                            }}
                          >
                            <div className="w-1 h-4 rounded-full"
                              style={{ backgroundColor: PRO_SUB_MODES.find(m => m.id === activeSubMode)?.color }} />
                            <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 400, color: '#6e6e73' }}>
                              {PRO_SUB_MODES.find(m => m.id === activeSubMode)?.description}
                            </span>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Input Row */}
              <div className="flex items-end gap-2">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className={`p-2 rounded-full transition-colors flex-shrink-0 ${
                    attachedFile ? 'bg-[#e0f2fe] dark:bg-[#1e3a5f]' : 'hover:bg-black/5 dark:hover:bg-white/5'
                  }`}
                  title="Attach file"
                >
                  <Paperclip className={`w-5 h-5 ${attachedFile ? 'text-[#3b82f6]' : 'text-[#6e6e73] dark:text-[#94a3b8]'}`} />
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  onChange={handleFileSelect}
                  accept=".txt,.pdf,.md,.json,.csv,.py,.js,.ts,.jsx,.tsx,.png,.jpg,.jpeg,.webp,.gif,.docx,.doc"
                />

                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={
                    activeSubMode
                      ? PRO_SUB_MODES.find(m => m.id === activeSubMode)?.placeholder
                      : 'Message Sentinel-E...'
                  }
                  rows={1}
                  className="flex-1 resize-none bg-transparent outline-none py-2 px-1 max-h-32 text-[#1d1d1f] dark:text-[#f1f5f9] placeholder-[#aeaeb2] dark:placeholder-[#64748b]"
                  style={{ fontFamily: FONT, fontSize: '16px', lineHeight: 1.5, fontWeight: 400 }}
                />

                <button
                  onClick={handleSendLocal}
                  disabled={(!input.trim() && !attachedFile) || loading}
                  className="p-2 rounded-full flex-shrink-0 transition-all"
                  style={{
                    backgroundColor: (!input.trim() && !attachedFile) || loading
                      ? '#e5e5ea'
                      : activeSubMode
                        ? PRO_SUB_MODES.find(m => m.id === activeSubMode)?.color
                        : '#007aff',
                    color: (input.trim() || attachedFile) && !loading ? 'white' : '#aeaeb2',
                    boxShadow: (input.trim() || attachedFile) && !loading && activeSubMode
                      ? `0 4px 12px ${PRO_SUB_MODES.find(m => m.id === activeSubMode)?.color}40`
                      : (input.trim() || attachedFile) && !loading
                        ? '0 4px 12px rgba(0,122,255,0.3)'
                        : 'none',
                  }}
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
            </div>

            <p className="text-center mt-2 text-[#aeaeb2]"
              style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 400 }}>
              {backendOnline
                ? `Connected to Sentinel-E${activeChatId ? ` · Session active` : ''}`
                : 'Sentinel-E can make mistakes. Consider checking important information.'}
            </p>
          </div>
        </div>
      </div>

      {/* ==================== SESSION ANALYTICS RIGHT SIDEBAR ==================== */}
      <AnimatePresence>
        {showSessionPanel && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 320, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
            className="h-full border-l border-black/5 dark:border-white/10 bg-white/80 dark:bg-[#1c1c1e]/80 backdrop-blur-xl overflow-hidden flex-shrink-0"
          >
            <div className="h-full flex flex-col">
              <div className="flex items-center justify-between px-4 py-3 border-b border-black/5 dark:border-white/5">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-[#8b5cf6]" />
                  <span className="text-[#1d1d1f] dark:text-[#f1f5f9]"
                    style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}>
                    Session Analytics
                  </span>
                </div>
                <button onClick={() => setShowSessionPanel(false)}
                  className="p-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                  <X className="w-4 h-4 text-[#6e6e73] dark:text-[#94a3b8]" />
                </button>
              </div>
              <div className="flex-1 overflow-y-auto">
                {renderSessionPanel()}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
