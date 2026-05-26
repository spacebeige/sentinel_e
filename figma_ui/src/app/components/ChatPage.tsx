import { useTheme } from "next-themes";
import { useState, useRef, useEffect, useCallback } from "react";
import {
  Send,
  Sparkles,
  ChevronDown,
  Plus,
  Paperclip,
  Mic,
  Swords,
  Gem,
  FileSearch,
  X,
  WifiOff,
  ThumbsUp,
  ThumbsDown,
  ChevronRight,
  Brain,
  BarChart3,
  Share2,
  Moon,
  Sun,
  Skull,
  Loader2,
  MessageSquare,
  PanelRightOpen,
  Search,
  Copy,
  Check,
  PenSquare,
  Settings,
  ChevronLeft,
  AlertCircle,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import {
  checkHealth,
  runStandard,
  runExperimental,
  submitFeedback,
  getChatHistory,
  getChatMessages,
  getKernelStatus,
  shareChat,
  runOmegaKill,
  type SentinelRunResponse,
  type HealthStatus,
  type ChatHistoryItem,
  type OmegaMetadata,
  type OmegaBoundaryResult,
  type OmegaReasoningTrace,
  type ConfidenceEvolution,
  type KernelStatus,
} from "../api";
import { OmegaInsightPanel } from "./OmegaInsightPanel";
import { useChatInteraction } from "../context/ChatInteractionContext";
import { SessionAnalyticsPanel } from "./SessionAnalyticsPanel";
import { CrossAnalysisTrigger } from "./CrossAnalysisPanel";
import { useSessionPersistence } from "../hooks/useSessionPersistence";
import {
  type DebateState,
  createDebateState,
  mergeDebateResult,
} from "../services/debateManager";
import {
  type GlassState,
  createGlassState,
  mergeGlassState,
  toggleKillOverride,
} from "../services/glassManager";
import {
  type EvidenceState,
  createEvidenceState,
  mergeEvidenceState,
} from "../services/evidenceManager";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  mode?: string | null;
  chatId?: string;
  confidence?: number;
  boundaryResult?: OmegaBoundaryResult;
  reasoningTrace?: OmegaReasoningTrace;
  confidenceEvolution?: ConfidenceEvolution;
  omegaMetadata?: OmegaMetadata;
  feedbackGiven?: "up" | "down";
}

// ── Pro model selector options ─────────────────────────────────────────────
const PRO_MODELS = [
  { id: "sentinel-sigma", name: "Sentinel Σ", tag: "Sigma", color: "#8b5cf6", sub: "Full orchestration" },
  { id: "gpt4", name: "GPT-4", tag: "OpenAI", color: "#10b981", sub: "Advanced reasoning" },
  { id: "claude", name: "Claude", tag: "Anthropic", color: "#f59e0b", sub: "Constitutional AI" },
  { id: "gemini", name: "Gemini", tag: "Google", color: "#3b82f6", sub: "Multimodal" },
  { id: "deepseek", name: "DeepSeek", tag: "DeepSeek", color: "#06b6d4", sub: "Research-grade" },
  { id: "mistral", name: "Mistral", tag: "Mistral AI", color: "#ef4444", sub: "Fast & efficient" },
  { id: "llama", name: "Llama 3.1", tag: "Meta", color: "#f97316", sub: "Open source" },
];

// ── Pro orchestration sub-modes ──────────────────────────────────────────────
const proSubModes = [
  { id: "debate", label: "Debate", icon: <Swords className="w-3.5 h-3.5" />, color: "#ef4444", description: "Argues both sides so you can decide", placeholder: "Give me a topic to debate..." },
  { id: "glass", label: "Glass", icon: <Gem className="w-3.5 h-3.5" />, color: "#8b5cf6", description: "Full reasoning chain — nothing hidden", placeholder: "Ask and I'll show my thinking..." },
  { id: "evidence", label: "Evidence", icon: <FileSearch className="w-3.5 h-3.5" />, color: "#06b6d4", description: "Every claim backed by a cited source", placeholder: "What do you need evidence for..." },
];

// ── Fallback responses ───────────────────────────────────────────────────────
const modeResponses: Record<string, string[]> = {
  debate: [
    "⚔️ **Debate Mode Active**\n\n**FOR:**\nThis approach has significant merit. Studies consistently show improved outcomes when applied correctly. The efficiency gains alone justify adoption — teams report 40% faster iteration cycles.\n\n**AGAINST:**\nHowever, the counterarguments are worth weighing. The upfront learning curve is steep, and not every team has the bandwidth. There's also the vendor lock-in risk.\n\n**VERDICT:** The answer depends on your team's size, timeline, and risk tolerance.",
  ],
  glass: [
    "🔍 **Glass Mode — Full Reasoning Chain**\n\n**Step 1 — Parsing your question:**\nIdentifying the core intent. You're asking about a topic that touches multiple domains.\n\n**Step 2 — Retrieving relevant knowledge:**\nPulling from training data. Moderate-to-high confidence here, flagging any gaps.\n\n**Step 3 — Forming a response:**\nHere's what I'd recommend, and here's *why* I chose it over the alternatives.\n\n**Confidence level:** ~85%.",
  ],
  evidence: [
    "📋 **Evidence Mode — Sources Cited**\n\nBased on available research:\n\n1. The primary mechanism works through attention layers that weigh token relationships ¹\n2. Performance scales roughly as a power law with compute and data ²\n3. Recent benchmarks show significant improvements in reasoning tasks ³\n\n---\n**Sources:**\n¹ Vaswani et al., \"Attention Is All You Need\" (2017)\n² Kaplan et al., Scaling Laws (2020)\n³ Multiple benchmark results, MMLU (2024)",
  ],
};

const sampleResponses = [
  "That's a great question! Let me break it down for you. The key concept here involves understanding how large language models process and generate text through a mechanism called attention. Each token in the input is compared against every other token to determine relevance, creating a rich contextual understanding.",
  "I'd be happy to help with that! Here's a comprehensive approach:\n\n1. **Start with the fundamentals** - Understanding the core architecture\n2. **Practice with examples** - Hands-on experimentation\n3. **Iterate and refine** - Continuous improvement\n\nWould you like me to dive deeper into any of these areas?",
  "Based on my analysis, there are several interesting perspectives to consider. The field has evolved rapidly, with new breakthroughs emerging almost weekly. The most significant recent development has been the improvement in reasoning capabilities.",
];

// ── Main ChatPage ────────────────────────────────────────────────────────────
export function ChatPage() {
  // State
  const [messages, setMessages] = useState<Message[]>([{
    id: "welcome",
    role: "assistant",
    content: "Hello! I'm Sentinel-E, your AI assistant powered by the Omega Cognitive Kernel. How can I help you today?",
    timestamp: new Date(),
  }]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [healthData, setHealthData] = useState<HealthStatus | null>(null);
  const [kernelData, setKernelData] = useState<KernelStatus | null>(null);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  // Sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Chat interaction context
  const { isHistoryOpen, toggleHistory, activeSubMode, setActiveSubMode, newChatTriggered, isProMode, setIsProMode } = useChatInteraction();

  // Pro features state
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [selectedModel, setSelectedModel] = useState(PRO_MODELS[0]);
  const [expandedMeta, setExpandedMeta] = useState<string | null>(null);
  const [showSessionPanel, setShowSessionPanel] = useState(false);
  const [hoveredMessage, setHoveredMessage] = useState<string | null>(null);
  const [copiedMessage, setCopiedMessage] = useState<string | null>(null);

  // Share state
  const [shareSuccess, setShareSuccess] = useState(false);

  // File upload
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Mode dropdown
  const [modeDropdownOpen, setModeDropdownOpen] = useState(false);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const healthCheckRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Session persistence
  const { restore, persist, reset: resetSession } = useSessionPersistence();

  // Mode state managers
  const [debateState, setDebateState] = useState<DebateState>(createDebateState(6));
  const [glassState, setGlassState] = useState<GlassState>(createGlassState());
  const [evidenceState, setEvidenceState] = useState<EvidenceState>(createEvidenceState());

  // ── Dark mode sync ─────────────────────────────────────────────────────────
  // ── Helpers ────────────────────────────────────────────────────────────────
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => { scrollToBottom(); }, [messages]);

  const textPrimary = isDark ? "#f5f5f7" : "#1d1d1f";
  const textSecondary = isDark ? "rgba(255,255,255,0.45)" : "rgba(0,0,0,0.45)";
  const borderColor = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.07)";
  const surfaceBg = isDark ? "#08090e" : "#f5f5f7";
  const sidebarBg = isDark ? "#0d0f18" : "#fafafa";
  const chatBg = isDark ? "#08090e" : "#ffffff";
  const inputBg = isDark ? "rgba(255,255,255,0.04)" : "rgba(255,255,255,0.95)";

  // ── Health check ───────────────────────────────────────────────────────────
  const performHealthCheck = useCallback(async () => {
    const health = await checkHealth();
    if (health) {
      setBackendOnline(true);
      setHealthData(health);
      const kernel = await getKernelStatus();
      setKernelData(kernel);
    } else {
      setBackendOnline(false);
      setHealthData(null);
      setKernelData(null);
    }
  }, []);

  useEffect(() => {
    performHealthCheck();
    healthCheckRef.current = setInterval(performHealthCheck, 15000);
    return () => { if (healthCheckRef.current) clearInterval(healthCheckRef.current); };
  }, [performHealthCheck]);

  // ── Session restore ────────────────────────────────────────────────────────
  useEffect(() => {
    const saved = restore();
    if (saved.chatId) {
      setCurrentChatId(saved.chatId);
      if (saved.subMode) setActiveSubMode(saved.subMode);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (errorMessage) {
      const t = setTimeout(() => setErrorMessage(null), 5000);
      return () => clearTimeout(t);
    }
  }, [errorMessage]);

  useEffect(() => {
    persist({
      chatId: currentChatId,
      mode: isProMode ? "experimental" : "standard",
      subMode: activeSubMode,
      isProMode,
      killOverride: glassState.killOverride,
    });
  }, [currentChatId, isProMode, activeSubMode, glassState.killOverride, persist]);

  // ── Load chat history ──────────────────────────────────────────────────────
  const loadChatHistory = useCallback(async () => {
    if (!backendOnline) return;
    setHistoryLoading(true);
    try {
      const history = await getChatHistory(50, 0);
      setChatHistory(history);
    } catch (err) {
      console.error("Failed to load chat history:", err);
    } finally {
      setHistoryLoading(false);
    }
  }, [backendOnline]);

  useEffect(() => {
    if (sidebarOpen && backendOnline) {
      loadChatHistory();
    }
  }, [sidebarOpen, backendOnline, loadChatHistory]);

  // ── Restore chat ───────────────────────────────────────────────────────────
  const restoreChat = async (chatItem: ChatHistoryItem) => {
    if (!backendOnline) return;
    try {
      const msgs = await getChatMessages(chatItem.id);
      const restored: Message[] = msgs.map((m, i) => ({
        id: `restored-${i}`,
        role: m.role,
        content: m.content,
        timestamp: m.timestamp ? new Date(m.timestamp) : new Date(),
      }));
      setMessages(restored.length > 0 ? restored : [{
        id: "welcome",
        role: "assistant" as const,
        content: "Chat restored but no messages found.",
        timestamp: new Date(),
      }]);
      setCurrentChatId(chatItem.id);
    } catch (err) {
      console.error("Failed to restore chat:", err);
      setErrorMessage("Failed to load chat messages");
    }
  };

  // ── Copy to clipboard ──────────────────────────────────────────────────────
  const copyMessage = async (messageId: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessage(messageId);
      setTimeout(() => setCopiedMessage(null), 2000);
    } catch {
      // Fallback
      const el = document.createElement("textarea");
      el.value = content;
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
      setCopiedMessage(messageId);
      setTimeout(() => setCopiedMessage(null), 2000);
    }
  };

  // ── Share chat ─────────────────────────────────────────────────────────────
  const handleShareChat = async () => {
    if (!currentChatId || !backendOnline) {
      // Copy current conversation as text
      const text = messages.map(m => `${m.role === "user" ? "You" : "Sentinel-E"}: ${m.content}`).join("\n\n");
      await navigator.clipboard.writeText(text);
      setShareSuccess(true);
      setTimeout(() => setShareSuccess(false), 2500);
      return;
    }
    try {
      const result = await shareChat(currentChatId);
      await navigator.clipboard.writeText(result.share_token);
      setShareSuccess(true);
      setTimeout(() => setShareSuccess(false), 2500);
    } catch {
      setErrorMessage("Failed to share chat");
    }
  };

  // ── File upload ─────────────────────────────────────────────────────────────
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setAttachedFile(file);
  };

  const removeFile = () => {
    setAttachedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // ── Kill switch ────────────────────────────────────────────────────────────
  const handleKillSwitch = async () => {
    if (!currentChatId || !backendOnline) return;
    setGlassState((prev) => toggleKillOverride(prev));
    setIsTyping(true);
    try {
      const response = await runOmegaKill(currentChatId, "", abortRef.current?.signal);
      const msg: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: response.formatted_output || "Kill diagnostic complete.",
        timestamp: new Date(),
        mode: "kill",
        chatId: response.chat_id,
        confidence: response.confidence,
        omegaMetadata: response.omega_metadata,
        reasoningTrace: response.reasoning_trace,
        boundaryResult: response.boundary_result,
      };
      setMessages((prev) => [...prev, msg]);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Kill switch failed");
    } finally {
      setIsTyping(false);
    }
  };

  // ── Send message ───────────────────────────────────────────────────────────
  const handleSend = async () => {
    if (!input.trim() && !attachedFile) return;

    const userText = input.trim().replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, "").slice(0, 10000);
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: userText + (attachedFile ? `\n\n[Attached: ${attachedFile.name}]` : ""),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);
    setErrorMessage(null);

    if (abortRef.current) abortRef.current.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    if (backendOnline) {
      try {
        let response: SentinelRunResponse;

        if (isProMode && activeSubMode) {
          response = await runExperimental(userText, activeSubMode, 6, currentChatId || undefined, glassState.killOverride, attachedFile || undefined, ac.signal);
        } else {
          response = await runStandard(userText, currentChatId || undefined, attachedFile || undefined, ac.signal);
        }

        if (ac.signal.aborted) return;

        if (response.chat_id) setCurrentChatId(response.chat_id);

        if (activeSubMode === "debate" && response.omega_metadata) setDebateState((p) => mergeDebateResult(p, response.omega_metadata));
        if (activeSubMode === "glass" && response.omega_metadata) setGlassState((p) => mergeGlassState(p, response.omega_metadata));
        if (activeSubMode === "evidence" && response.omega_metadata) setEvidenceState((p) => mergeEvidenceState(p, response.omega_metadata));

        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: response.formatted_output || response.data?.priority_answer || "No response generated.",
          timestamp: new Date(),
          mode: response.sub_mode || activeSubMode,
          chatId: response.chat_id,
          confidence: response.confidence,
          boundaryResult: response.boundary_result,
          reasoningTrace: response.reasoning_trace,
          confidenceEvolution: response.omega_metadata?.confidence_evolution,
          omegaMetadata: response.omega_metadata,
        };

        setMessages((prev) => [...prev, assistantMessage]);
        setIsTyping(false);
        removeFile();
      } catch (err) {
        console.error("Backend request failed:", err);
        setErrorMessage(err instanceof Error ? err.message : "Request failed");
        generateFallbackResponse();
        removeFile();
      }
    } else {
      generateFallbackResponse();
      removeFile();
    }
  };

  const generateFallbackResponse = () => {
    setTimeout(() => {
      const currentMode = activeSubMode;
      let response: string;
      if (currentMode && modeResponses[currentMode]) {
        const responses = modeResponses[currentMode];
        response = responses[Math.floor(Math.random() * responses.length)];
      } else {
        response = sampleResponses[Math.floor(Math.random() * sampleResponses.length)];
      }
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response,
        timestamp: new Date(),
        mode: activeSubMode,
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setIsTyping(false);
    }, 1200);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewChat = () => {
    if (abortRef.current) abortRef.current.abort();
    setCurrentChatId(null);
    setAttachedFile(null);
    setExpandedMeta(null);
    setShowSessionPanel(false);
    setDebateState(createDebateState(6));
    setGlassState(createGlassState());
    setEvidenceState(createEvidenceState());
    resetSession();
    setMessages([{
      id: "welcome",
      role: "assistant",
      content: "Hello! I'm Sentinel-E, your AI assistant powered by the Omega Cognitive Kernel. How can I help you today?",
      timestamp: new Date(),
    }]);
  };

  const handleFeedback = async (messageId: string, vote: "up" | "down") => {
    const msg = messages.find((m) => m.id === messageId);
    if (!msg?.chatId) return;
    setMessages((prev) => prev.map((m) => (m.id === messageId ? { ...m, feedbackGiven: vote } : m)));
    if (!backendOnline) return;
    try {
      await submitFeedback({
        run_id: msg.chatId,
        feedback: vote,
        mode: isProMode ? "experimental" : "standard",
        sub_mode: msg.mode || undefined,
        confidence: msg.confidence,
        boundary_severity: msg.boundaryResult?.severity_score,
        fragility_index: msg.omegaMetadata?.fragility_index,
        disagreement_score: msg.omegaMetadata?.session_state?.disagreement_score,
      });
    } catch { /* silent fail */ }
  };

  // ── Confidence bar ─────────────────────────────────────────────────────────
  const renderConfidenceBar = (value: number, label: string, color: string) => (
    <div className="flex items-center gap-2">
      <span className="w-20 flex-shrink-0" style={{ fontSize: "10px", fontWeight: 500, color: textSecondary }}>{label}</span>
      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)" }}>
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.round(value * 100)}%`, backgroundColor: color }} />
      </div>
      <span className="w-10 text-right flex-shrink-0" style={{ fontSize: "10px", fontWeight: 600, color: textPrimary }}>{Math.round(value * 100)}%</span>
    </div>
  );

  // ── Omega insights panel ───────────────────────────────────────────────────
  const renderOmegaInsights = (message: Message) => {
    const isExpanded = expandedMeta === message.id;
    const hasData = message.omegaMetadata || message.reasoningTrace || message.confidenceEvolution || message.boundaryResult;
    if (!hasData || message.role !== "assistant") return null;

    return (
      <div className="mt-2">
        <button
          onClick={() => setExpandedMeta(isExpanded ? null : message.id)}
          className="flex items-center gap-1 px-2 py-1 rounded-lg transition-colors"
          style={{ background: isExpanded ? (isDark ? "rgba(139,92,246,0.1)" : "rgba(139,92,246,0.06)") : "transparent" }}
        >
          <Brain className="w-3 h-3 text-[#8b5cf6]" />
          <span style={{ fontSize: "10px", fontWeight: 600, color: "#8b5cf6" }}>Omega Insights</span>
          <ChevronRight className="w-3 h-3 text-[#8b5cf6] transition-transform" style={{ transform: isExpanded ? "rotate(90deg)" : "rotate(0deg)" }} />
        </button>

        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div
                className="mt-2 p-3 rounded-xl space-y-3"
                style={{ background: isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)" }}
              >
                {message.confidenceEvolution && (
                  <div>
                    <div className="flex items-center gap-1 mb-2">
                      <BarChart3 className="w-3 h-3 text-[#3b82f6]" />
                      <span style={{ fontSize: "10px", fontWeight: 600, color: "#3b82f6", textTransform: "uppercase", letterSpacing: "0.05em" }}>Confidence Evolution</span>
                    </div>
                    <div className="space-y-1.5">
                      {renderConfidenceBar(message.confidenceEvolution.initial, "Initial", "#aeaeb2")}
                      {message.confidenceEvolution.post_debate != null && renderConfidenceBar(message.confidenceEvolution.post_debate, "Post-debate", "#ef4444")}
                      {message.confidenceEvolution.post_boundary != null && renderConfidenceBar(message.confidenceEvolution.post_boundary, "Post-bound.", "#f59e0b")}
                      {message.confidenceEvolution.post_evidence != null && renderConfidenceBar(message.confidenceEvolution.post_evidence, "Post-evid.", "#06b6d4")}
                      {message.confidenceEvolution.post_stress != null && renderConfidenceBar(message.confidenceEvolution.post_stress, "Post-stress", "#8b5cf6")}
                      {renderConfidenceBar(message.confidenceEvolution.final, "Final", "#10b981")}
                    </div>
                  </div>
                )}
                {message.omegaMetadata?.omega_version && (
                  <div className="flex items-center justify-between pt-1" style={{ borderTop: `1px solid ${borderColor}` }}>
                    <span style={{ fontSize: "9px", color: textSecondary }}>Omega Kernel v{message.omegaMetadata.omega_version}</span>
                    {message.omegaMetadata.session_state?.inferred_domain && (
                      <span style={{ fontSize: "9px", color: textSecondary }}>Domain: {message.omegaMetadata.session_state.inferred_domain}</span>
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

  // ── Filtered history ──────────────────────────────────────────────────────
  const filteredHistory = chatHistory.filter(c =>
    !searchQuery || (c.name || "").toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Group history by time
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  const groupedHistory = {
    today: filteredHistory.filter(c => new Date(c.updated_at || c.created_at).toDateString() === today.toDateString()),
    yesterday: filteredHistory.filter(c => new Date(c.updated_at || c.created_at).toDateString() === yesterday.toDateString()),
    older: filteredHistory.filter(c => {
      const d = new Date(c.updated_at || c.created_at);
      return d.toDateString() !== today.toDateString() && d.toDateString() !== yesterday.toDateString();
    }),
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div
      className="relative flex h-screen w-full overflow-hidden"
      style={{ background: chatBg }}
    >

      {/* ── LEFT SIDEBAR ──────────────────────────────────────────────────── */}
      {/* Mobile overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSidebarOpen(false)}
            className="fixed inset-0 z-30 md:hidden"
            style={{ background: "rgba(0,0,0,0.4)", backdropFilter: "blur(4px)" }}
          />
        )}
      </AnimatePresence>

      {/* Sidebar panel */}
      
          <aside
            className={`absolute md:relative left-0 top-0 h-full z-40 flex flex-col flex-shrink-0 overflow-hidden transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] ${
              sidebarOpen ? "translate-x-0 w-[260px]" : "-translate-x-full md:translate-x-0 md:w-[68px]"
            }`}
            style={{
              background: sidebarBg,
              borderRight: `1px solid ${borderColor}`,
            }}
          >
            {/* Sidebar header */}
            <div
              className="flex items-center justify-between px-4 py-3.5"
              style={{ borderBottom: `1px solid ${borderColor}` }}
            >
              {sidebarOpen ? (
                <span
                  style={{
                    fontFamily: "'Inter', sans-serif",
                    fontSize: "13px",
                    fontWeight: 600,
                    color: textPrimary,
                    letterSpacing: "-0.01em",
                  }}
                >
                  Sentinel-E
                </span>
              ) : (
                <div className="w-6 h-6 flex-shrink-0 rounded-lg bg-gradient-to-br from-[#3b82f6] to-[#06b6d4] flex items-center justify-center">
                  <span className="text-white text-[10px] font-bold">S</span>
                </div>
              )}
              <div className="flex items-center gap-1">
                <button
                  onClick={handleNewChat}
                  className="p-1.5 rounded-lg transition-colors"
                  title="New Chat"
                  style={{ color: textSecondary }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                >
                  <PenSquare className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setSidebarOpen(false)}
                  className="p-1.5 rounded-lg transition-colors lg:hidden"
                  style={{ color: textSecondary }}
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Search */}
            <div className="px-3 py-2.5">
              <button
                onClick={() => !sidebarOpen && setSidebarOpen(true)}
                className={`flex items-center gap-2 rounded-xl transition-all ${sidebarOpen ? 'px-3 py-2' : 'p-2.5 justify-center'}`}
                style={{ background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", width: "100%" }}
              >
                <Search className="w-3.5 h-3.5 flex-shrink-0" style={{ color: textSecondary }} />
                {sidebarOpen && (
                  <>
                    <input
                      type="text"
                      placeholder="Search chats..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="flex-1 bg-transparent outline-none"
                      style={{
                        fontFamily: "'Inter', sans-serif",
                        fontSize: "13px",
                        fontWeight: 400,
                        color: textPrimary,
                      }}
                    />
                    {searchQuery && (
                      <div onClick={(e) => { e.stopPropagation(); setSearchQuery(""); }} className="cursor-pointer">
                        <X className="w-3 h-3" style={{ color: textSecondary }} />
                      </div>
                    )}
                  </>
                )}
              </button>
            </div>

            {/* New Chat button */}
            <div className="px-3 pb-2">
              <button
                onClick={handleNewChat}
                className={`w-full flex items-center transition-all ${sidebarOpen ? 'gap-2.5 px-3 py-2.5 rounded-xl' : 'justify-center p-2.5 rounded-xl'}`}
                style={{
                  background: isDark ? "rgba(59,130,246,0.1)" : "rgba(59,130,246,0.06)",
                  border: "1px solid rgba(59,130,246,0.2)",
                  color: "#3b82f6",
                }}
              >
                <Plus className="w-4 h-4 flex-shrink-0" />
                {sidebarOpen && <span style={{ fontFamily: "'Inter', sans-serif", fontSize: "13px", fontWeight: 600 }}>New Chat</span>}
              </button>
            </div>

            {/* Chat history */}
            <div className={`flex-1 overflow-y-auto py-1 ${sidebarOpen ? 'px-2' : 'px-0 opacity-0 pointer-events-none'}`}>
              {!backendOnline ? (
                <div className="px-3 py-8 text-center">
                  <WifiOff className="w-6 h-6 mx-auto mb-2" style={{ color: textSecondary }} />
                  <p style={{ fontSize: "12px", color: textSecondary }}>Connect backend to see history</p>
                </div>
              ) : historyLoading ? (
                <div className="px-3 py-8 text-center">
                  <Loader2 className="w-5 h-5 mx-auto mb-2 animate-spin" style={{ color: textSecondary }} />
                  <p style={{ fontSize: "12px", color: textSecondary }}>Loading...</p>
                </div>
              ) : filteredHistory.length === 0 ? (
                <div className="px-3 py-8 text-center">
                  <MessageSquare className="w-6 h-6 mx-auto mb-2" style={{ color: textSecondary }} />
                  <p style={{ fontSize: "12px", color: textSecondary }}>No chats yet</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {Object.entries({ Today: groupedHistory.today, Yesterday: groupedHistory.yesterday, "Earlier": groupedHistory.older }).map(([group, chats]) => (
                    chats.length > 0 && (
                      <div key={group}>
                        <div
                          className="px-3 py-1"
                          style={{ fontSize: "10px", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.25)" }}
                        >
                          {group}
                        </div>
                        <div className="space-y-0.5">
                          {chats.map((chat) => (
                            <button
                              key={chat.id}
                              onClick={() => restoreChat(chat)}
                              className="w-full text-left px-3 py-2.5 rounded-xl transition-all"
                              style={{
                                background: currentChatId === chat.id
                                  ? (isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)")
                                  : "transparent",
                              }}
                              onMouseEnter={(e) => {
                                if (currentChatId !== chat.id) e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)";
                              }}
                              onMouseLeave={(e) => {
                                if (currentChatId !== chat.id) e.currentTarget.style.background = "transparent";
                              }}
                            >
                              <div className="truncate" style={{ fontSize: "13px", fontWeight: 500, color: textPrimary }}>
                                {chat.name || "Untitled Chat"}
                              </div>
                              <div
                                className="mt-0.5"
                                style={{ fontSize: "10px", color: textSecondary }}
                              >
                                {new Date(chat.updated_at || chat.created_at).toLocaleDateString()}
                              </div>
                            </button>
                          ))}
                        </div>
                      </div>
                    )
                  ))}
                </div>
              )}
            </div>

            {/* Sidebar footer */}
            <div
              className="px-2 py-3 space-y-0.5"
              style={{ borderTop: `1px solid ${borderColor}` }}
            >
              <button
                className={`w-full flex items-center transition-colors ${sidebarOpen ? 'gap-2.5 px-3 py-2.5 rounded-xl' : 'justify-center p-2.5 rounded-xl'}`}
                style={{ color: textSecondary }}
                onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
              >
                <Share2 className="w-4 h-4 flex-shrink-0" />
                {sidebarOpen && <span style={{ fontSize: "13px", fontWeight: 500 }}>Share Chat</span>}
              </button>
              
              <button
                className={`w-full flex items-center transition-colors ${sidebarOpen ? 'gap-2.5 px-3 py-2.5 rounded-xl' : 'justify-center p-2.5 rounded-xl'}`}
                style={{ color: textSecondary }}
                onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
              >
                <Copy className="w-4 h-4 flex-shrink-0" />
                {sidebarOpen && <span style={{ fontSize: "13px", fontWeight: 500 }}>Copy Chat</span>}
              </button>

              <button
                className={`w-full flex items-center transition-colors ${sidebarOpen ? 'gap-2.5 px-3 py-2.5 rounded-xl' : 'justify-center p-2.5 rounded-xl'}`}
                style={{ color: textSecondary }}
                onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
              >
                <Settings className="w-4 h-4 flex-shrink-0" />
                {sidebarOpen && <span style={{ fontSize: "13px", fontWeight: 500 }}>Settings</span>}
              </button>
            </div>
          </aside>

      {/* ── MAIN CHAT AREA ──────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 h-full relative" style={{ background: chatBg }}>

        {/* ── TOP BAR ─────────────────────────────────────────────────── */}
        <div
          className="flex items-center justify-between px-4 py-3 sticky top-0 z-20"
          style={{
            background: isDark ? "rgba(8,9,14,0.85)" : "rgba(255,255,255,0.85)",
            backdropFilter: "blur(20px)",
            WebkitBackdropFilter: "blur(20px)",
            borderBottom: `1px solid ${borderColor}`,
          }}
        >
          {/* Left — Sidebar toggle + Mode dropdown */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-xl transition-colors"
              style={{ color: textSecondary }}
              onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
            >
              {sidebarOpen ? <ChevronLeft className="w-4.5 h-4.5" /> : <MessageSquare className="w-4.5 h-4.5" />}
            </button>

            {/* Mode dropdown */}
            <div className="relative">
              <button
                onClick={() => setModeDropdownOpen(!modeDropdownOpen)}
                className="group flex items-center gap-2 px-3 py-1.5 rounded-xl transition-all duration-300 hover:shadow-sm"
                style={{
                  background: isDark ? "rgba(255,255,255,0.04)" : "transparent",
                  border: isDark ? `1px solid rgba(255,255,255,0.08)` : `1px solid transparent`,
                  color: textPrimary,
                }}
              >
                <span className="font-semibold text-[15px] flex items-center gap-2" style={{ fontFamily: "'Inter', sans-serif" }}>
                  Sentinel {isProMode ? <span className="text-[#3b82f6]">Pro</span> : <span className="text-gray-500">Standard</span>}
                </span>
                <ChevronDown className={`w-4 h-4 transition-transform duration-300 ${modeDropdownOpen ? "rotate-180" : ""}`} style={{ color: textSecondary }} />
              </button>

              <AnimatePresence>
                {modeDropdownOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -6, scale: 0.97 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -6, scale: 0.97 }}
                    transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
                    className="absolute left-0 top-full mt-1.5 w-52 rounded-2xl overflow-hidden z-50"
                    style={{
                      background: isDark ? "#0f1117" : "#ffffff",
                      border: `1px solid ${borderColor}`,
                      boxShadow: isDark ? "0 16px 40px rgba(0,0,0,0.5)" : "0 16px 40px rgba(0,0,0,0.1)",
                    }}
                  >
                    <div className="p-1.5">
                      {[
                        { id: "standard", label: "Standard", sub: "Simple AI experience", pro: false },
                        { id: "pro", label: "Pro", sub: "Full orchestration · multi-model", pro: true },
                      ].map((m) => (
                        <button
                          key={m.id}
                          onClick={() => { setIsProMode(m.pro); setModeDropdownOpen(false); if (!m.pro) setActiveSubMode(null); }}
                          className="w-full flex items-start gap-2.5 px-3 py-2.5 rounded-xl transition-colors text-left"
                          style={{
                            background: isProMode === m.pro ? (isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)") : "transparent",
                          }}
                          onMouseEnter={(e) => { if (isProMode !== m.pro) e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)"; }}
                          onMouseLeave={(e) => { if (isProMode !== m.pro) e.currentTarget.style.background = "transparent"; }}
                        >
                          <div className="mt-0.5 w-4 h-4 rounded-full flex-shrink-0 flex items-center justify-center" style={{ background: m.pro ? "rgba(139,92,246,0.15)" : "rgba(59,130,246,0.15)" }}>
                            <div className="w-1.5 h-1.5 rounded-full" style={{ background: m.pro ? "#8b5cf6" : "#3b82f6" }} />
                          </div>
                          <div>
                            <div style={{ fontSize: "13px", fontWeight: 600, color: textPrimary }}>{m.label}</div>
                            <div style={{ fontSize: "11px", color: textSecondary, marginTop: "1px" }}>{m.sub}</div>
                          </div>
                          {isProMode === m.pro && <Check className="w-3.5 h-3.5 ml-auto mt-1 text-[#3b82f6]" />}
                        </button>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Right — Actions */}
          <div className="flex items-center gap-2">
            <button
              onClick={toggleTheme}
              className="p-2 rounded-xl transition-all duration-300"
              style={{ color: textSecondary }}
              onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; e.currentTarget.style.transform = "translateY(-1px)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.transform = "translateY(0)"; }}
              title="Toggle Theme"
            >
              {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
            
            {/* Connection status — subtle, not a banner */}
            <div className="hidden sm:flex items-center gap-1.5 mr-2">
              {backendOnline === true ? (
                <>
                  <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                  <span style={{ fontSize: "11px", color: "#10b981", fontWeight: 500 }}>Live</span>
                </>
              ) : backendOnline === false ? (
                <>
                  <div className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                  <span style={{ fontSize: "11px", color: "#f59e0b", fontWeight: 500 }}>Offline</span>
                </>
              ) : null}
            </div>

            {/* Share */}
            <button
              onClick={handleShareChat}
              className="p-2 rounded-xl transition-all"
              title="Share Chat"
              style={{ color: shareSuccess ? "#10b981" : textSecondary }}
              onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
            >
              {shareSuccess ? <Check className="w-4 h-4" /> : <Share2 className="w-4 h-4" />}
            </button>

            {/* Session analytics */}
            {currentChatId && backendOnline && (
              <button
                onClick={() => setShowSessionPanel(!showSessionPanel)}
                className="p-2 rounded-xl transition-all"
                title="Session Analytics"
                style={{ color: showSessionPanel ? "#8b5cf6" : textSecondary }}
                onMouseEnter={(e) => { if (!showSessionPanel) e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; }}
                onMouseLeave={(e) => { if (!showSessionPanel) e.currentTarget.style.background = "transparent"; }}
              >
                <PanelRightOpen className="w-4 h-4" />
              </button>
            )}

            {/* Kill switch */}
            {isProMode && currentChatId && backendOnline && (
              <button
                onClick={handleKillSwitch}
                className="p-2 rounded-xl transition-colors"
                title="Kill Diagnostic"
                style={{ color: textSecondary }}
                onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(239,68,68,0.08)"; e.currentTarget.style.color = "#ef4444"; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = textSecondary; }}
              >
                <Skull className="w-4 h-4" />
              </button>
            )}

            {/* New Chat */}
            <button
              onClick={handleNewChat}
              className="p-2 rounded-xl transition-colors"
              title="New Chat"
              style={{ color: textSecondary }}
              onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
            >
              <Plus className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* ── ERROR BANNER ────────────────────────────────────────────────── */}
        <AnimatePresence>
          {errorMessage && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="flex items-center justify-between px-4 py-2"
              style={{ background: isDark ? "rgba(239,68,68,0.08)" : "#fef2f2", borderBottom: `1px solid rgba(239,68,68,0.15)` }}
            >
              <div className="flex items-center gap-2">
                <AlertCircle className="w-3.5 h-3.5 text-[#ef4444] flex-shrink-0" />
                <span style={{ fontSize: "12px", fontWeight: 500, color: isDark ? "#fca5a5" : "#991b1b" }}>{errorMessage}</span>
              </div>
              <button onClick={() => setErrorMessage(null)}>
                <X className="w-3.5 h-3.5" style={{ color: isDark ? "#fca5a5" : "#991b1b" }} />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── MESSAGES ─────────────────────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto px-4 py-8" onClick={() => { setModeDropdownOpen(false); setShowModelSelector(false); }}>
          <div className="max-w-2xl mx-auto space-y-5">
            <AnimatePresence>
              {messages.map((message) => {
                const msgMode = message.mode ? proSubModes.find(m => m.id === message.mode) : null;
                const isHovered = hoveredMessage === message.id;

                return (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                    onMouseEnter={() => setHoveredMessage(message.id)}
                    onMouseLeave={() => setHoveredMessage(null)}
                  >
                    <div className="relative max-w-[85%] sm:max-w-[72%]">
                      {/* Message bubble */}
                      <div
                        className="relative overflow-hidden"
                        style={message.role === "user" ? {
                          borderRadius: "20px 20px 5px 20px",
                          background: isDark ? "rgb(32, 32, 36)" : "rgb(240, 240, 245)",
                          color: isDark ? "#f5f5f7" : "#1d1d1f",
                          padding: "12px 16px",
                          boxShadow: isDark ? "0 4px 12px rgba(0,0,0,0.3)" : "0 2px 8px rgba(0,0,0,0.03)",
                          border: isDark ? "1px solid rgba(255,255,255,0.05)" : "1px solid rgba(0,0,0,0.04)",
                        } : {
                          borderRadius: "20px 20px 20px 5px",
                          background: isDark ? "rgb(24, 24, 28)" : "#ffffff",
                          color: textPrimary,
                          border: `1px solid ${msgMode ? msgMode.color + "25" : borderColor}`,
                          borderLeft: msgMode ? `3px solid ${msgMode.color}` : message.mode === "kill" ? "3px solid #ef4444" : `1px solid ${borderColor}`,
                          boxShadow: isDark ? "0 4px 16px rgba(0,0,0,0.4)" : "0 4px 16px rgba(0,0,0,0.04)",
                          overflow: "visible",
                        }}
                      >
                        {/* Mode badge for assistant */}
                        {message.role === "assistant" && msgMode && (
                          <div
                            className="flex items-center gap-1.5 px-4 py-1.5"
                            style={{ background: msgMode.color + "0c", borderBottom: `1px solid ${msgMode.color}18` }}
                          >
                            <div style={{ color: msgMode.color }}>{msgMode.icon}</div>
                            <span style={{ fontSize: "10px", fontWeight: 700, color: msgMode.color, letterSpacing: "0.08em", textTransform: "uppercase" }}>
                              {msgMode.label}
                            </span>
                            {message.confidence !== undefined && (
                              <span className="ml-auto" style={{ fontSize: "10px", fontWeight: 500, color: textSecondary }}>
                                {Math.round(message.confidence * 100)}% conf.
                              </span>
                            )}
                          </div>
                        )}

                        {/* Kill mode badge */}
                        {message.role === "assistant" && message.mode === "kill" && (
                          <div className="flex items-center gap-1.5 px-4 py-1.5" style={{ background: "rgba(239,68,68,0.06)", borderBottom: "1px solid rgba(239,68,68,0.12)" }}>
                            <Skull className="w-3.5 h-3.5 text-[#ef4444]" />
                            <span style={{ fontSize: "10px", fontWeight: 700, color: "#ef4444", letterSpacing: "0.08em", textTransform: "uppercase" }}>Kill Diagnostic</span>
                          </div>
                        )}

                        {/* Content */}
                        <div className={message.role === "assistant" ? "px-4 py-3" : ""}>
                          <p
                            className="whitespace-pre-wrap"
                            style={{
                              fontFamily: "'Inter', sans-serif",
                              fontSize: "15px",
                              lineHeight: 1.6,
                              fontWeight: 400,
                              color: message.role === "user" ? (isDark ? "#f5f5f7" : "#1d1d1f") : textPrimary,
                            }}
                          >
                            {message.content}
                          </p>

                          {/* Omega insights */}
                          {renderOmegaInsights(message)}

                          {/* Cross-analysis trigger */}
                          {message.role === "assistant" && message.mode === "glass" && message.id !== "welcome" && backendOnline && (
                            <CrossAnalysisTrigger chatId={currentChatId} messageContent={message.content} backendOnline={backendOnline} />
                          )}

                          {/* Timestamp + feedback */}
                          <div className="flex items-center justify-between mt-2">
                            <span style={{ fontSize: "10px", color: message.role === "user" ? "rgba(255,255,255,0.4)" : textSecondary }}>
                              {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                            </span>

                            {message.role === "assistant" && message.id !== "welcome" && (
                              <div className="flex items-center gap-0.5 ml-2">
                                <button
                                  onClick={() => handleFeedback(message.id, "up")}
                                  className="p-1 rounded-md transition-colors"
                                  disabled={!!message.feedbackGiven}
                                  style={{ background: message.feedbackGiven === "up" ? "rgba(16,185,129,0.15)" : "transparent" }}
                                >
                                  <ThumbsUp className="w-3 h-3" style={{ color: message.feedbackGiven === "up" ? "#10b981" : textSecondary }} />
                                </button>
                                <button
                                  onClick={() => handleFeedback(message.id, "down")}
                                  className="p-1 rounded-md transition-colors"
                                  disabled={!!message.feedbackGiven}
                                  style={{ background: message.feedbackGiven === "down" ? "rgba(239,68,68,0.1)" : "transparent" }}
                                >
                                  <ThumbsDown className="w-3 h-3" style={{ color: message.feedbackGiven === "down" ? "#ef4444" : textSecondary }} />
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Hover: Copy button */}
                      <AnimatePresence>
                        {isHovered && (
                          <motion.button
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.8 }}
                            transition={{ duration: 0.12 }}
                            onClick={() => copyMessage(message.id, message.content)}
                            className="absolute -top-3 flex items-center gap-1 px-2 py-1 rounded-full transition-all"
                            style={{
                              right: message.role === "user" ? "4px" : undefined,
                              left: message.role === "assistant" ? "4px" : undefined,
                              background: isDark ? "#1a1d26" : "#ffffff",
                              border: `1px solid ${borderColor}`,
                              boxShadow: isDark ? "0 4px 16px rgba(0,0,0,0.4)" : "0 4px 12px rgba(0,0,0,0.08)",
                              color: copiedMessage === message.id ? "#10b981" : textSecondary,
                              zIndex: 10,
                            }}
                          >
                            {copiedMessage === message.id
                              ? <><Check className="w-3 h-3" /><span style={{ fontSize: "10px", fontWeight: 600 }}>Copied</span></>
                              : <><Copy className="w-3 h-3" /><span style={{ fontSize: "10px", fontWeight: 600 }}>Copy</span></>
                            }
                          </motion.button>
                        )}
                      </AnimatePresence>
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>

            {/* Typing indicator */}
            {isTyping && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-start"
              >
                <div
                  className="px-5 py-4 rounded-[20px] rounded-bl-md"
                  style={{
                    background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.03)",
                    border: `1px solid ${activeSubMode ? proSubModes.find(m => m.id === activeSubMode)!.color + "30" : borderColor}`,
                    borderLeft: activeSubMode ? `3px solid ${proSubModes.find(m => m.id === activeSubMode)!.color}` : `1px solid ${borderColor}`,
                  }}
                >
                  {activeSubMode && (
                    <div className="flex items-center gap-1 mb-2" style={{ color: proSubModes.find(m => m.id === activeSubMode)!.color }}>
                      {proSubModes.find(m => m.id === activeSubMode)!.icon}
                      <span style={{ fontSize: "10px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                        {backendOnline ? "Processing via Omega Kernel..." : "Thinking..."}
                      </span>
                    </div>
                  )}
                  <div className="flex gap-1.5">
                    {[0, 120, 240].map((delay) => (
                      <div
                        key={delay}
                        className="w-2 h-2 rounded-full animate-bounce"
                        style={{
                          background: activeSubMode ? proSubModes.find(m => m.id === activeSubMode)!.color : (isDark ? "#6e6e73" : "#aeaeb2"),
                          animationDelay: `${delay}ms`,
                        }}
                      />
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ── FLOATING INPUT DOCK ──────────────────────────────────────────── */}
        <div
          className="px-4 pt-3 z-20"
          style={{
            paddingBottom: "calc(24px + env(safe-area-inset-bottom, 0px))",
            background: isDark
              ? "linear-gradient(to top, rgba(8,9,14,1) 60%, rgba(8,9,14,0) 100%)"
              : "linear-gradient(to top, rgba(255,255,255,1) 60%, rgba(255,255,255,0) 100%)",
          }}
        >
          <div className="max-w-2xl mx-auto">
            {/* Pro model selector popup */}
            <AnimatePresence>
              {showModelSelector && isProMode && (
                <motion.div
                  initial={{ opacity: 0, y: 10, scale: 0.97 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.97 }}
                  transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
                  className="mb-2 p-2 rounded-2xl grid grid-cols-2 sm:grid-cols-3 gap-1.5"
                  style={{
                    background: isDark ? "#0f1117" : "#ffffff",
                    border: `1px solid ${borderColor}`,
                    boxShadow: isDark ? "0 16px 40px rgba(0,0,0,0.5)" : "0 16px 40px rgba(0,0,0,0.1)",
                  }}
                >
                  {PRO_MODELS.map((model) => (
                    <button
                      key={model.id}
                      onClick={() => { setSelectedModel(model); setShowModelSelector(false); }}
                      className="flex items-start gap-2 p-2.5 rounded-xl transition-all text-left"
                      style={{
                        background: selectedModel.id === model.id
                          ? (isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)")
                          : "transparent",
                        border: selectedModel.id === model.id ? `1px solid ${model.color}30` : "1px solid transparent",
                      }}
                      onMouseEnter={(e) => { if (selectedModel.id !== model.id) e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)"; }}
                      onMouseLeave={(e) => { if (selectedModel.id !== model.id) e.currentTarget.style.background = "transparent"; }}
                    >
                      <div className="w-6 h-6 rounded-lg flex-shrink-0 flex items-center justify-center" style={{ background: model.color + "20" }}>
                        <div className="w-2.5 h-2.5 rounded-full" style={{ background: model.color }} />
                      </div>
                      <div>
                        <div style={{ fontSize: "12px", fontWeight: 600, color: textPrimary, lineHeight: 1.2 }}>{model.name}</div>
                        <div style={{ fontSize: "10px", color: textSecondary, marginTop: "1px" }}>{model.sub}</div>
                      </div>
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Pro submodes row */}
            <AnimatePresence>
              {isProMode && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.25 }}
                  className="overflow-hidden mb-2"
                >
                  <div className="flex items-center gap-1.5 pb-2">
                    {proSubModes.map((mode) => {
                      const isActive = activeSubMode === mode.id;
                      return (
                        <button
                          key={mode.id}
                          onClick={() => setActiveSubMode(isActive ? null : mode.id)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-full transition-all"
                          style={{
                            background: isActive ? mode.color : (isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)"),
                            color: isActive ? "#ffffff" : textSecondary,
                            border: `1px solid ${isActive ? mode.color : borderColor}`,
                            boxShadow: isActive ? `0 2px 12px ${mode.color}40` : "none",
                          }}
                        >
                          {mode.icon}
                          <span style={{ fontSize: "12px", fontWeight: 600 }}>{mode.label}</span>
                        </button>
                      );
                    })}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Input container */}
            <div
              className="relative rounded-[24px] transition-all duration-300"
              style={{
                background: inputBg,
                border: activeSubMode
                  ? `1px solid ${proSubModes.find(m => m.id === activeSubMode)!.color}40`
                  : `1px solid ${borderColor}`,
                boxShadow: activeSubMode
                  ? `0 4px 24px -4px ${proSubModes.find(m => m.id === activeSubMode)!.color}20, ${isDark ? "0 0 0 1px rgba(255,255,255,0.04)" : "0 0 0 1px rgba(0,0,0,0.04)"}`
                  : isDark
                    ? "0 4px 24px rgba(0,0,0,0.3), 0 0 0 1px rgba(255,255,255,0.04)"
                    : "0 4px 24px rgba(0,0,0,0.07)",
                backdropFilter: "blur(20px)",
              }}
            >
              {/* File attachment preview */}
              <AnimatePresence>
                {attachedFile && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="overflow-hidden"
                  >
                    <div
                      className="flex items-center gap-2 mx-3 mt-3 px-3 py-2 rounded-xl"
                      style={{ background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)" }}
                    >
                      <Paperclip className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "#3b82f6" }} />
                      <span className="flex-1 truncate" style={{ fontSize: "12px", fontWeight: 500, color: textPrimary }}>
                        {attachedFile.name}
                      </span>
                      <span style={{ fontSize: "10px", color: textSecondary }}>
                        {(attachedFile.size / 1024).toFixed(1)} KB
                      </span>
                      <button onClick={removeFile}>
                        <X className="w-3 h-3" style={{ color: textSecondary }} />
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Input row */}
              <div className="flex items-end gap-1 p-2">
                {/* Left actions */}
                <div className="flex items-center gap-0.5 pb-0.5">
                  {/* Plus — opens model selector in Pro mode */}
                  {isProMode && (
                    <button
                      onClick={() => setShowModelSelector(!showModelSelector)}
                      className="p-2 rounded-full transition-all"
                      style={{ color: showModelSelector ? "#3b82f6" : textSecondary }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                    >
                      <Plus className="w-5 h-5 flex-shrink-0" />
                    </button>
                  )}
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className={`p-2 rounded-full transition-all ${attachedFile ? "text-[#3b82f6]" : ""}`}
                    style={{ color: attachedFile ? "#3b82f6" : textSecondary }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                  >
                    <Paperclip className="w-4.5 h-4.5" />
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    onChange={handleFileSelect}
                    accept=".txt,.pdf,.md,.json,.csv,.py,.js,.ts,.jsx,.tsx"
                  />
                </div>

                {/* Pro model label */}
                {isProMode && (
                  <button
                    onClick={() => setShowModelSelector(!showModelSelector)}
                    className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl mb-0.5 flex-shrink-0 transition-all"
                    style={{
                      background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)",
                      border: `1px solid ${borderColor}`,
                      color: selectedModel.color,
                    }}
                  >
                    <div className="w-2 h-2 rounded-full" style={{ background: selectedModel.color }} />
                    <span style={{ fontSize: "11px", fontWeight: 600 }}>{selectedModel.name}</span>
                    <ChevronDown className="w-3 h-3" style={{ color: textSecondary }} />
                  </button>
                )}

                {/* Textarea */}
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => {
                    setInput(e.target.value);
                    e.target.style.height = "auto";
                    e.target.style.height = Math.min(e.target.scrollHeight, 128) + "px";
                  }}
                  onKeyDown={handleKeyDown}
                  placeholder={
                    activeSubMode
                      ? proSubModes.find(m => m.id === activeSubMode)!.placeholder
                      : "Message Sentinel-E..."
                  }
                  rows={1}
                  className="flex-1 resize-none bg-transparent outline-none py-2 px-1 max-h-32 sentinel-input"
                  style={{
                    fontFamily: "'Inter', sans-serif",
                    fontSize: "15px",
                    lineHeight: 1.55,
                    fontWeight: 400,
                    color: textPrimary,
                    minHeight: "36px",
                  }}
                />

                {/* Right actions */}
                <div className="flex items-center gap-0.5 pb-0.5">
                  <button
                    className="p-2 rounded-full transition-colors"
                    style={{ color: textSecondary }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                  >
                    <Mic className="w-4.5 h-4.5" />
                  </button>
                  <button
                    onClick={handleSend}
                    disabled={(!input.trim() && !attachedFile) || isTyping}
                    className="p-2 rounded-full transition-all"
                    style={{
                      background: (!input.trim() && !attachedFile) || isTyping
                        ? (isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.06)")
                        : activeSubMode
                          ? proSubModes.find(m => m.id === activeSubMode)!.color
                          : "#1d1d1f",
                      color: (!input.trim() && !attachedFile) || isTyping ? textSecondary : "#ffffff",
                      boxShadow: (input.trim() || attachedFile) && !isTyping
                        ? `0 4px 12px rgba(0,0,0,0.2)`
                        : "none",
                    }}
                  >
                    <Send className="w-4.5 h-4.5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Disclaimer */}
            <p
              className="text-center mt-2"
              style={{
                fontFamily: "'Inter', sans-serif",
                fontSize: "11px",
                fontWeight: 400,
                color: isDark ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.25)",
              }}
            >
              {backendOnline
                ? `Sentinel-E Omega v${healthData?.version || "4.5"}${currentChatId ? ` · ${currentChatId.slice(0, 8)}` : ""}`
                : "Sentinel-E can make mistakes. Check important information."}
            </p>
          </div>
        </div>
      </div>

      {/* ── SESSION ANALYTICS PANEL ──────────────────────────────────────── */}
      <AnimatePresence>
        {showSessionPanel && (
          <SessionAnalyticsPanel
            chatId={currentChatId}
            backendOnline={backendOnline}
            onClose={() => setShowSessionPanel(false)}
          />
        )}
      </AnimatePresence>

      {/* Click outside to close dropdowns */}
      {(modeDropdownOpen || showModelSelector) && (
        <div
          className="fixed inset-0 z-10"
          onClick={() => { setModeDropdownOpen(false); setShowModelSelector(false); }}
        />
      )}
    </div>
  );
}
