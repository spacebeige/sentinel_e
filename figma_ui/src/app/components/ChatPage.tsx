import { useTheme } from "next-themes";
import { MODELS as AVAILABLE_MODELS, getModelConfig, getModeConfig, ALL_RUNTIME_MODES } from "../config/runtime";
import { supabase } from "../lib/supabase";
import { Link } from "react-router";
import { useState, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
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
  Menu,
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
  shareChat,
  runOmegaKill,
  syncConversation,
  syncMessage,
  type SentinelRunResponse,
  type HealthStatus,
  type ChatHistoryItem,
  type OmegaMetadata,
  type OmegaBoundaryResult,
  type OmegaReasoningTrace,
  type ConfidenceEvolution,
} from "../api";
import { OmegaInsightPanel } from "./OmegaInsightPanel";
import { useChatInteraction } from "../context/ChatInteractionContext";
import { useSupabaseAuth } from "../hooks/useSupabaseAuth";
import { trackMessageSent, trackConversationStarted } from '../services/analyticsService';
import { SessionAnalyticsPanel } from "./SessionAnalyticsPanel";
import { CinematicOrchestratorLoader } from "./CinematicOrchestratorLoader";
import { CinematicDebatePanel } from "./CinematicDebatePanel";
import { CinematicEvidencePanel } from "./CinematicEvidencePanel";
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

// ── Main ChatPage ────────────────────────────────────────────────────────────
export function ChatPage() {
  const [selectedModel, setSelectedModel] = useState<string | null>("llama-3-3-70b");
  const [selectedMode, setSelectedMode] = useState<string | null>(null);
  const [runtimeTier, setRuntimeTier] = useState<"standard" | "pro">("standard");
  const [isOrchestrationExpanded, setIsOrchestrationExpanded] = useState(false);
  const [runtimePreferences, setRuntimePreferences] = useState({
    responseStyle: "balanced",
    debateDepth: 6,
  });

  useEffect(() => {
    document.title = "Chat • Sentinel-E";
  }, []);

  useEffect(() => {
    console.log({
      runtimeTier,
      selectedModel,
      selectedMode,
    });
  }, [runtimeTier, selectedModel, selectedMode]);

  const { user } = useSupabaseAuth();
  
  // Real subscription tier from user metadata
  const subscriptionTier = user?.user_metadata?.subscription || "standard";

  const activeModel = getModelConfig(selectedModel || "llama-3-3-70b");
  const availableModels = AVAILABLE_MODELS;
  const effectiveMode = (runtimeTier === "pro" && isOrchestrationExpanded && selectedMode) ? selectedMode : "standard";

  const availableModes = [
    { id: "debate", name: "Debate", color: "#ef4444" },
    { id: "glass", name: "Glass", color: "#8b5cf6" },
    { id: "evidence", name: "Evidence", color: "#06b6d4" },
    { id: "synthesis", name: "Synthesis", color: "#10b981" },
  ];
  // State
  const [messages, setMessages] = useState<Message[]>([{
    id: "welcome",
    role: "assistant",
    content: "Hello! How can I help you today?",
    timestamp: new Date(),
  }]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [healthData, setHealthData] = useState<HealthStatus | null>(null);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  // Sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");

  // Chat history state
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Chat interaction context
  const { isHistoryOpen, toggleHistory, newChatTriggered, /* removed (runtimeTier === "pro") */ } = useChatInteraction();

  // Pro features state
      const [expandedMeta, setExpandedMeta] = useState<string | null>(null);
  const [showSessionPanel, setShowSessionPanel] = useState(false);
  const [hoveredMessage, setHoveredMessage] = useState<string | null>(null);
  const [copiedMessage, setCopiedMessage] = useState<string | null>(null);

  // Share & Copy state
  const [shareSuccess, setShareSuccess] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // File upload
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Mode dropdown
  const [modeDropdownOpen, setModeDropdownOpen] = useState(false);
  const modeTriggerRef = useRef<HTMLButtonElement>(null);
  const [modeDropdownCoords, setModeDropdownCoords] = useState({ top: 0, left: 0 });

  useEffect(() => {
    if (modeDropdownOpen && modeTriggerRef.current) {
      const rect = modeTriggerRef.current.getBoundingClientRect();
      setModeDropdownCoords({ top: rect.bottom + 6, left: rect.left });
    }
  }, [modeDropdownOpen]);

  useEffect(() => {
    if (!modeDropdownOpen) return;
    const updatePosition = () => {
      if (modeTriggerRef.current) {
        const rect = modeTriggerRef.current.getBoundingClientRect();
        setModeDropdownCoords({ top: rect.bottom + 6, left: rect.left });
      }
    };
    window.addEventListener('resize', updatePosition);
    return () => window.removeEventListener('resize', updatePosition);
  }, [modeDropdownOpen]);

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
  const { theme, setTheme } = useTheme();
  const isDark = theme === "dark";

  const toggleTheme = () => {
    setTheme(isDark ? "light" : "dark");
  };

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
    } else {
      setBackendOnline(false);
      setHealthData(null);
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
      if (saved.subMode) setSelectedMode(saved.subMode);
      setRuntimeTier(saved.runtimeTier || "standard");
      setIsOrchestrationExpanded(saved.isOrchestrationExpanded || false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch persisted chat history once user is authenticated
  useEffect(() => {
    const onlyWelcome = messages.length === 1 && messages[0]?.id === "welcome";
    if (user && currentChatId && onlyWelcome) {
      getChatMessages(currentChatId)
        .then(msgs => {
          const restored: Message[] = msgs.map((m, i) => ({
            id: `restored-${i}`,
            role: m.role as "user" | "assistant",
            content: m.content,
            timestamp: m.timestamp ? new Date(m.timestamp) : new Date(),
          }));
          if (restored.length > 0) {
            setMessages(restored);
          }
        })
        .catch(err => {
          console.error("Failed to restore session chat on load:", err);
        });
    }
  }, [user, currentChatId]); // Only trigger when user is authenticated and we have a chat ID

  useEffect(() => {
    if (!user) return;
    supabase
      .from("profiles")
      .select("runtime_preference,favorite_model,response_style,debate_depth")
      .eq("id", user.id)
      .single()
      .then(({ data }) => {
        if (!data) return;
        if (data.favorite_model) setSelectedModel(data.favorite_model);
        if (data.runtime_preference === "pro") {
          setRuntimeTier("pro");
          setIsOrchestrationExpanded(true);
        }
        setRuntimePreferences({
          responseStyle: data.response_style || "balanced",
          debateDepth: Number(data.debate_depth || 6),
        });
      });
  }, [user]);

  useEffect(() => {
    if (errorMessage) {
      const t = setTimeout(() => setErrorMessage(null), 5000);
      return () => clearTimeout(t);
    }
  }, [errorMessage]);

  useEffect(() => {
    persist({
      chatId: currentChatId,
      mode: (runtimeTier === "pro") ? "experimental" : "standard",
      subMode: selectedMode,
      runtimeTier, isOrchestrationExpanded,
      killOverride: glassState.killOverride,
    });
  }, [currentChatId, runtimeTier, isOrchestrationExpanded, selectedMode, glassState.killOverride, persist]);

  // ── Load chat history ──────────────────────────────────────────────────────
  const loadChatHistory = useCallback(async () => {
    if (!backendOnline || !user) return;
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
    if (sidebarOpen && backendOnline && user) {
      loadChatHistory();
    }
  }, [sidebarOpen, backendOnline, user, loadChatHistory]);

  // ── Restore chat ───────────────────────────────────────────────────────────
  const restoreChat = async (chatItem: ChatHistoryItem) => {
    if (!backendOnline || !user) return;
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
      const metadata = chatItem.machine_metadata as Record<string, unknown> | undefined;
      const restoredModel = metadata?.selected_model || metadata?.winning_model;
      if (typeof restoredModel === "string") setSelectedModel(restoredModel);
      if (chatItem.mode?.includes("mco") || metadata?.sub_mode) {
        setRuntimeTier("pro");
        setIsOrchestrationExpanded(true);
        if (typeof metadata?.sub_mode === "string") setSelectedMode(metadata.sub_mode);
      }
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
    let urlToShare = window.location.href;
    if (currentChatId && backendOnline) {
      try {
        const result = await shareChat(currentChatId);
        urlToShare = window.location.origin + "/share/" + result.share_token;
      } catch {
        // Fallback to current URL if backend sharing fails
      }
    }

    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Sentinel-E Conversation',
          url: urlToShare
        });
        setShareSuccess(true);
        setTimeout(() => setShareSuccess(false), 2500);
      } catch (err) {
        // Fallback to clipboard if share dialog fails (and wasn't cancelled)
        if ((err as Error).name !== "AbortError") {
          navigator.clipboard.writeText(urlToShare);
          setShareSuccess(true);
          setToastMessage("Share link copied");
          setTimeout(() => { setShareSuccess(false); setToastMessage(null); }, 2500);
        }
      }
    } else {
      navigator.clipboard.writeText(urlToShare);
      setShareSuccess(true);
      setToastMessage("Share link copied");
      setTimeout(() => { setShareSuccess(false); setToastMessage(null); }, 2500);
    }
  };

  const handleCopyChat = async () => {
    try {
      const text = messages.map(m => `${m.role === "user" ? "User" : "Assistant"}:\n${m.content}`).join("\n\n");
      await navigator.clipboard.writeText(text);
      setCopySuccess(true);
      setToastMessage("Copied conversation");
      setTimeout(() => { setCopySuccess(false); setToastMessage(null); }, 2500);
    } catch (e) {
      setToastMessage("Copy failed");
      setTimeout(() => setToastMessage(null), 2500);
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

    if (user && !currentChatId) {
      trackConversationStarted(user.id);
    }
    if (user) {
      trackMessageSent(user.id, effectiveMode, selectedModel);
    }

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

        if (runtimeTier === "pro" && selectedMode) {
          response = await runExperimental(
            userText,
            selectedMode,
            runtimePreferences.debateDepth,
            currentChatId || undefined,
            glassState.killOverride,
            attachedFile || undefined,
            ac.signal,
            {
              responseStyle: runtimePreferences.responseStyle,
              preferences: {
                default_mode: selectedMode,
                default_model: selectedModel,
              },
            }
          );
        } else {
          response = await runStandard(
            userText,
            selectedModel || "llama-3-3-70b",
            currentChatId || undefined,
            attachedFile || undefined,
            ac.signal,
            {
              responseStyle: runtimePreferences.responseStyle,
              preferences: {
                default_mode: "standard",
                default_model: selectedModel,
              },
            }
          );
        }

        if (ac.signal.aborted) return;

        let responseChatId = currentChatId || response.chat_id;
        
        if (!currentChatId && response.chat_id) {
          setCurrentChatId(response.chat_id);
          syncConversation(response.chat_id, userText.slice(0, 50) + (userText.length > 50 ? '...' : ''), selectedMode || "standard");
        }
        
        if (responseChatId) {
          syncMessage(responseChatId, "user", userText + (attachedFile ? `\n\n[Attached: ${attachedFile.name}]` : ""));
        }

        if (selectedMode === "debate" && response.omega_metadata) setDebateState((p) => mergeDebateResult(p, response.omega_metadata));
        if (selectedMode === "glass" && response.omega_metadata) setGlassState((p) => mergeGlassState(p, response.omega_metadata));
        if (selectedMode === "evidence" && response.omega_metadata) setEvidenceState((p) => mergeEvidenceState(p, response.omega_metadata));

        const assistantContent = response.formatted_output || response.data?.priority_answer || "No response generated.";

        if (responseChatId) {
          syncMessage(responseChatId, "assistant", assistantContent);
        }

        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: assistantContent,
          timestamp: new Date(),
          mode: response.sub_mode || selectedMode,
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
        setIsTyping(false);
        removeFile();
      }
    } else {
      setErrorMessage("Backend is offline. Real model execution is unavailable.");
      setIsTyping(false);
      removeFile();
    }
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
      content: "Hello! How can I help you today?",
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
        mode: (runtimeTier === "pro") ? "experimental" : "standard",
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
                
                {message.omegaMetadata?.orchestration_run?.event_timeline && (
                  <div className="pt-2" style={{ borderTop: `1px solid ${borderColor}` }}>
                    <div className="flex items-center gap-1 mb-2">
                      <Sparkles className="w-3 h-3 text-[#8b5cf6]" />
                      <span style={{ fontSize: "10px", fontWeight: 600, color: "#8b5cf6", textTransform: "uppercase", letterSpacing: "0.05em" }}>Orchestration Timeline</span>
                    </div>
                    <div className="flex flex-col gap-1.5 pl-1">
                      {message.omegaMetadata.orchestration_run.event_timeline.map((evt: any, i: number) => (
                        <div key={i} className="flex items-start gap-2">
                          <div className="w-1 h-1 rounded-full mt-1.5 bg-indigo-500/60 shrink-0" />
                          <div className="text-[11px] font-mono text-slate-500">
                            {new Date(evt.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                          </div>
                          <div className="text-[11px] text-slate-400 capitalize">
                            {evt.event_type.replace(/_/g, ' ')}
                          </div>
                        </div>
                      ))}
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
      className="relative flex h-screen w-full overflow-hidden overflow-x-hidden"
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
            className="flex flex-col flex-shrink-0 h-screen z-20"
            style={{
              width: sidebarOpen ? "280px" : "0px",
              transform: sidebarOpen ? "translateX(0)" : "translateX(-100%)",
              opacity: sidebarOpen ? 1 : 0,
              pointerEvents: sidebarOpen ? "auto" : "none",
              overflow: "hidden",
              transition: "width 300ms cubic-bezier(0.16,1,0.3,1), transform 300ms cubic-bezier(0.16,1,0.3,1), opacity 220ms ease",
              background: sidebarBg,
              borderRight: sidebarOpen ? `1px solid ${borderColor}` : "none",
            }}
          >
            {/* Sidebar header */}
            <div
              className="flex items-center justify-between px-4 py-3.5"
              style={{ borderBottom: `1px solid ${borderColor}` }}
            >
              {sidebarOpen ? (
                <Link to="/" className="flex items-center">
                  <img src="/logo.png" alt="Logo" className="h-[22px] w-auto transition-transform hover:scale-105" />
                </Link>
              ) : (
                <Link to="/" className="w-6 h-6 flex-shrink-0 rounded-lg bg-gradient-to-br from-[#3b82f6] to-[#06b6d4] flex items-center justify-center transition-transform hover:scale-105">
                  <span className="text-white text-[10px] font-bold">S</span>
                </Link>
              )}
              <div className="flex items-center gap-1">
                
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
              {historyLoading ? (
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
              className="px-2 py-3 space-y-0.5 flex-shrink-0 relative z-20 mt-auto"
              style={{ borderTop: `1px solid ${borderColor}` }}
            >
              <button
                onClick={handleShareChat}
                className={`w-full flex items-center transition-colors ${sidebarOpen ? 'gap-2.5 px-3 py-2.5 rounded-xl' : 'justify-center p-2.5 rounded-xl'}`}
                style={{ color: shareSuccess ? "#10b981" : textSecondary }}
                onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
              >
                {shareSuccess ? <Check className="w-4 h-4 flex-shrink-0" /> : <Share2 className="w-4 h-4 flex-shrink-0" />}
                {sidebarOpen && <span style={{ fontSize: "13px", fontWeight: 500 }}>{shareSuccess ? "Shared!" : "Share Chat"}</span>}
              </button>
              
              <button
                onClick={handleCopyChat}
                className={`w-full flex items-center transition-colors ${sidebarOpen ? 'gap-2.5 px-3 py-2.5 rounded-xl' : 'justify-center p-2.5 rounded-xl'}`}
                style={{ color: copySuccess ? "#10b981" : textSecondary }}
                onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
              >
                {copySuccess ? <Check className="w-4 h-4 flex-shrink-0" /> : <Copy className="w-4 h-4 flex-shrink-0" />}
                {sidebarOpen && <span style={{ fontSize: "13px", fontWeight: 500 }}>{copySuccess ? "Copied!" : "Copy Chat"}</span>}
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



      {/* ── Main Chat Area ── */}
      <div className="flex-1 flex flex-col h-screen relative overflow-hidden" style={{ background: chatBg, transition: "margin 300ms cubic-bezier(0.16,1,0.3,1), width 300ms cubic-bezier(0.16,1,0.3,1)" }}>
        {import.meta.env.DEV && (
          <div className="absolute top-4 right-4 z-50 p-4 rounded-xl text-xs font-mono shadow-xl backdrop-blur-md" style={{ background: isDark ? 'rgba(0,0,0,0.8)' : 'rgba(255,255,255,0.8)', border: isDark ? '1px solid rgba(255,255,255,0.1)' : '1px solid rgba(0,0,0,0.1)' }}>
            <div className="font-bold mb-2">Runtime Diagnostics</div>
            <div>Tier: {runtimeTier}</div>
            <div>Model: {selectedModel || 'null'}</div>
            <div>Mode: {selectedMode || 'null'}</div>
            <div>Session: {currentChatId || 'none'}</div>
            <div>User: {user?.email || 'none'}</div>
          </div>
        )}

        {/* ── TOP BAR ─────────────────────────────────────────────────── */}
        <div
          className="relative flex items-center justify-between w-full h-14 px-4 z-40"
          style={{
            background: isDark ? "rgba(8,9,14,0.85)" : "rgba(255,255,255,0.85)",
            backdropFilter: "blur(20px)",
            WebkitBackdropFilter: "blur(20px)",
            borderBottom: `1px solid ${borderColor}`,
          }}
        >
          {/* LEFT */}
          <div className="flex items-center">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="flex items-center justify-center w-8 h-8 rounded-xl transition-all duration-300"
              title="Toggle Sidebar"
              style={{
                color: textSecondary,
                background: isDark ? "rgba(18,18,24,0.72)" : "rgba(255,255,255,0.72)",
                backdropFilter: "blur(20px) saturate(180%)",
                WebkitBackdropFilter: "blur(20px) saturate(180%)",
                border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)",
              }}
              onMouseEnter={(e) => { 
                e.currentTarget.style.transform = "translateY(-1px)";
                e.currentTarget.style.color = textPrimary;
              }}
              onMouseLeave={(e) => { 
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.color = textSecondary;
              }}
            >
              <div className="transition-transform duration-300">
                {sidebarOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
              </div>
            </button>
          </div>

          {/* CENTER */}
          <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-3 z-50">
            <div className="relative">
              <button
                ref={modeTriggerRef}
                onClick={() => setModeDropdownOpen(!modeDropdownOpen)}
                className="flex items-center justify-center gap-2 transition-all duration-300 pointer-events-auto"
                style={{
                  height: "40px",
                  padding: "0 16px",
                  borderRadius: "18px",
                  fontSize: "14px",
                  fontWeight: 600,
                  background: isDark ? "rgba(18,18,24,0.72)" : "rgba(255,255,255,0.72)",
                  backdropFilter: "blur(24px) saturate(180%)",
                  WebkitBackdropFilter: "blur(24px) saturate(180%)",
                  border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)",
                  boxShadow: isDark
                    ? "0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05)"
                    : "0 6px 20px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.7)",
                  color: isDark ? "#f5f5f7" : "#1d1d1f",
                }}
              >
                Sentinel {(runtimeTier === "pro") ? "Pro" : "Standard"}
                <ChevronDown className={`w-4 h-4 transition-transform duration-300 ${modeDropdownOpen ? "rotate-180" : ""}`} style={{ color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)" }} />
              </button>

              {typeof document !== 'undefined' && createPortal(
                <AnimatePresence>
                  {modeDropdownOpen && (
                    <motion.div
                      key="sentinel-mode-dropdown"
                      initial={{ opacity: 0, y: -6, scale: 0.97 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -6, scale: 0.97 }}
                      transition={{ duration: 0.15, ease: [0.16, 1, 0.3, 1] }}
                      className="fixed w-52 z-[120] pointer-events-auto"
                      style={{
                        top: modeDropdownCoords.top,
                        left: modeDropdownCoords.left,
                        background: isDark ? "rgba(18,18,24,0.72)" : "rgba(255,255,255,0.72)",
                        backdropFilter: "blur(30px) saturate(180%)",
                        WebkitBackdropFilter: "blur(30px) saturate(180%)",
                        border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)",
                        borderRadius: "22px",
                        padding: "8px",
                        boxShadow: isDark
                          ? "0 16px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05)"
                          : "0 10px 30px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.7)",
                      }}
                    >
                      <div className="flex flex-col gap-1">
                        {[
                          { id: "standard", label: "Standard", sub: "Simple AI experience", pro: false },
                          { id: "pro", label: "Pro", sub: "Full orchestration · multi-model", pro: true },
                        ].map((m) => (
                          <button
                            key={m.id}
                            onClick={() => { setRuntimeTier(m.pro ? "pro" : "standard"); setModeDropdownOpen(false); if (!m.pro) { setSelectedMode(null); setSelectedModel("llama-3-3-70b"); } }}
                            className="w-full flex items-start gap-2.5 px-3 py-2.5 rounded-xl transition-colors text-left"
                            style={{
                              background: (runtimeTier === "pro") === m.pro ? (isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)") : "transparent",
                            }}
                            onMouseEnter={(e) => { if ((runtimeTier === "pro") !== m.pro) e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)"; }}
                            onMouseLeave={(e) => { if ((runtimeTier === "pro") !== m.pro) e.currentTarget.style.background = "transparent"; }}
                          >
                            <div className="mt-0.5 w-4 h-4 rounded-full flex-shrink-0 flex items-center justify-center" style={{ background: m.pro ? "rgba(139,92,246,0.15)" : "rgba(59,130,246,0.15)" }}>
                              <div className="w-1.5 h-1.5 rounded-full" style={{ background: m.pro ? "#8b5cf6" : "#3b82f6" }} />
                            </div>
                            <div>
                              <div style={{ fontSize: "13px", fontWeight: 600, color: isDark ? "#fff" : "#000" }}>{m.label}</div>
                              <div style={{ fontSize: "11px", color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)", marginTop: "1px" }}>{m.sub}</div>
                            </div>
                            {(runtimeTier === "pro") === m.pro && <Check className="w-3.5 h-3.5 ml-auto mt-1 text-[#3b82f6]" />}
                          </button>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>,
                document.body
              )}
            </div>

            </div>

        {/* RIGHT */}
          <div className="flex items-center gap-2">
            <button
              onClick={toggleTheme}
              className="p-2 rounded-xl transition-all duration-300"
              style={{ color: textSecondary }}
              onMouseEnter={(e) => { e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; e.currentTarget.style.transform = "translateY(-1px)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.transform = "translateY(0)"; }}
              title="Toggle Theme"
            >
              {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
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
        <div className="flex-1 overflow-y-auto px-4 py-8" onClick={() => { setModeDropdownOpen(false); setIsOrchestrationExpanded(false); }}>
          <div className="max-w-2xl mx-auto space-y-5">
            <AnimatePresence>
              {messages.map((message) => {
                const msgMode = message.mode && message.mode !== "standard" ? getModeConfig(message.mode) : null;
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
                        style={{
                          borderRadius: message.role === "user" ? "20px 20px 5px 20px" : "20px 20px 20px 5px",
                          background: isDark
                            ? "rgba(255,255,255,0.08)"
                            : "rgb(240,240,245)",
                          color: isDark ? "#f5f5f7" : "#1d1d1f",
                          padding: message.role === "user" ? "12px 16px" : "0",
                          boxShadow: isDark
                            ? "inset 0 1px 0 rgba(255,255,255,0.05)"
                            : "0 2px 8px rgba(0,0,0,0.03)",
                          border: isDark 
                            ? "1px solid rgba(255,255,255,0.12)"
                            : "1px solid rgba(0,0,0,0.04)",
                          backdropFilter: isDark ? "blur(20px)" : "none",
                          WebkitBackdropFilter: isDark ? "blur(20px)" : "none",
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
                              color: isDark ? "#f5f5f7" : "#1d1d1f",
                            }}
                          >
                            {message.content}
                          </p>

                          {/* Omega insights */}
                          {renderOmegaInsights(message)}
                          {message.omegaMetadata?.debate_result && <CinematicDebatePanel metadata={message.omegaMetadata} />}
                          {message.omegaMetadata?.evidence_result && <CinematicEvidencePanel metadata={message.omegaMetadata} />}

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
            {/* Cinematic Typing Indicator */}
            <div className="flex justify-start pb-4">
              <div className="max-w-[80%]">
                <CinematicOrchestratorLoader
                  isLoading={isTyping}
                  mode={(runtimeTier === "pro") ? "experimental" : "standard"}
                  subMode={selectedMode}
                />
              </div>
            </div>

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ── FLOATING INPUT DOCK ──────────────────────────────────────────── */}
        <div
          className="px-4 pt-3 z-[5] relative"
          style={{
            paddingBottom: "calc(24px + env(safe-area-inset-bottom, 0px))",
            background: isDark
              ? "linear-gradient(to top, rgba(8,9,14,1) 60%, rgba(8,9,14,0) 100%)"
              : "linear-gradient(to top, rgba(255,255,255,1) 60%, rgba(255,255,255,0) 100%)",
          }}
        >
          <div className="max-w-2xl mx-auto relative flex flex-col gap-[10px] z-[5]">

            {/* Input container */}
            <div
              className="relative rounded-[24px] transition-all duration-300 z-10 flex-shrink-0"
              style={{
                background: inputBg,
                border: selectedMode
                  ? `1px solid ${getModeConfig(selectedMode).color}40`
                  : `1px solid ${borderColor}`,
                boxShadow: selectedMode
                  ? `0 4px 24px -4px ${getModeConfig(selectedMode).color}20, ${isDark ? "0 0 0 1px rgba(255,255,255,0.04)" : "0 0 0 1px rgba(0,0,0,0.04)"}`
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
                <div className="flex items-center gap-0.5 pb-0.5 relative" style={{ overflow: "visible" }}>
                  
                  {/* Floating Mode Panel attached to + button */}
                  <AnimatePresence>
                    {isOrchestrationExpanded && runtimeTier === "pro" && (
                      <motion.div
                        initial={{ opacity: 0, y: 8, scale: 0.96 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 8, scale: 0.96 }}
                        transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                        className="absolute z-[140] pointer-events-auto w-[240px] p-3"
                        style={{
                          bottom: "calc(100% + 12px)",
                          left: "0",
                          background: isDark ? "rgba(18,18,24,0.78)" : "rgba(255,255,255,0.72)",
                          backdropFilter: "blur(30px) saturate(180%)",
                          WebkitBackdropFilter: "blur(30px) saturate(180%)",
                          border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)",
                          borderRadius: "22px",
                          boxShadow: isDark 
                            ? "0 18px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05)"
                            : "0 10px 30px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.7)",
                        }}
                      >
                         <div className="mb-2 px-2 text-[11px] font-semibold tracking-wider uppercase" style={{ color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)" }}>Orchestration Modes</div>
                         <div className="grid grid-cols-2 gap-1.5">
                           {availableModes.map((m) => {
                             const isActive = selectedMode === m.id;
                             return (
                               <button
                                 key={m.id}
                                 onClick={() => { 
                                   setSelectedMode(isActive ? null : m.id); 
                                   if (!isActive) setSelectedModel("llama-3-3-70b");
                                   setIsOrchestrationExpanded(false); 
                                 }}
                                 className="flex flex-col items-start gap-1.5 p-2.5 rounded-2xl transition-all text-left"
                                 style={{
                                   background: isActive ? m.color : (isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)"),
                                   border: isActive ? `1px solid ${m.color}` : "1px solid transparent",
                                   color: isActive ? "#ffffff" : (isDark ? "#fff" : "#000"),
                                   boxShadow: isActive ? `0 4px 16px ${m.color}40` : "none",
                                 }}
                                 onMouseEnter={(e) => { 
                                   if (!isActive) e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)"; 
                                   e.currentTarget.style.transform = "translateY(-1px)";
                                 }}
                                 onMouseLeave={(e) => { 
                                   if (!isActive) e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)"; 
                                   e.currentTarget.style.transform = "translateY(0)";
                                 }}
                               >
                                 <div className="p-1.5 rounded-xl" style={{ background: isActive ? "rgba(255,255,255,0.2)" : (isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)"), color: isActive ? "#fff" : m.color }}>
                                   <div className="w-2.5 h-2.5 rounded-full" style={{ background: m.color }} />
                                 </div>
                                 <div>
                                   <div className="text-[13px] font-semibold">{m.name}</div>
                                 </div>
                               </button>
                             );
                           })}
                         </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Plus — opens Mode selector in Pro mode */}
                  {runtimeTier === "pro" && (
                    <button
                      onClick={() => {
                        setIsOrchestrationExpanded(!isOrchestrationExpanded);
                        setModeDropdownOpen(false);
                      }}
                      className="flex items-center justify-center rounded-full transition-all duration-300 flex-shrink-0"
                      style={{ 
                        width: "40px",
                        height: "40px",
                        background: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.05)",
                        color: isOrchestrationExpanded ? "#3b82f6" : textSecondary,
                        backdropFilter: "blur(12px)",
                        WebkitBackdropFilter: "blur(12px)",
                        boxShadow: isDark ? "inset 0 1px 0 rgba(255,255,255,0.05)" : "none",
                      }}
                      onMouseEnter={(e) => { 
                        e.currentTarget.style.transform = "scale(1.04)"; 
                        if (isDark) e.currentTarget.style.background = "rgba(255,255,255,0.12)";
                      }}
                      onMouseLeave={(e) => { 
                        e.currentTarget.style.transform = "scale(1)";
                        e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.05)";
                      }}
                    >
                      <Plus className="w-5 h-5" />
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
                    selectedMode
                      ? getModeConfig(selectedMode).placeholder
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
                        : selectedMode
                          ? getModeConfig(selectedMode).color
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
              className="text-center relative z-10 flex-shrink-0"
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

      {/* ── TOAST ──────────────────────────────────────────────────────────── */}
      <AnimatePresence>
        {toastMessage && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[200]"
            style={{
              background: isDark ? "rgba(18,18,24,0.82)" : "rgba(255,255,255,0.88)",
              backdropFilter: "blur(24px)",
              WebkitBackdropFilter: "blur(24px)",
              color: isDark ? "#fff" : "#000",
              fontSize: "13px",
              fontWeight: 500,
              borderRadius: "999px",
              padding: "10px 18px",
              boxShadow: "0 10px 30px rgba(0,0,0,0.18)",
            }}
          >
            {toastMessage}
          </motion.div>
        )}
      </AnimatePresence>

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
      {(modeDropdownOpen || isOrchestrationExpanded) && (
        <div
          className="fixed inset-0 z-10"
          onClick={() => { setModeDropdownOpen(false); setIsOrchestrationExpanded(false); }}
        />
      )}
    </div>
  );
}
