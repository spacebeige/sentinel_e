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
  Wifi,
  WifiOff,
  AlertCircle,
  ThumbsUp,
  ThumbsDown,
  History,
  ChevronRight,
  Activity,
  Brain,
  Shield,
  BarChart3,
  Zap,
  Share2,
  Skull,
  Loader2,
  MessageSquare,
  PanelRightOpen,
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
import { SessionAnalyticsPanel } from "./SessionAnalyticsPanel";
import { CrossAnalysisTrigger } from "./CrossAnalysisPanel";

// Service layer — session persistence + mode managers
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

const models = [
  { id: "sentinel-std", name: "Sentinel-E Standard", provider: "Standard", color: "#3b82f6", category: "standard" },
  { id: "qwen", name: "Qwen 2.5", provider: "Standard", color: "#06b6d4", category: "standard" },
  { id: "mistral", name: "Mistral Large", provider: "Standard", color: "#f59e0b", category: "standard" },
  { id: "groq", name: "Groq LPU", provider: "Standard", color: "#10b981", category: "standard" },
  { id: "sentinel-exp", name: "Sentinel-E Pro", provider: "Experimental", color: "#8b5cf6", category: "experimental" },
];

const proSubModes = [
  { id: "debate", label: "Debate Mode", icon: <Swords className="w-3.5 h-3.5" />, color: "#ef4444", description: "Argues both sides of a topic so you can decide", placeholder: "Give me a topic to debate..." },
  { id: "glass", label: "Glass Mode", icon: <Gem className="w-3.5 h-3.5" />, color: "#8b5cf6", description: "Shows its full reasoning chain — nothing hidden", placeholder: "Ask something and I'll show my thinking..." },
  { id: "evidence", label: "Evidence Mode", icon: <FileSearch className="w-3.5 h-3.5" />, color: "#06b6d4", description: "Every claim backed by a cited source", placeholder: "What do you need evidence for..." },
];

// Fallback responses when backend is offline
const modeResponses: Record<string, string[]> = {
  debate: [
    "\u2694\ufe0f **Debate Mode Active**\n\n**FOR:**\nThis approach has significant merit. Studies consistently show improved outcomes when applied correctly. The efficiency gains alone justify adoption \u2014 teams report 40% faster iteration cycles. Additionally, the long-term scalability makes it a sound investment.\n\n**AGAINST:**\nHowever, the counterarguments are worth weighing. The upfront learning curve is steep, and not every team has the bandwidth. There\u2019s also the vendor lock-in risk. Some practitioners argue simpler alternatives achieve 80% of the benefit at a fraction of the cost.\n\n**VERDICT:** The answer depends on your team\u2019s size, timeline, and risk tolerance. Want me to dig into a specific angle?",
    "\u2694\ufe0f **Debate Mode Active**\n\n**Side A \u2014 The Case For:**\nProponents point to three things: broader accessibility, lower barriers to entry, and a growing body of real-world success stories. The momentum is clearly heading this direction.\n\n**Side B \u2014 The Case Against:**\nSkeptics raise valid concerns about quality control, sustainability of current growth, and whether the hype outpaces the substance. History is full of technologies that plateaued after initial excitement.\n\n**My take:** Both sides have merit. The truth likely sits somewhere in the middle \u2014 adoption makes sense, but with eyes open.",
  ],
  glass: [
    "\ud83d\udd0d **Glass Mode \u2014 Full Reasoning Chain**\n\n**Step 1 \u2014 Parsing your question:**\nI\u2019m identifying the core intent. You\u2019re asking about [topic], which touches on multiple domains.\n\n**Step 2 \u2014 Retrieving relevant knowledge:**\nPulling from training data related to this area. I have moderate-to-high confidence here, but I\u2019ll flag any gaps.\n\n**Step 3 \u2014 Weighing approaches:**\nThere are ~3 reasonable paths. I\u2019m ranking them by reliability, not just popularity.\n\n**Step 4 \u2014 Forming a response:**\nI\u2019m going with the most grounded answer. Here\u2019s what I\u2019d recommend, and here\u2019s *why* I chose it over the alternatives.\n\n**Confidence level:** ~85%. The 15% uncertainty is around edge cases I can\u2019t fully verify.",
    "\ud83d\udd0d **Glass Mode \u2014 Full Reasoning Chain**\n\n**What I understood:** You want clarity on this topic.\n\n**What I considered:** Three possible interpretations of your question. I went with the most likely one based on context.\n\n**What I don\u2019t know:** I\u2019m not 100% sure about the latest developments post-2024. I\u2019ll tell you what I\u2019m confident about and flag the rest.\n\n**My reasoning path:**\n1. Start from first principles\n2. Cross-reference with known patterns\n3. Arrive at the simplest accurate explanation\n\n**Final answer:** Here\u2019s what I believe is correct, and here\u2019s exactly where my certainty drops off.",
  ],
  evidence: [
    "\ud83d\udccb **Evidence Mode \u2014 Sources Cited**\n\nBased on available research:\n\n1. The primary mechanism works through attention layers that weigh token relationships \u00b9\n2. Performance scales roughly as a power law with compute and data \u00b2 \n3. Recent benchmarks show significant improvements in reasoning tasks, with accuracy gains of 15-30% over previous generations \u00b3\n\n---\n**Sources:**\n\u00b9 Vaswani et al., \"Attention Is All You Need\" (2017), NeurIPS\n\u00b2 Kaplan et al., \"Scaling Laws for Neural Language Models\" (2020), OpenAI\n\u00b3 Multiple benchmark results, MMLU & HumanEval (2024)",
    "\ud83d\udccb **Evidence Mode \u2014 Sources Cited**\n\nHere\u2019s what the evidence says:\n\n\u2022 Claim: This approach outperforms alternatives in 7 out of 10 benchmarks \u2192 **Supported** by peer-reviewed evaluations \u00b9\n\u2022 Claim: Adoption has grown 3x year-over-year \u2192 **Partially supported**, growth varies by region \u00b2\n\u2022 Claim: No significant drawbacks \u2192 **Not supported**, several studies note trade-offs \u00b3\n\n---\n**Sources:**\n\u00b9 Stanford AI Index Report (2024)\n\u00b2 McKinsey Global Survey on AI (2024)\n\u00b3 MIT Technology Review, \"The Hidden Costs\" (2023)",
  ],
};

const sampleResponses = [
  "That\u2019s a great question! Let me break it down for you. The key concept here involves understanding how large language models process and generate text through a mechanism called attention. Each token in the input is compared against every other token to determine relevance, creating a rich contextual understanding.",
  "I\u2019d be happy to help with that! Here\u2019s a comprehensive approach:\n\n1. **Start with the fundamentals** - Understanding the core architecture\n2. **Practice with examples** - Hands-on experimentation\n3. **Iterate and refine** - Continuous improvement\n\nWould you like me to dive deeper into any of these areas?",
  "Based on my analysis, there are several interesting perspectives to consider. The field has evolved rapidly, with new breakthroughs emerging almost weekly. The most significant recent development has been the improvement in reasoning capabilities, allowing models to tackle increasingly complex problems.",
];

export function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Hello! I'm Sentinel-E, your AI assistant powered by the Omega Cognitive Kernel. How can I help you today?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [selectedModel, setSelectedModel] = useState(models[0]);
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [activeSubMode, setActiveSubMode] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [healthData, setHealthData] = useState<HealthStatus | null>(null);
  const [kernelData, setKernelData] = useState<KernelStatus | null>(null);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Chat history sidebar
  const [showHistory, setShowHistory] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Omega insights panel
  const [expandedMeta, setExpandedMeta] = useState<string | null>(null);

  // Session analytics right sidebar
  const [showSessionPanel, setShowSessionPanel] = useState(false);

  // File upload
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const healthCheckRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // AbortController for race condition safety (cancels in-flight request on new send)
  const abortRef = useRef<AbortController | null>(null);

  // Session persistence hook
  const { restore, persist, reset: resetSession } = useSessionPersistence();

  // Mode-specific state managers (in-memory, survive across messages)
  const [debateState, setDebateState] = useState<DebateState>(createDebateState(6));
  const [glassState, setGlassState] = useState<GlassState>(createGlassState());
  const [evidenceState, setEvidenceState] = useState<EvidenceState>(createEvidenceState());

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Health check + kernel status on mount + periodic polling
  const performHealthCheck = useCallback(async () => {
    const health = await checkHealth();
    if (health) {
      setBackendOnline(true);
      setHealthData(health);
      // Also fetch kernel status
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
    return () => {
      if (healthCheckRef.current) clearInterval(healthCheckRef.current);
    };
  }, [performHealthCheck]);

  // Restore session from localStorage on mount
  useEffect(() => {
    const saved = restore();
    if (saved.chatId) {
      setCurrentChatId(saved.chatId);
      // Restore model selection
      const savedModel = models.find((m) => m.id === saved.selectedModelId);
      if (savedModel) setSelectedModel(savedModel);
      if (saved.subMode) setActiveSubMode(saved.subMode);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Clear error after timeout
  useEffect(() => {
    if (errorMessage) {
      const t = setTimeout(() => setErrorMessage(null), 5000);
      return () => clearTimeout(t);
    }
  }, [errorMessage]);

  // Persist session whenever critical state changes
  useEffect(() => {
    persist({
      chatId: currentChatId,
      mode: selectedModel.category,
      subMode: activeSubMode,
      selectedModelId: selectedModel.id,
      killOverride: glassState.killOverride,
    });
  }, [currentChatId, selectedModel, activeSubMode, glassState.killOverride, persist]);

  // Load chat history when sidebar opens
  const loadChatHistory = useCallback(async () => {
    if (!backendOnline) return;
    setHistoryLoading(true);
    try {
      const history = await getChatHistory(30, 0);
      setChatHistory(history);
    } catch (err) {
      console.error("Failed to load chat history:", err);
    } finally {
      setHistoryLoading(false);
    }
  }, [backendOnline]);

  useEffect(() => {
    if (showHistory && backendOnline) {
      loadChatHistory();
    }
  }, [showHistory, backendOnline, loadChatHistory]);

  // Restore a previous chat
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
      setShowHistory(false);
    } catch (err) {
      console.error("Failed to restore chat:", err);
      setErrorMessage("Failed to load chat messages");
    }
  };

  // File upload handler
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setAttachedFile(file);
    }
  };

  const removeFile = () => {
    setAttachedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // Share chat
  const handleShareChat = async () => {
    if (!currentChatId || !backendOnline) return;
    try {
      const result = await shareChat(currentChatId);
      navigator.clipboard?.writeText(result.share_token);
      setErrorMessage(null);
    } catch {
      setErrorMessage("Failed to share chat");
    }
  };

  // Kill switch diagnostic
  const handleKillSwitch = async () => {
    if (!currentChatId || !backendOnline) return;

    // Toggle glass kill override state
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

  const handleSend = async () => {
    if (!input.trim() && !attachedFile) return;

    // Sanitize input — strip control chars, limit length
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

    // Abort any in-flight request to prevent race conditions
    if (abortRef.current) abortRef.current.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    if (backendOnline) {
      try {
        let response: SentinelRunResponse;

        if (selectedModel.id === "sentinel-exp" && activeSubMode) {
          response = await runExperimental(
            userText,
            activeSubMode,
            6,
            currentChatId || undefined,
            glassState.killOverride,
            attachedFile || undefined,
            ac.signal
          );
        } else {
          response = await runStandard(
            userText,
            currentChatId || undefined,
            attachedFile || undefined,
            ac.signal
          );
        }

        // Ignore if this request was aborted (a newer request is in flight)
        if (ac.signal.aborted) return;

        // Store chat_id for session continuity
        if (response.chat_id) {
          setCurrentChatId(response.chat_id);
        }

        // Process response through mode managers
        if (activeSubMode === "debate" && response.omega_metadata) {
          setDebateState((prev) => mergeDebateResult(prev, response.omega_metadata));
        }
        if (activeSubMode === "glass" && response.omega_metadata) {
          setGlassState((prev) => mergeGlassState(prev, response.omega_metadata));
        }
        if (activeSubMode === "evidence" && response.omega_metadata) {
          setEvidenceState((prev) => mergeEvidenceState(prev, response.omega_metadata));
        }

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
      let response: string;
      const currentMode = activeSubMode;
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
    }, 1500);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewChat = () => {
    // Abort any in-flight request
    if (abortRef.current) abortRef.current.abort();

    setCurrentChatId(null);
    setAttachedFile(null);
    setExpandedMeta(null);
    setShowSessionPanel(false);

    // Reset mode managers
    setDebateState(createDebateState(6));
    setGlassState(createGlassState());
    setEvidenceState(createEvidenceState());

    // Clear persisted session
    resetSession();

    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content: "Hello! I'm Sentinel-E, your AI assistant powered by the Omega Cognitive Kernel. How can I help you today?",
        timestamp: new Date(),
      },
    ]);
  };

  const handleFeedback = async (messageId: string, vote: "up" | "down") => {
    const msg = messages.find((m) => m.id === messageId);
    if (!msg?.chatId) return;

    // Optimistic UI update
    setMessages((prev) =>
      prev.map((m) => (m.id === messageId ? { ...m, feedbackGiven: vote } : m))
    );

    if (!backendOnline) return;

    try {
      await submitFeedback({
        run_id: msg.chatId,
        feedback: vote,
        mode: selectedModel.id === "sentinel-exp" ? "experimental" : "standard",
        sub_mode: msg.mode || undefined,
        confidence: msg.confidence,
        boundary_severity: msg.boundaryResult?.severity_score,
        fragility_index: msg.omegaMetadata?.fragility_index,
        disagreement_score: msg.omegaMetadata?.session_state?.disagreement_score,
      });
    } catch {
      // Silent fail for feedback
    }
  };

  // Render confidence bar
  const renderConfidenceBar = (value: number, label: string, color: string) => (
    <div className="flex items-center gap-2">
      <span
        className="text-[#6e6e73] w-20 flex-shrink-0"
        style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 500 }}
      >
        {label}
      </span>
      <div className="flex-1 h-1.5 bg-black/5 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${Math.round(value * 100)}%`, backgroundColor: color }}
        />
      </div>
      <span
        className="text-[#1d1d1f] w-10 text-right flex-shrink-0"
        style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 600 }}
      >
        {Math.round(value * 100)}%
      </span>
    </div>
  );

  // Render Omega metadata panel
  const renderOmegaInsights = (message: Message) => {
    const isExpanded = expandedMeta === message.id;
    const hasData = message.omegaMetadata || message.reasoningTrace || message.confidenceEvolution || message.boundaryResult;
    if (!hasData || message.role !== "assistant") return null;

    return (
      <div className="mt-2">
        <button
          onClick={() => setExpandedMeta(isExpanded ? null : message.id)}
          className="flex items-center gap-1 px-2 py-1 rounded-lg hover:bg-black/5 transition-colors"
        >
          <Brain className="w-3 h-3 text-[#8b5cf6]" />
          <span
            style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 600, color: '#8b5cf6' }}
          >
            Omega Insights
          </span>
          <ChevronRight
            className="w-3 h-3 text-[#8b5cf6] transition-transform"
            style={{ transform: isExpanded ? "rotate(90deg)" : "rotate(0deg)" }}
          />
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
              <div className="mt-2 p-3 rounded-xl bg-[#f5f5f7]/80 space-y-3">
                {/* Confidence Evolution */}
                {message.confidenceEvolution && (
                  <div>
                    <div className="flex items-center gap-1 mb-2">
                      <BarChart3 className="w-3 h-3 text-[#3b82f6]" />
                      <span
                        style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 600, color: '#3b82f6', textTransform: 'uppercase', letterSpacing: '0.05em' }}
                      >
                        Confidence Evolution
                      </span>
                    </div>
                    <div className="space-y-1.5">
                      {renderConfidenceBar(message.confidenceEvolution.initial, "Initial", "#aeaeb2")}
                      {message.confidenceEvolution.post_debate != null &&
                        renderConfidenceBar(message.confidenceEvolution.post_debate, "Post-debate", "#ef4444")}
                      {message.confidenceEvolution.post_boundary != null &&
                        renderConfidenceBar(message.confidenceEvolution.post_boundary, "Post-bound.", "#f59e0b")}
                      {message.confidenceEvolution.post_evidence != null &&
                        renderConfidenceBar(message.confidenceEvolution.post_evidence, "Post-evid.", "#06b6d4")}
                      {message.confidenceEvolution.post_stress != null &&
                        renderConfidenceBar(message.confidenceEvolution.post_stress, "Post-stress", "#8b5cf6")}
                      {renderConfidenceBar(message.confidenceEvolution.final, "Final", "#10b981")}
                    </div>
                  </div>
                )}

                {/* Reasoning Trace */}
                {message.reasoningTrace && (
                  <div>
                    <div className="flex items-center gap-1 mb-2">
                      <Activity className="w-3 h-3 text-[#f59e0b]" />
                      <span
                        style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 600, color: '#f59e0b', textTransform: 'uppercase', letterSpacing: '0.05em' }}
                      >
                        Reasoning Trace
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                      {[
                        { label: "Passes", value: message.reasoningTrace.passes_executed },
                        { label: "Assumptions", value: message.reasoningTrace.assumptions_extracted },
                        { label: "Logic gaps", value: message.reasoningTrace.logical_gaps_detected },
                        { label: "Boundary sev.", value: message.reasoningTrace.boundary_severity },
                      ].map((item) => (
                        <div key={item.label} className="flex items-center justify-between">
                          <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 400, color: '#6e6e73' }}>{item.label}</span>
                          <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 600, color: '#1d1d1f' }}>{item.value}</span>
                        </div>
                      ))}
                    </div>
                    <div className="flex gap-2 mt-1.5">
                      {message.reasoningTrace.self_critique_applied && (
                        <span className="px-1.5 py-0.5 rounded-md bg-[#f59e0b]/10 text-[#f59e0b]" style={{ fontSize: '9px', fontWeight: 600 }}>
                          Self-critique
                        </span>
                      )}
                      {message.reasoningTrace.refinement_applied && (
                        <span className="px-1.5 py-0.5 rounded-md bg-[#10b981]/10 text-[#10b981]" style={{ fontSize: '9px', fontWeight: 600 }}>
                          Refined
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {/* Boundary Result */}
                {message.boundaryResult && message.boundaryResult.severity_score > 0 && (
                  <div>
                    <div className="flex items-center gap-1 mb-2">
                      <Shield className="w-3 h-3 text-[#ef4444]" />
                      <span
                        style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 600, color: '#ef4444', textTransform: 'uppercase', letterSpacing: '0.05em' }}
                      >
                        Boundary Evaluation
                      </span>
                    </div>
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', color: '#6e6e73' }}>Risk Level</span>
                        <span
                          className="px-1.5 py-0.5 rounded-md"
                          style={{
                            fontSize: '9px',
                            fontWeight: 600,
                            backgroundColor: message.boundaryResult.risk_level === "HIGH" ? '#fef2f2' : message.boundaryResult.risk_level === "MEDIUM" ? '#fffbeb' : '#f0fdf4',
                            color: message.boundaryResult.risk_level === "HIGH" ? '#ef4444' : message.boundaryResult.risk_level === "MEDIUM" ? '#f59e0b' : '#10b981',
                          }}
                        >
                          {message.boundaryResult.risk_level}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', color: '#6e6e73' }}>Severity</span>
                        <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 600, color: '#1d1d1f' }}>{message.boundaryResult.severity_score}/100</span>
                      </div>
                      {message.boundaryResult.explanation && (
                        <p style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', color: '#6e6e73', lineHeight: 1.4 }}>
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
                            renderConfidenceBar(val as number / 100, dim, '#ef4444')
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Fragility Index & Behavioral Risk (from omegaMetadata) */}
                {message.omegaMetadata?.fragility_index != null && message.omegaMetadata.fragility_index > 0 && (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1">
                      <Zap className="w-3 h-3 text-[#f59e0b]" />
                      <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 500, color: '#6e6e73' }}>Fragility Index</span>
                    </div>
                    <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 600, color: '#1d1d1f' }}>
                      {(message.omegaMetadata.fragility_index * 100).toFixed(1)}%
                    </span>
                  </div>
                )}

                {message.omegaMetadata?.behavioral_risk && (
                  <div>
                    <div className="flex items-center gap-1 mb-1">
                      <Activity className="w-3 h-3 text-[#8b5cf6]" />
                      <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 600, color: '#8b5cf6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Behavioral Risk
                      </span>
                      <span
                        className="ml-auto px-1.5 py-0.5 rounded-md"
                        style={{
                          fontSize: '9px',
                          fontWeight: 600,
                          backgroundColor: message.omegaMetadata.behavioral_risk.risk_level === "HIGH" ? '#fef2f2' : '#f0fdf4',
                          color: message.omegaMetadata.behavioral_risk.risk_level === "HIGH" ? '#ef4444' : '#10b981',
                        }}
                      >
                        {message.omegaMetadata.behavioral_risk.risk_level}
                      </span>
                    </div>
                    <div className="space-y-1">
                      {renderConfidenceBar(message.omegaMetadata.behavioral_risk.manipulation_probability, "Manipulation", "#ef4444")}
                      {renderConfidenceBar(message.omegaMetadata.behavioral_risk.evasion_index, "Evasion", "#f59e0b")}
                      {renderConfidenceBar(message.omegaMetadata.behavioral_risk.confidence_inflation, "Inflation", "#8b5cf6")}
                    </div>
                  </div>
                )}

                {/* Evidence Sources */}
                {message.omegaMetadata?.evidence_result && message.omegaMetadata.evidence_result.sources.length > 0 && (
                  <div>
                    <div className="flex items-center gap-1 mb-1">
                      <FileSearch className="w-3 h-3 text-[#06b6d4]" />
                      <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 600, color: '#06b6d4', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Evidence Sources ({message.omegaMetadata.evidence_result.source_count})
                      </span>
                    </div>
                    <div className="space-y-1">
                      {message.omegaMetadata.evidence_result.sources.slice(0, 5).map((src, i) => (
                        <div key={i} className="flex items-start gap-1.5">
                          <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '9px', fontWeight: 600, color: '#06b6d4' }}>[{i + 1}]</span>
                          <div>
                            <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 500, color: '#1d1d1f' }}>{src.title || src.domain}</span>
                            {src.url && (
                              <a href={src.url} target="_blank" rel="noopener noreferrer" className="block text-[#3b82f6] hover:underline" style={{ fontSize: '9px' }}>
                                {src.url.length > 50 ? src.url.slice(0, 50) + "..." : src.url}
                              </a>
                            )}
                            <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '9px', color: '#6e6e73' }}>
                              Reliability: {Math.round(src.reliability_score * 100)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Omega version tag */}
                {message.omegaMetadata?.omega_version && (
                  <div className="flex items-center justify-between pt-1 border-t border-black/5">
                    <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '9px', color: '#aeaeb2' }}>
                      Omega Kernel v{message.omegaMetadata.omega_version}
                    </span>
                    {message.omegaMetadata.session_state?.inferred_domain && (
                      <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '9px', color: '#aeaeb2' }}>
                        Domain: {message.omegaMetadata.session_state.inferred_domain}
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

  return (
    <div className="h-screen flex bg-[#f5f5f7] pt-14">
      {/* Chat History Sidebar */}
      <AnimatePresence>
        {showHistory && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 280, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
            className="h-full border-r border-black/5 bg-white/80 backdrop-blur-xl overflow-hidden flex-shrink-0"
          >
            <div className="h-full flex flex-col">
              <div className="flex items-center justify-between px-4 py-3 border-b border-black/5">
                <span
                  className="text-[#1d1d1f]"
                  style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '14px', fontWeight: 600 }}
                >
                  Chat History
                </span>
                <button
                  onClick={() => setShowHistory(false)}
                  className="p-1 rounded-lg hover:bg-black/5 transition-colors"
                >
                  <X className="w-4 h-4 text-[#6e6e73]" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto">
                {!backendOnline ? (
                  <div className="px-4 py-8 text-center">
                    <WifiOff className="w-8 h-8 text-[#aeaeb2] mx-auto mb-2" />
                    <p style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '12px', color: '#aeaeb2' }}>
                      Connect to backend to see history
                    </p>
                  </div>
                ) : historyLoading ? (
                  <div className="px-4 py-8 text-center">
                    <Loader2 className="w-6 h-6 text-[#aeaeb2] mx-auto mb-2 animate-spin" />
                    <p style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '12px', color: '#aeaeb2' }}>Loading...</p>
                  </div>
                ) : chatHistory.length === 0 ? (
                  <div className="px-4 py-8 text-center">
                    <MessageSquare className="w-8 h-8 text-[#aeaeb2] mx-auto mb-2" />
                    <p style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '12px', color: '#aeaeb2' }}>
                      No previous chats
                    </p>
                  </div>
                ) : (
                  <div className="p-2 space-y-0.5">
                    {chatHistory.map((chat) => (
                      <button
                        key={chat.id}
                        onClick={() => restoreChat(chat)}
                        className={`w-full text-left px-3 py-2.5 rounded-xl transition-colors ${
                          currentChatId === chat.id ? "bg-[#e8e8ed]" : "hover:bg-[#f5f5f7]"
                        }`}
                      >
                        <div
                          className="text-[#1d1d1f] truncate"
                          style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '13px', fontWeight: 500 }}
                        >
                          {chat.name || "Untitled Chat"}
                        </div>
                        <div className="flex items-center gap-2 mt-0.5">
                          <span
                            className="px-1.5 py-0.5 rounded-md"
                            style={{
                              fontFamily: "'Inter', -apple-system, sans-serif",
                              fontSize: '9px',
                              fontWeight: 600,
                              backgroundColor: chat.mode === "experimental" ? '#f3e8ff' : '#e0f2fe',
                              color: chat.mode === "experimental" ? '#8b5cf6' : '#3b82f6',
                            }}
                          >
                            {chat.mode || "standard"}
                          </span>
                          <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', color: '#aeaeb2' }}>
                            {new Date(chat.updated_at || chat.created_at).toLocaleDateString()}
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

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-white/80 backdrop-blur-xl border-b border-black/5">
          <div className="flex items-center gap-2">
            {/* History toggle */}
            <button
              onClick={() => setShowHistory(!showHistory)}
              className={`p-2 rounded-xl transition-colors ${showHistory ? "bg-[#e8e8ed]" : "hover:bg-black/5"}`}
              title="Chat History"
            >
              <History className="w-4.5 h-4.5 text-[#6e6e73]" />
            </button>

            {/* Model picker */}
            <button
              onClick={() => setShowModelPicker(!showModelPicker)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-xl hover:bg-black/5 transition-colors"
            >
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: selectedModel.color }}
              />
              <span
                className="text-[#1d1d1f]"
                style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '15px', fontWeight: 600 }}
              >
                {selectedModel.name}
              </span>
              <ChevronDown className="w-3.5 h-3.5 text-[#6e6e73]" />
            </button>

            {/* Connection Status */}
            <div className="flex items-center gap-1.5">
              {backendOnline === null ? (
                <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-[#f5f5f7]">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#aeaeb2] animate-pulse" />
                  <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '11px', fontWeight: 500, color: '#aeaeb2' }}>
                    Connecting...
                  </span>
                </div>
              ) : backendOnline ? (
                <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-[#d1fae5]/60">
                  <Wifi className="w-3 h-3 text-[#10b981]" />
                  <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '11px', fontWeight: 500, color: '#10b981' }}>
                    Live {healthData?.version ? `v${healthData.version}` : ""}
                  </span>
                </div>
              ) : (
                <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-[#fef3c7]/60">
                  <WifiOff className="w-3 h-3 text-[#f59e0b]" />
                  <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '11px', fontWeight: 500, color: '#f59e0b' }}>
                    Offline
                  </span>
                </div>
              )}
            </div>

            {/* Kernel status badge */}
            {kernelData && kernelData.status === "online" && (
              <div className="hidden sm:flex items-center gap-1 px-2 py-1 rounded-lg bg-[#f3e8ff]/60">
                <Brain className="w-3 h-3 text-[#8b5cf6]" />
                <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 500, color: '#8b5cf6' }}>
                  {kernelData.active_sessions} sessions
                </span>
              </div>
            )}
          </div>

          <div className="flex items-center gap-1">
            {/* Session Analytics toggle */}
            {currentChatId && backendOnline && (
              <button
                onClick={() => setShowSessionPanel(!showSessionPanel)}
                className={`p-2 rounded-xl transition-colors ${showSessionPanel ? "bg-[#f3e8ff]" : "hover:bg-black/5"}`}
                title="Session Analytics"
              >
                <PanelRightOpen className={`w-4 h-4 ${showSessionPanel ? "text-[#8b5cf6]" : "text-[#6e6e73]"}`} />
              </button>
            )}

            {/* Share button */}
            {currentChatId && backendOnline && (
              <button
                onClick={handleShareChat}
                className="p-2 rounded-xl hover:bg-black/5 transition-colors"
                title="Share Chat"
              >
                <Share2 className="w-4 h-4 text-[#6e6e73]" />
              </button>
            )}

            {/* Kill switch (only when in Pro mode with active session) */}
            {selectedModel.id === "sentinel-exp" && currentChatId && backendOnline && (
              <button
                onClick={handleKillSwitch}
                className="p-2 rounded-xl hover:bg-[#fef2f2] transition-colors"
                title="Kill Diagnostic — Session cognitive state snapshot"
              >
                <Skull className="w-4 h-4 text-[#ef4444]" />
              </button>
            )}

            <button
              onClick={handleNewChat}
              className="p-2 rounded-xl hover:bg-black/5 transition-colors"
              title="New Chat"
            >
              <Plus className="w-5 h-5 text-[#6e6e73]" />
            </button>
          </div>
        </div>

        {/* Backend Offline Banner */}
        <AnimatePresence>
          {backendOnline === false && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="bg-[#fffbeb] border-b border-[#f59e0b]/20 px-4 py-2 flex items-center gap-2"
            >
              <AlertCircle className="w-3.5 h-3.5 text-[#f59e0b] flex-shrink-0" />
              <span
                style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '12px', fontWeight: 500, color: '#92400e' }}
              >
                Backend offline — using local mock responses. Start your FastAPI server at localhost:8000 for live AI.
              </span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error Banner */}
        <AnimatePresence>
          {errorMessage && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="bg-[#fef2f2] border-b border-[#ef4444]/20 px-4 py-2 flex items-center justify-between"
            >
              <div className="flex items-center gap-2">
                <AlertCircle className="w-3.5 h-3.5 text-[#ef4444] flex-shrink-0" />
                <span
                  style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '12px', fontWeight: 500, color: '#991b1b' }}
                >
                  {errorMessage}
                </span>
              </div>
              <button onClick={() => setErrorMessage(null)}>
                <X className="w-3.5 h-3.5 text-[#991b1b]" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Model Picker Dropdown */}
        <AnimatePresence>
          {showModelPicker && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setShowModelPicker(false)} />
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute top-[7.5rem] left-16 z-50 w-64 p-2 rounded-2xl bg-white/95 backdrop-blur-xl shadow-2xl shadow-black/10 border border-black/5"
              >
                {/* Standard Section */}
                <div
                  className="px-3 pt-2 pb-1 text-[#6e6e73]"
                  style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '11px', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' as const }}
                >
                  Standard
                </div>
                {models.filter(m => m.category === "standard").map((model) => (
                  <button
                    key={model.id}
                    onClick={() => {
                      setSelectedModel(model);
                      setShowModelPicker(false);
                      setActiveSubMode(null);
                    }}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors ${
                      selectedModel.id === model.id ? "bg-[#f5f5f7]" : "hover:bg-[#f5f5f7]"
                    }`}
                  >
                    <div
                      className="w-8 h-8 rounded-xl flex items-center justify-center"
                      style={{ backgroundColor: model.color + "20" }}
                    >
                      <Sparkles className="w-4 h-4" style={{ color: model.color }} />
                    </div>
                    <div className="text-left">
                      <div
                        className="text-[#1d1d1f]"
                        style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '14px', fontWeight: 600 }}
                      >
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

                {/* Divider */}
                <div className="my-1.5 mx-3 border-t border-black/5" />

                {/* Experimental Section */}
                <div
                  className="px-3 pt-2 pb-1 text-[#6e6e73]"
                  style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '11px', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' as const }}
                >
                  Experimental
                </div>
                {models.filter(m => m.category === "experimental").map((model) => (
                  <button
                    key={model.id}
                    onClick={() => {
                      setSelectedModel(model);
                      setShowModelPicker(false);
                      if (model.id !== "sentinel-exp") setActiveSubMode(null);
                    }}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors ${
                      selectedModel.id === model.id ? "bg-[#f5f5f7]" : "hover:bg-[#f5f5f7]"
                    }`}
                  >
                    <div
                      className="w-8 h-8 rounded-xl flex items-center justify-center"
                      style={{ backgroundColor: model.color + "20" }}
                    >
                      <Sparkles className="w-4 h-4" style={{ color: model.color }} />
                    </div>
                    <div className="text-left">
                      <div
                        className="text-[#1d1d1f]"
                        style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '14px', fontWeight: 600 }}
                      >
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

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-3xl mx-auto space-y-4">
            <AnimatePresence>
              {messages.map((message) => {
                const msgMode = message.mode ? proSubModes.find(m => m.id === message.mode) : null;
                return (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[85%] sm:max-w-[70%] ${
                        message.role === "user"
                          ? "rounded-[20px] rounded-br-md bg-[#007aff] text-white px-4 py-3"
                          : "rounded-[20px] rounded-bl-md bg-white border border-black/5 text-[#1d1d1f] shadow-sm overflow-hidden"
                      }`}
                      style={
                        message.role === "assistant" && msgMode
                          ? { borderLeft: `3px solid ${msgMode.color}` }
                          : message.role === "assistant" && message.mode === "kill"
                            ? { borderLeft: "3px solid #ef4444" }
                            : undefined
                      }
                    >
                      {/* Mode badge for assistant messages */}
                      {message.role === "assistant" && msgMode && (
                        <div
                          className="flex items-center gap-1.5 px-4 py-1.5"
                          style={{ backgroundColor: msgMode.color + '0a' }}
                        >
                          <div
                            className="flex items-center justify-center w-4 h-4"
                            style={{ color: msgMode.color }}
                          >
                            {msgMode.icon}
                          </div>
                          <span
                            style={{
                              fontFamily: "'Inter', -apple-system, sans-serif",
                              fontSize: '11px',
                              fontWeight: 600,
                              color: msgMode.color,
                              letterSpacing: '0.03em',
                              textTransform: 'uppercase' as const,
                            }}
                          >
                            {msgMode.label}
                          </span>
                          {message.confidence !== undefined && (
                            <span
                              className="ml-auto"
                              style={{
                                fontFamily: "'Inter', -apple-system, sans-serif",
                                fontSize: '10px',
                                fontWeight: 500,
                                color: '#6e6e73',
                              }}
                            >
                              {Math.round(message.confidence * 100)}% confidence
                            </span>
                          )}
                        </div>
                      )}

                      {/* Kill mode header */}
                      {message.role === "assistant" && message.mode === "kill" && (
                        <div className="flex items-center gap-1.5 px-4 py-1.5 bg-[#fef2f2]">
                          <Skull className="w-3.5 h-3.5 text-[#ef4444]" />
                          <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '11px', fontWeight: 600, color: '#ef4444', letterSpacing: '0.03em', textTransform: 'uppercase' as const }}>
                            Kill Diagnostic
                          </span>
                        </div>
                      )}

                      <div className={message.role === "assistant" ? "px-4 py-3" : ""}>
                        <p
                          className="whitespace-pre-wrap"
                          style={{
                            fontFamily: "'Inter', -apple-system, sans-serif",
                            fontSize: '15px',
                            lineHeight: 1.5,
                            fontWeight: 400,
                          }}
                        >
                          {message.content}
                        </p>

                        {/* Boundary warning for high-risk responses */}
                        {message.role === "assistant" &&
                          message.boundaryResult &&
                          message.boundaryResult.severity_score > 40 && (
                            <div
                              className="flex items-center gap-1.5 mt-2 px-2 py-1 rounded-lg"
                              style={{ backgroundColor: '#fef3c7' }}
                            >
                              <AlertCircle className="w-3 h-3 text-[#f59e0b]" />
                              <span
                                style={{
                                  fontFamily: "'Inter', -apple-system, sans-serif",
                                  fontSize: '11px',
                                  fontWeight: 500,
                                  color: '#92400e',
                                }}
                              >
                                Boundary: {message.boundaryResult.risk_level} (severity {message.boundaryResult.severity_score})
                              </span>
                            </div>
                          )}

                        {/* Omega Insights toggle */}
                        {renderOmegaInsights(message)}

                        {/* Cross-Analysis trigger for Glass Mode responses */}
                        {message.role === "assistant" && message.mode === "glass" && message.id !== "welcome" && backendOnline && (
                          <CrossAnalysisTrigger
                            chatId={currentChatId}
                            messageContent={message.content}
                            backendOnline={backendOnline}
                          />
                        )}

                        <div className="flex items-center justify-between mt-1">
                          <div
                            className={`${message.role === "user" ? "text-white/50" : "text-[#6e6e73]"}`}
                            style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '11px', fontWeight: 400 }}
                          >
                            {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                          </div>

                          {/* Feedback buttons */}
                          {message.role === "assistant" && message.id !== "welcome" && (
                            <div className="flex items-center gap-1 ml-2">
                              <button
                                onClick={() => handleFeedback(message.id, "up")}
                                className={`p-1 rounded-md transition-colors ${
                                  message.feedbackGiven === "up"
                                    ? "bg-[#d1fae5]"
                                    : "hover:bg-black/5"
                                }`}
                                title="Good response"
                                disabled={!!message.feedbackGiven}
                              >
                                <ThumbsUp className={`w-3 h-3 ${message.feedbackGiven === "up" ? "text-[#10b981]" : "text-[#aeaeb2]"}`} />
                              </button>
                              <button
                                onClick={() => handleFeedback(message.id, "down")}
                                className={`p-1 rounded-md transition-colors ${
                                  message.feedbackGiven === "down"
                                    ? "bg-[#fef2f2]"
                                    : "hover:bg-black/5"
                                }`}
                                title="Bad response"
                                disabled={!!message.feedbackGiven}
                              >
                                <ThumbsDown className={`w-3 h-3 ${message.feedbackGiven === "down" ? "text-[#ef4444]" : "text-[#aeaeb2]"}`} />
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

            {isTyping && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-start"
              >
                <div
                  className="px-5 py-3.5 rounded-[20px] rounded-bl-md bg-white border shadow-sm overflow-hidden"
                  style={{
                    borderColor: activeSubMode
                      ? proSubModes.find(m => m.id === activeSubMode)!.color + '30'
                      : 'rgba(0,0,0,0.05)',
                    borderLeftWidth: activeSubMode ? '3px' : '1px',
                    borderLeftColor: activeSubMode
                      ? proSubModes.find(m => m.id === activeSubMode)!.color
                      : 'rgba(0,0,0,0.05)',
                  }}
                >
                  {activeSubMode && (
                    <div
                      className="flex items-center gap-1 mb-2"
                      style={{ color: proSubModes.find(m => m.id === activeSubMode)!.color }}
                    >
                      {proSubModes.find(m => m.id === activeSubMode)!.icon}
                      <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', fontWeight: 600, textTransform: 'uppercase' as const, letterSpacing: '0.05em' }}>
                        {backendOnline ? "Processing via Omega Kernel..." : "Thinking..."}
                      </span>
                    </div>
                  )}
                  <div className="flex gap-1.5">
                    {[0, 150, 300].map((delay) => (
                      <div
                        key={delay}
                        className="w-2 h-2 rounded-full animate-bounce"
                        style={{
                          backgroundColor: activeSubMode
                            ? proSubModes.find(m => m.id === activeSubMode)!.color
                            : '#6e6e73',
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

        {/* Input Area */}
        <div className="px-4 pb-6 pt-2">
          <div className="max-w-3xl mx-auto">
            <div
              className="flex flex-col gap-0 p-2 rounded-[28px] bg-white shadow-lg transition-all duration-300"
              style={{
                borderWidth: '1px',
                borderStyle: 'solid',
                borderColor: activeSubMode
                  ? proSubModes.find(m => m.id === activeSubMode)!.color + '40'
                  : 'rgba(0,0,0,0.1)',
                boxShadow: activeSubMode
                  ? `0 4px 24px -4px ${proSubModes.find(m => m.id === activeSubMode)!.color}20, 0 0 0 1px ${proSubModes.find(m => m.id === activeSubMode)!.color}15`
                  : '0 4px 6px -1px rgba(0,0,0,0.05)',
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
                    <div className="flex items-center gap-2 px-3 py-2 mx-1 mt-1 rounded-xl bg-[#f5f5f7]">
                      <Paperclip className="w-3.5 h-3.5 text-[#6e6e73]" />
                      <span
                        className="flex-1 truncate text-[#1d1d1f]"
                        style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '12px', fontWeight: 500 }}
                      >
                        {attachedFile.name}
                      </span>
                      <span style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '10px', color: '#aeaeb2' }}>
                        {(attachedFile.size / 1024).toFixed(1)} KB
                      </span>
                      <button onClick={removeFile} className="p-0.5 rounded-md hover:bg-black/10 transition-colors">
                        <X className="w-3 h-3 text-[#6e6e73]" />
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Sentinel-E Pro Submodes */}
              <AnimatePresence>
                {selectedModel.id === "sentinel-exp" && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.25, ease: "easeOut" }}
                    className="overflow-hidden"
                  >
                    <div className="flex flex-col gap-2 px-1 pb-2 pt-1">
                      <div className="flex items-center gap-1.5">
                        {proSubModes.map((mode) => {
                          const isActive = activeSubMode === mode.id;
                          return (
                            <button
                              key={mode.id}
                              onClick={() => setActiveSubMode(isActive ? null : mode.id)}
                              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full transition-all duration-200 ${
                                isActive
                                  ? "text-white shadow-md"
                                  : "bg-[#f5f5f7] text-[#6e6e73] hover:bg-[#e8e8ed]"
                              }`}
                              style={isActive ? {
                                backgroundColor: mode.color,
                                boxShadow: `0 2px 8px ${mode.color}40`,
                              } : undefined}
                            >
                              {mode.icon}
                              <span
                                style={{
                                  fontFamily: "'Inter', -apple-system, sans-serif",
                                  fontSize: '12px',
                                  fontWeight: 600,
                                }}
                              >
                                {mode.label}
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
                            style={{ backgroundColor: proSubModes.find(m => m.id === activeSubMode)!.color + '08' }}
                          >
                            <div
                              className="w-1 h-4 rounded-full"
                              style={{ backgroundColor: proSubModes.find(m => m.id === activeSubMode)!.color }}
                            />
                            <span
                              style={{
                                fontFamily: "'Inter', -apple-system, sans-serif",
                                fontSize: '12px',
                                fontWeight: 400,
                                color: '#6e6e73',
                              }}
                            >
                              {proSubModes.find(m => m.id === activeSubMode)!.description}
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
                <button className="p-2 rounded-full hover:bg-black/5 transition-colors flex-shrink-0">
                  <Plus className="w-5 h-5 text-[#6e6e73]" />
                </button>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className={`p-2 rounded-full transition-colors flex-shrink-0 ${attachedFile ? 'bg-[#e0f2fe]' : 'hover:bg-black/5'}`}
                  title="Attach file"
                >
                  <Paperclip className={`w-5 h-5 ${attachedFile ? 'text-[#3b82f6]' : 'text-[#6e6e73]'}`} />
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  onChange={handleFileSelect}
                  accept=".txt,.pdf,.md,.json,.csv,.py,.js,.ts,.jsx,.tsx"
                />
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={
                    activeSubMode
                      ? proSubModes.find(m => m.id === activeSubMode)!.placeholder
                      : "Message Sentinel-E..."
                  }
                  rows={1}
                  className="flex-1 resize-none bg-transparent outline-none py-2 px-1 max-h-32 text-[#1d1d1f] placeholder-[#aeaeb2]"
                  style={{
                    fontFamily: "'Inter', -apple-system, sans-serif",
                    fontSize: '16px',
                    lineHeight: 1.5,
                    fontWeight: 400,
                  }}
                />
                <button className="p-2 rounded-full hover:bg-black/5 transition-colors flex-shrink-0">
                  <Mic className="w-5 h-5 text-[#6e6e73]" />
                </button>
                <button
                  onClick={handleSend}
                  disabled={(!input.trim() && !attachedFile) || isTyping}
                  className="p-2 rounded-full flex-shrink-0 transition-all"
                  style={{
                    backgroundColor: (!input.trim() && !attachedFile) || isTyping
                      ? '#e5e5ea'
                      : activeSubMode
                        ? proSubModes.find(m => m.id === activeSubMode)!.color
                        : '#007aff',
                    color: (input.trim() || attachedFile) && !isTyping ? 'white' : '#aeaeb2',
                    boxShadow: (input.trim() || attachedFile) && !isTyping && activeSubMode
                      ? `0 4px 12px ${proSubModes.find(m => m.id === activeSubMode)!.color}40`
                      : (input.trim() || attachedFile) && !isTyping
                        ? '0 4px 12px rgba(0,122,255,0.3)'
                        : 'none',
                  }}
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
            </div>
            <p
              className="text-center mt-2 text-[#aeaeb2]"
              style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '11px', fontWeight: 400 }}
            >
              {backendOnline
                ? `Connected to Sentinel-E Omega Cognitive Kernel v${healthData?.version || "4.5"}${currentChatId ? ` \u2022 Session: ${currentChatId.slice(0, 8)}...` : ""}`
                : "Sentinel-E can make mistakes. Consider checking important information."}
            </p>
          </div>
        </div>
      </div>

      {/* Session Analytics Right Sidebar */}
      <AnimatePresence>
        {showSessionPanel && (
          <SessionAnalyticsPanel
            chatId={currentChatId}
            backendOnline={backendOnline}
            onClose={() => setShowSessionPanel(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
