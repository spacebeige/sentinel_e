import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send, Plus, Brain, Shield, Activity, Cpu, Swords,
  ChevronDown, Settings, MoreHorizontal, GitBranch,
  Zap, Eye, Network, BarChart3, Clock, CheckCircle2,
  AlertTriangle, PanelRightClose, PanelRightOpen,
} from "lucide-react";
import { GlassPanel } from "./GlassPanel";
import { ConsensusMeter } from "./ConsensusMeter";

// ─── Types ──────────────────────────────────────────────────────────────────
interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  model?: string;
  phase?: string;
  confidence?: number;
}

const SESSIONS = [
  { id: "s1", title: "Causal inference in deep networks", time: "2h ago",  active: true  },
  { id: "s2", title: "Climate policy multi-model debate",  time: "5h ago",  active: false },
  { id: "s3", title: "Adversarial stress test #128",       time: "1d ago",  active: false },
  { id: "s4", title: "Governance boundary analysis",       time: "2d ago",  active: false },
  { id: "s5", title: "Semantic routing optimisation",      time: "3d ago",  active: false },
];

const MODES = [
  { id: "deliberation", label: "Deliberation",  icon: Brain,    color: "#6ee7f9" },
  { id: "debate",       label: "Debate",        icon: Swords,   color: "#f59e0b" },
  { id: "evidence",     label: "Evidence",      icon: Eye,      color: "#34d399" },
  { id: "governance",   label: "Governance",    icon: Shield,   color: "#8b5cf6" },
];

const MODEL_AGENTS = [
  { id: "gpt5o",    name: "GPT-5o",          role: "Synthesis",   color: "#6ee7f9", load: 82 },
  { id: "claude4",  name: "Claude Opus 4",   role: "Critique",    color: "#8b5cf6", load: 67 },
  { id: "gemini3",  name: "Gemini 3 Flash",  role: "Evidence",    color: "#34d399", load: 91 },
  { id: "mistral",  name: "Mistral Large",   role: "Governance",  color: "#f59e0b", load: 54 },
];

const WELCOME_MSG: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "Sentinel-E Deliberation Runtime initialised. The AI Council is standing by.\n\nPresent a question, hypothesis, or directive — the ensemble will deliberate, surface reasoning chains, and converge on a verified consensus position.\n\nAll cognition is visible. All reasoning is traceable.",
  timestamp: new Date(),
  model: "Ensemble",
  phase: "Initialisation",
  confidence: 100,
};

const DEMO_REPLIES: Message[] = [
  {
    id: "demo-1",
    role: "assistant",
    content:
      "**Synthesis phase complete** — Council position:\n\nAcross 4 parallel reasoning paths, consensus converged at **91.4%** after 3 deliberation rounds.\n\n**GPT-5o** (Synthesis): The primary mechanism aligns with established causal inference literature — the confounding variables are accountable under the proposed framework.\n\n**Claude Opus 4** (Critique): Edge-case resistance is lower than optimal. Flagging boundary condition at distribution shift δ > 0.4.\n\n**Gemini 3** (Evidence): Source validation complete — 7 of 8 cited references independently verifiable. Confidence elevated.\n\n**Mistral Large** (Governance): No hallucination markers detected. Semantic integrity: ✓ Passed.\n\n---\n**Council verdict:** Proceed with high confidence. Residual uncertainty at 8.6% centres on out-of-distribution generalisation.",
    timestamp: new Date(),
    model: "Ensemble · 4 Models",
    phase: "Consensus",
    confidence: 91,
  },
];

// ─── Component ───────────────────────────────────────────────────────────────
export function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([WELCOME_MSG]);
  const [input, setInput] = useState("");
  const [activeMode, setActiveMode] = useState("deliberation");
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [demoSent, setDemoSent] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = () => {
    const text = input.trim();
    if (!text) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      timestamp: new Date(),
    };
    setMessages((p) => [...p, userMsg]);
    setInput("");
    setIsThinking(true);

    setTimeout(() => {
      const reply = demoSent
        ? {
            id: (Date.now() + 1).toString(),
            role: "assistant" as const,
            content:
              "Deliberation cycle initiated. Multi-agent ensemble processing your directive...\n\nConsensus formation in progress — semantic confidence building across 4 active cognition streams.",
            timestamp: new Date(),
            model: "Ensemble · 4 Models",
            phase: "Processing",
            confidence: 78,
          }
        : DEMO_REPLIES[0];

      setMessages((p) => [...p, reply]);
      setIsThinking(false);
      setDemoSent(true);
    }, 2200);
  };

  const currentMode = MODES.find((m) => m.id === activeMode) ?? MODES[0];

  return (
    <div className="flex h-screen pt-[68px] bg-[#060708] overflow-hidden">

      {/* ── LEFT SIDEBAR ── */}
      <aside className="w-60 shrink-0 flex flex-col border-r border-[rgba(110,231,249,0.07)] bg-[rgba(6,7,8,0.8)]">
        <div className="p-4 border-b border-[rgba(110,231,249,0.07)]">
          <button
            onClick={() => setMessages([WELCOME_MSG])}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg border border-[rgba(110,231,249,0.15)] bg-[rgba(110,231,249,0.05)] text-[#6ee7f9] text-xs font-medium hover:bg-[rgba(110,231,249,0.1)] transition-colors"
          >
            <Plus className="w-3.5 h-3.5" /> New Session
          </button>
        </div>

        {/* Sessions */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          <div className="px-2 mb-2">
            <span className="text-[#8a9099] text-[9px] tracking-[0.2em] uppercase font-medium">Sessions</span>
          </div>
          {SESSIONS.map((s) => (
            <div
              key={s.id}
              className={`px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                s.active
                  ? "bg-[rgba(110,231,249,0.08)] border border-[rgba(110,231,249,0.12)]"
                  : "hover:bg-[rgba(255,255,255,0.03)]"
              }`}
            >
              <p className={`text-xs leading-snug mb-0.5 truncate ${s.active ? "text-[#f3f5f7]" : "text-[#8a9099]"}`}>
                {s.title}
              </p>
              <p className="text-[10px] text-[#8a9099]">{s.time}</p>
            </div>
          ))}

          <div className="px-2 mt-4 mb-2">
            <span className="text-[#8a9099] text-[9px] tracking-[0.2em] uppercase font-medium">Orchestration Mode</span>
          </div>
          {MODES.map(({ id, label, icon: Icon, color }) => (
            <button
              key={id}
              onClick={() => setActiveMode(id)}
              className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs transition-colors ${
                activeMode === id
                  ? "bg-[rgba(255,255,255,0.05)] border border-[rgba(255,255,255,0.08)]"
                  : "hover:bg-[rgba(255,255,255,0.03)] text-[#8a9099]"
              }`}
              style={{ color: activeMode === id ? color : undefined }}
            >
              <Icon className="w-3 h-3" /> {label}
            </button>
          ))}
        </div>
      </aside>

      {/* ── CENTRE: COGNITION AREA ── */}
      <main className="flex-1 flex flex-col min-w-0 relative">
        {/* Top bar */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-[rgba(110,231,249,0.07)] bg-[rgba(6,7,8,0.7)] backdrop-blur-md">
          <div className="flex items-center gap-3">
            <div
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-semibold tracking-widest uppercase"
              style={{ background: `${currentMode.color}12`, border: `1px solid ${currentMode.color}25`, color: currentMode.color }}
            >
              <currentMode.icon className="w-2.5 h-2.5" />
              {currentMode.label} Mode
            </div>
            <span className="text-[#8a9099] text-xs">Causal inference in deep networks</span>
          </div>
          <div className="flex items-center gap-2">
            <motion.div
              className="w-1.5 h-1.5 rounded-full bg-[#34d399]"
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1.6, repeat: Infinity }}
            />
            <span className="text-[#34d399] text-[10px] font-medium tracking-widest uppercase">Deliberating</span>
            <button
              onClick={() => setRightPanelOpen(!rightPanelOpen)}
              className="ml-2 p-1.5 rounded-md text-[#8a9099] hover:text-[#f3f5f7] hover:bg-[rgba(255,255,255,0.05)] transition-colors"
            >
              {rightPanelOpen ? <PanelRightClose className="w-4 h-4" /> : <PanelRightOpen className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5">
          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35 }}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {msg.role === "assistant" ? (
                  <div className="max-w-2xl w-full">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-5 h-5 rounded flex items-center justify-center bg-[rgba(110,231,249,0.1)] border border-[rgba(110,231,249,0.2)]">
                        <Brain className="w-2.5 h-2.5 text-[#6ee7f9]" />
                      </div>
                      <span className="text-[#6ee7f9] text-[10px] font-semibold tracking-wide">{msg.model ?? "Sentinel-E"}</span>
                      {msg.phase && (
                        <span className="text-[#8a9099] text-[10px]">· {msg.phase}</span>
                      )}
                      {msg.confidence != null && (
                        <span className="ml-auto text-[10px] text-[#34d399]">{msg.confidence}% confidence</span>
                      )}
                    </div>
                    <GlassPanel className="p-4" glow="cyan">
                      <div className="text-[#c7cbd1] text-sm leading-relaxed whitespace-pre-wrap">
                        {msg.content.split("**").map((part, i) =>
                          i % 2 === 1 ? (
                            <strong key={i} className="text-[#f3f5f7] font-semibold">{part}</strong>
                          ) : (
                            <span key={i}>{part}</span>
                          )
                        )}
                      </div>
                    </GlassPanel>
                  </div>
                ) : (
                  <div className="max-w-lg">
                    <div className="px-4 py-3 rounded-lg bg-[rgba(110,231,249,0.08)] border border-[rgba(110,231,249,0.12)]">
                      <p className="text-[#f3f5f7] text-sm leading-relaxed">{msg.content}</p>
                    </div>
                    <p className="text-right text-[10px] text-[#8a9099] mt-1">
                      {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </p>
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Thinking indicator */}
          <AnimatePresence>
            {isThinking && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-3"
              >
                <div className="w-5 h-5 rounded bg-[rgba(110,231,249,0.1)] border border-[rgba(110,231,249,0.2)] flex items-center justify-center">
                  <Brain className="w-2.5 h-2.5 text-[#6ee7f9]" />
                </div>
                <GlassPanel className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    {[0, 1, 2, 3].map((i) => (
                      <motion.div
                        key={i}
                        className="w-1.5 h-1.5 rounded-full bg-[#6ee7f9]"
                        animate={{ opacity: [0.3, 1, 0.3], scale: [0.8, 1, 0.8] }}
                        transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
                      />
                    ))}
                    <span className="text-[#8a9099] text-xs ml-2">Council deliberating</span>
                  </div>
                </GlassPanel>
              </motion.div>
            )}
          </AnimatePresence>
          <div ref={bottomRef} />
        </div>

        {/* Input area */}
        <div className="p-4 border-t border-[rgba(110,231,249,0.07)] bg-[rgba(6,7,8,0.7)] backdrop-blur-md">
          <GlassPanel className="p-3" glow="cyan">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
              placeholder="Direct the Council — pose a question, hypothesis, or directive..."
              rows={2}
              className="w-full bg-transparent text-[#f3f5f7] text-sm placeholder-[#8a9099] resize-none outline-none leading-relaxed"
            />
            <div className="flex items-center justify-between mt-2 pt-2 border-t border-[rgba(110,231,249,0.08)]">
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1.5 px-2 py-1 rounded text-[10px] text-[#8a9099] border border-[rgba(255,255,255,0.06)]">
                  <Cpu className="w-3 h-3" />
                  <span>4 Models Active</span>
                </div>
                <div className="flex items-center gap-1.5 px-2 py-1 rounded text-[10px] text-[#8a9099] border border-[rgba(255,255,255,0.06)]">
                  <Shield className="w-3 h-3" />
                  <span>Governance On</span>
                </div>
              </div>
              <button
                onClick={sendMessage}
                disabled={!input.trim() || isThinking}
                className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-[#6ee7f9] text-[#060708] text-xs font-semibold disabled:opacity-40 hover:bg-[rgba(110,231,249,0.85)] transition-colors"
              >
                <Send className="w-3.5 h-3.5" />
                Deliberate
              </button>
            </div>
          </GlassPanel>
        </div>
      </main>

      {/* ── RIGHT PANEL: INTELLIGENCE ── */}
      <AnimatePresence>
        {rightPanelOpen && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 272, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
            className="shrink-0 border-l border-[rgba(110,231,249,0.07)] bg-[rgba(6,7,8,0.8)] overflow-hidden"
          >
            <div className="w-68 h-full flex flex-col overflow-y-auto p-4 space-y-4" style={{ width: 272 }}>

              {/* Cognition Phase */}
              <div>
                <p className="text-[#8a9099] text-[9px] tracking-[0.2em] uppercase font-medium mb-2">Cognition Phase</p>
                {[
                  { label: "Ingestion",   done: true  },
                  { label: "Routing",     done: true  },
                  { label: "Deliberation",done: true  },
                  { label: "Synthesis",   done: false, active: true },
                  { label: "Governance",  done: false },
                ].map(({ label, done, active }) => (
                  <div key={label} className="flex items-center gap-2 py-1.5">
                    <div className={`w-4 h-4 rounded-full flex items-center justify-center ${done ? "bg-[rgba(52,211,153,0.15)]" : active ? "bg-[rgba(110,231,249,0.12)]" : "bg-[rgba(255,255,255,0.05)]"}`}>
                      {done
                        ? <CheckCircle2 className="w-2.5 h-2.5 text-[#34d399]" />
                        : active
                          ? <motion.div className="w-1.5 h-1.5 rounded-full bg-[#6ee7f9]" animate={{ opacity: [1, 0.3, 1] }} transition={{ duration: 1.2, repeat: Infinity }} />
                          : <div className="w-1.5 h-1.5 rounded-full bg-[#8a9099]" />
                      }
                    </div>
                    <span className={`text-xs ${done ? "text-[#34d399]" : active ? "text-[#6ee7f9]" : "text-[#8a9099]"}`}>{label}</span>
                    {active && <span className="ml-auto text-[9px] text-[#6ee7f9] font-medium">Active</span>}
                  </div>
                ))}
              </div>

              {/* Consensus meters */}
              <GlassPanel className="p-3">
                <p className="text-[#8a9099] text-[9px] tracking-[0.2em] uppercase font-medium mb-3">Live Metrics</p>
                <div className="flex justify-around">
                  <ConsensusMeter value={91} color="cyan"    size="sm" label="Consensus" />
                  <ConsensusMeter value={24} color="amber"   size="sm" label="Conflict"  />
                  <ConsensusMeter value={96} color="emerald" size="sm" label="Integrity" />
                </div>
              </GlassPanel>

              {/* Model activity */}
              <div>
                <p className="text-[#8a9099] text-[9px] tracking-[0.2em] uppercase font-medium mb-2">Model Activity</p>
                {MODEL_AGENTS.map(({ id, name, role, color, load }) => (
                  <div key={id} className="mb-2">
                    <div className="flex items-center justify-between text-[11px] mb-1">
                      <div className="flex items-center gap-1.5">
                        <div className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
                        <span className="text-[#c7cbd1]">{name}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="text-[#8a9099]">{role}</span>
                        <span className="font-medium" style={{ color }}>{load}%</span>
                      </div>
                    </div>
                    <div className="h-0.5 rounded-full bg-[rgba(255,255,255,0.06)]">
                      <motion.div
                        className="h-full rounded-full"
                        style={{ background: color }}
                        initial={{ width: 0 }}
                        animate={{ width: `${load}%` }}
                        transition={{ duration: 1.2 }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* Governance */}
              <GlassPanel className="p-3" glow="emerald">
                <p className="text-[#8a9099] text-[9px] tracking-[0.2em] uppercase font-medium mb-2">Governance</p>
                {[
                  { label: "Hallucination Gate",  status: "Pass", ok: true  },
                  { label: "Semantic Integrity",  status: "Pass", ok: true  },
                  { label: "Boundary Check",       status: "Pass", ok: true  },
                  { label: "Adversarial Probe",    status: "1 Flag",ok: false },
                ].map(({ label, status, ok }) => (
                  <div key={label} className="flex items-center justify-between py-1 border-b border-[rgba(255,255,255,0.04)] last:border-0">
                    <span className="text-[11px] text-[#8a9099]">{label}</span>
                    <span className={`text-[10px] font-medium ${ok ? "text-[#34d399]" : "text-[#f59e0b]"}`}>
                      {status}
                    </span>
                  </div>
                ))}
              </GlassPanel>

              {/* Semantic confidence */}
              <div>
                <p className="text-[#8a9099] text-[9px] tracking-[0.2em] uppercase font-medium mb-2">Semantic Confidence</p>
                {[
                  { topic: "Causal Attribution", val: 88, color: "#6ee7f9" },
                  { topic: "Evidence Grounding",  val: 94, color: "#34d399" },
                  { topic: "Logical Coherence",   val: 79, color: "#8b5cf6" },
                ].map(({ topic, val, color }) => (
                  <div key={topic} className="mb-2">
                    <div className="flex justify-between text-[10px] mb-1">
                      <span className="text-[#8a9099]">{topic}</span>
                      <span style={{ color }}>{val}%</span>
                    </div>
                    <div className="h-0.5 rounded-full bg-[rgba(255,255,255,0.06)]">
                      <motion.div className="h-full rounded-full" style={{ background: color, width: `${val}%` }}
                        initial={{ width: 0 }} animate={{ width: `${val}%` }} transition={{ duration: 1 }} />
                    </div>
                  </div>
                ))}
              </div>

            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  );
}
