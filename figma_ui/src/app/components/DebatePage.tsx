import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Swords, Brain, Shield, Eye, Zap, GitBranch,
  BarChart3, ChevronRight, AlertTriangle, CheckCircle2,
  TrendingUp, TrendingDown,
} from "lucide-react";
import { GlassPanel } from "./GlassPanel";
import { ConsensusMeter } from "./ConsensusMeter";

const MODELS = [
  {
    id: "gpt5o", name: "GPT-5o", role: "Synthesis Lead",
    color: "#6ee7f9", bg: "rgba(110,231,249,0.08)", border: "rgba(110,231,249,0.2)",
    stance: "FOR",
    argument:
      "The causal framework demonstrates strong explanatory power across the primary dataset. Three independent cross-validation runs yield consistent results, with a confidence interval that satisfies the pre-registered threshold. The mechanism aligns with established theoretical priors.",
    confidence: 87, evidence: 6,
  },
  {
    id: "claude4", name: "Claude Opus 4", role: "Adversarial Critic",
    color: "#8b5cf6", bg: "rgba(139,92,246,0.08)", border: "rgba(139,92,246,0.2)",
    stance: "AGAINST",
    argument:
      "Critical confound identified: the distribution shift between training and test conditions is non-trivial. The proposed mechanism cannot account for the observed variance at tail distributions. I flag a 23% probability this conclusion fails on out-of-distribution generalisation.",
    confidence: 72, evidence: 4,
  },
  {
    id: "gemini3", name: "Gemini 3 Flash", role: "Evidence Auditor",
    color: "#34d399", bg: "rgba(52,211,153,0.08)", border: "rgba(52,211,153,0.2)",
    stance: "QUALIFIED",
    argument:
      "Independent literature review supports the primary claim. However, two of six citations present partial alignment only. The mechanism is plausible but requires a robustness analysis across demographic subgroups before a strong consensus verdict is warranted.",
    confidence: 81, evidence: 8,
  },
  {
    id: "mistral", name: "Mistral Large", role: "Governance Auditor",
    color: "#f59e0b", bg: "rgba(245,158,11,0.08)", border: "rgba(245,158,11,0.2)",
    stance: "NEUTRAL",
    argument:
      "Governance assessment: no hallucination markers detected across primary claims. Semantic integrity passing. One adversarial boundary condition triggered — flagging potential overgeneralisation in the conclusion. Recommend conditional consensus with noted caveat.",
    confidence: 90, evidence: 5,
  },
];

const ROUNDS = [
  { round: 1, label: "Opening positions", consensus: 44, conflict: 72 },
  { round: 2, label: "Evidence exchange",  consensus: 61, conflict: 55 },
  { round: 3, label: "Cross-examination",  consensus: 74, conflict: 38 },
  { round: 4, label: "Synthesis",          consensus: 88, conflict: 21 },
];

const STANCES = {
  FOR:       { label: "For",       color: "#34d399", bg: "rgba(52,211,153,0.1)"  },
  AGAINST:   { label: "Against",   color: "#ef4444", bg: "rgba(239,68,68,0.1)"   },
  QUALIFIED: { label: "Qualified", color: "#f59e0b", bg: "rgba(245,158,11,0.1)"  },
  NEUTRAL:   { label: "Neutral",   color: "#8a9099", bg: "rgba(138,144,153,0.1)" },
};

export function DebatePage() {
  const [activeRound, setActiveRound] = useState(4);
  const [expandedModel, setExpandedModel] = useState<string | null>("gpt5o");

  const round = ROUNDS[activeRound - 1];

  return (
    <div className="min-h-screen pt-24 pb-16 px-6">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-10">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="w-6 h-6 rounded-md bg-[rgba(245,158,11,0.1)] border border-[rgba(245,158,11,0.2)] flex items-center justify-center">
                <Swords className="w-3 h-3 text-[#f59e0b]" />
              </div>
              <span className="text-[#f59e0b] text-[10px] font-medium tracking-[0.2em] uppercase">Debate Arena</span>
            </div>
            <h1 className="text-[clamp(24px,3vw,40px)] font-bold text-[#f3f5f7] tracking-tight leading-tight text-balance">
              Causal Inference in Deep Networks
            </h1>
            <p className="text-[#8a9099] text-sm mt-1">4-model adversarial tribunal · Round {activeRound} of 4</p>
          </div>
          <div className="flex items-center gap-4">
            <ConsensusMeter value={round.consensus} color="cyan"  label="Consensus" size="md" />
            <ConsensusMeter value={round.conflict}  color="amber" label="Conflict"  size="md" />
          </div>
        </div>

        {/* Round timeline */}
        <div className="flex gap-2 mb-8 overflow-x-auto pb-1">
          {ROUNDS.map((r) => (
            <button
              key={r.round}
              onClick={() => setActiveRound(r.round)}
              className={`shrink-0 flex flex-col items-start px-4 py-2.5 rounded-lg border transition-all text-left ${
                activeRound === r.round
                  ? "bg-[rgba(110,231,249,0.08)] border-[rgba(110,231,249,0.2)] text-[#f3f5f7]"
                  : "border-[rgba(255,255,255,0.06)] text-[#8a9099] hover:text-[#c7cbd1] hover:bg-[rgba(255,255,255,0.03)]"
              }`}
            >
              <span className={`text-[9px] font-semibold tracking-widest uppercase ${activeRound === r.round ? "text-[#6ee7f9]" : "text-[#8a9099]"}`}>
                Round {r.round}
              </span>
              <span className="text-xs mt-0.5">{r.label}</span>
              <div className="flex items-center gap-2 mt-1.5">
                <div className="flex items-center gap-1">
                  <TrendingUp className="w-2.5 h-2.5 text-[#34d399]" />
                  <span className="text-[10px] text-[#34d399]">{r.consensus}%</span>
                </div>
                <div className="flex items-center gap-1">
                  <TrendingDown className="w-2.5 h-2.5 text-[#ef4444]" />
                  <span className="text-[10px] text-[#ef4444]">{r.conflict}%</span>
                </div>
              </div>
            </button>
          ))}
        </div>

        {/* Main layout: columns + topology */}
        <div className="grid lg:grid-cols-3 gap-5">

          {/* Left: model positions */}
          <div className="lg:col-span-2 space-y-3">
            {MODELS.map((m, i) => {
              const stance = STANCES[m.stance as keyof typeof STANCES];
              const isExpanded = expandedModel === m.id;
              return (
                <motion.div
                  key={m.id}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.07 }}
                >
                  <GlassPanel
                    className="p-4 cursor-pointer hover:translate-y-[-1px] transition-transform duration-200"
                    style={{ borderColor: isExpanded ? m.border : undefined } as React.CSSProperties}
                  >
                    {/* Header row */}
                    <div
                      className="flex items-center justify-between"
                      onClick={() => setExpandedModel(isExpanded ? null : m.id)}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className="w-9 h-9 rounded-lg flex items-center justify-center text-[11px] font-bold"
                          style={{ background: m.bg, border: `1px solid ${m.border}`, color: m.color }}
                        >
                          {m.name.slice(0, 2)}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-[#f3f5f7] text-sm font-semibold">{m.name}</span>
                            <span className="text-[10px] px-2 py-0.5 rounded-full font-medium" style={{ background: stance.bg, color: stance.color }}>
                              {stance.label}
                            </span>
                          </div>
                          <span className="text-[#8a9099] text-[11px]">{m.role}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right hidden sm:block">
                          <div className="text-sm font-semibold" style={{ color: m.color }}>{m.confidence}%</div>
                          <div className="text-[10px] text-[#8a9099]">confidence</div>
                        </div>
                        <div className="text-right hidden sm:block">
                          <div className="text-sm font-semibold text-[#c7cbd1]">{m.evidence}</div>
                          <div className="text-[10px] text-[#8a9099]">citations</div>
                        </div>
                        <ChevronRight
                          className={`w-4 h-4 text-[#8a9099] transition-transform ${isExpanded ? "rotate-90" : ""}`}
                        />
                      </div>
                    </div>

                    {/* Expanded argument */}
                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.22 }}
                          className="overflow-hidden"
                        >
                          <div className="mt-4 pt-4 border-t border-[rgba(255,255,255,0.06)]">
                            <p className="text-[#c7cbd1] text-sm leading-relaxed">{m.argument}</p>
                            <div className="mt-3 h-1 rounded-full bg-[rgba(255,255,255,0.06)]">
                              <motion.div
                                className="h-full rounded-full"
                                style={{ background: m.color }}
                                initial={{ width: 0 }}
                                animate={{ width: `${m.confidence}%` }}
                                transition={{ duration: 0.8 }}
                              />
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </GlassPanel>
                </motion.div>
              );
            })}
          </div>

          {/* Right: topology + metrics */}
          <div className="space-y-4">
            {/* Conflict map */}
            <GlassPanel className="p-4" glow="amber">
              <p className="text-[#8a9099] text-[9px] tracking-[0.2em] uppercase font-medium mb-4">Conflict Topology</p>
              <div className="relative h-36 flex items-center justify-center">
                {/* Centre */}
                <motion.div
                  className="absolute w-10 h-10 rounded-full bg-[rgba(110,231,249,0.1)] border border-[rgba(110,231,249,0.3)] flex items-center justify-center z-10"
                  animate={{ scale: [1, 1.08, 1] }}
                  transition={{ duration: 2.8, repeat: Infinity }}
                >
                  <GitBranch className="w-4 h-4 text-[#6ee7f9]" />
                </motion.div>
                {/* Model nodes */}
                {MODELS.map((m, i) => {
                  const angle = (i / MODELS.length) * Math.PI * 2 - Math.PI / 2;
                  const r = 52;
                  const x = 50 + Math.cos(angle) * r;
                  const y = 50 + Math.sin(angle) * r;
                  return (
                    <motion.div
                      key={m.id}
                      className="absolute w-8 h-8 rounded-full flex items-center justify-center text-[9px] font-bold"
                      style={{
                        left: `${x}%`, top: `${y}%`,
                        transform: "translate(-50%,-50%)",
                        background: m.bg, border: `1px solid ${m.border}`, color: m.color,
                      }}
                      animate={{ boxShadow: [`0 0 0px ${m.color}00`, `0 0 12px ${m.color}40`, `0 0 0px ${m.color}00`] }}
                      transition={{ duration: 2.4, repeat: Infinity, delay: i * 0.5 }}
                    >
                      {m.name.slice(0, 2)}
                    </motion.div>
                  );
                })}
              </div>
            </GlassPanel>

            {/* Consensus evolution */}
            <GlassPanel className="p-4">
              <p className="text-[#8a9099] text-[9px] tracking-[0.2em] uppercase font-medium mb-3">Consensus Evolution</p>
              <div className="space-y-2">
                {ROUNDS.map((r) => (
                  <div key={r.round}>
                    <div className="flex justify-between text-[10px] mb-1">
                      <span className="text-[#8a9099]">Round {r.round}</span>
                      <span className="text-[#34d399]">{r.consensus}%</span>
                    </div>
                    <div className="h-1 rounded-full bg-[rgba(255,255,255,0.06)]">
                      <motion.div
                        className="h-full rounded-full bg-[#34d399]"
                        initial={{ width: 0 }}
                        whileInView={{ width: `${r.consensus}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8 }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </GlassPanel>

            {/* Verdict */}
            {activeRound === 4 && (
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
                <GlassPanel className="p-4" glow="cyan">
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle2 className="w-4 h-4 text-[#34d399]" />
                    <p className="text-[#34d399] text-xs font-semibold tracking-wide uppercase">Council Verdict</p>
                  </div>
                  <p className="text-[#c7cbd1] text-xs leading-relaxed">
                    Consensus reached at <strong className="text-[#f3f5f7]">88%</strong>. The primary causal claim is
                    supported with the noted boundary condition caveat from Claude Opus 4 incorporated into the final position.
                  </p>
                  <div className="flex items-center gap-2 mt-3 pt-3 border-t border-[rgba(255,255,255,0.05)]">
                    <AlertTriangle className="w-3 h-3 text-[#f59e0b]" />
                    <span className="text-[10px] text-[#f59e0b]">1 governance flag noted</span>
                  </div>
                </GlassPanel>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
