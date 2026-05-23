import { Link } from "react-router";
import { ArrowRight, Zap, Shield, Brain, Activity } from "lucide-react";
import { motion } from "motion/react";
import { GlassPanel } from "./GlassPanel";

const MODELS = [
  { label: "GPT-5o",         color: "#6ee7f9" },
  { label: "Claude-4",       color: "#8b5cf6" },
  { label: "Gemma-2",        color: "#34d399" },
  { label: "Mistral-L",      color: "#f59e0b" },
  { label: "Llama 3.1",      color: "#6ee7f9" },
];

export function HeroSection() {
  return (
    <section className="relative min-h-screen flex flex-col justify-center overflow-hidden">
      {/* Atmospheric depth glow */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[500px] rounded-full bg-[rgba(110,231,249,0.025)] blur-[140px] pointer-events-none" />
      <div className="absolute bottom-0 right-[10%] w-[400px] h-[300px] rounded-full bg-[rgba(139,92,246,0.03)] blur-[100px] pointer-events-none" />

      <div className="relative z-10 max-w-7xl mx-auto px-6 pt-28 pb-20">
        <div className="max-w-4xl">
          {/* Status badge */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-[rgba(52,211,153,0.2)] bg-[rgba(52,211,153,0.05)] mb-8"
          >
            <motion.div
              className="w-1.5 h-1.5 rounded-full bg-[#34d399]"
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1.6, repeat: Infinity }}
            />
            <span className="text-[#34d399] text-[11px] font-medium tracking-[0.18em] uppercase">
              Orchestration Runtime Active
            </span>
          </motion.div>

          {/* Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
            className="text-[clamp(44px,7.5vw,88px)] font-bold leading-[0.95] tracking-[-0.035em] text-[#f3f5f7] mb-6 text-balance"
          >
            Visible Machine
            <br />
            <span className="text-[#6ee7f9]">Cognition</span>
          </motion.h1>

          {/* Subhead */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.22, ease: [0.22, 1, 0.36, 1] }}
            className="text-[clamp(15px,1.8vw,18px)] text-[#8a9099] max-w-2xl mb-10 leading-relaxed font-light"
          >
            Sentinel-E orchestrates multi-model AI deliberation in real time — surfacing
            consensus, conflict, and semantic reasoning across an ensemble of cognitive systems.
          </motion.p>

          {/* CTA row */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.34, ease: [0.22, 1, 0.36, 1] }}
            className="flex flex-wrap gap-3"
          >
            <Link
              to="/chat"
              className="group inline-flex items-center gap-2 px-7 py-3 rounded-lg bg-[#6ee7f9] text-[#060708] font-semibold text-sm tracking-wide hover:bg-[rgba(110,231,249,0.85)] transition-all hover:scale-[1.02] active:scale-[0.98]"
            >
              Enter Deliberation
              <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </Link>
            <Link
              to="/models"
              className="inline-flex items-center gap-2 px-7 py-3 rounded-lg border border-[rgba(110,231,249,0.18)] bg-[rgba(110,231,249,0.04)] text-[#c7cbd1] font-medium text-sm tracking-wide hover:text-[#f3f5f7] hover:bg-[rgba(110,231,249,0.08)] transition-all"
            >
              Explore Models
            </Link>
          </motion.div>
        </div>

        {/* Live orchestration panel */}
        <motion.div
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="mt-20 max-w-3xl"
        >
          <GlassPanel className="p-5" glow="cyan">
            {/* Panel header */}
            <div className="flex items-center gap-3 mb-4 pb-3 border-b border-[rgba(110,231,249,0.08)]">
              <div className="flex gap-1.5">
                <div className="w-2 h-2 rounded-full bg-[#ef4444] opacity-70" />
                <div className="w-2 h-2 rounded-full bg-[#f59e0b] opacity-70" />
                <div className="w-2 h-2 rounded-full bg-[#34d399] opacity-70" />
              </div>
              <span className="text-[#8a9099] text-[10px] font-mono tracking-wider flex-1">
                sentinel-e / council-session · live deliberation
              </span>
              <div className="flex items-center gap-1.5">
                <motion.div
                  className="w-1.5 h-1.5 rounded-full bg-[#34d399]"
                  animate={{ opacity: [1, 0.2, 1] }}
                  transition={{ duration: 1.4, repeat: Infinity }}
                />
                <span className="text-[#34d399] text-[10px] font-medium">deliberating</span>
              </div>
            </div>

            {/* Agent messages */}
            <div className="space-y-3 mb-4">
              {[
                { model: "GPT-5o",   color: "#6ee7f9", msg: "Semantic coherence confirms the causal chain across all context windows." },
                { model: "Claude-4", color: "#8b5cf6", msg: "Divergence on attribution weight — evidence layer at 0.71 confidence." },
                { model: "Gemma-2",  color: "#34d399", msg: "Orthogonal synthesis path resolves conflict. Consensus threshold met." },
              ].map(({ model, color, msg }) => (
                <div key={model} className="flex items-start gap-3">
                  <div
                    className="mt-0.5 w-6 h-6 rounded shrink-0 flex items-center justify-center text-[9px] font-bold"
                    style={{ background: `${color}12`, border: `1px solid ${color}25`, color }}
                  >
                    {model.slice(0, 2)}
                  </div>
                  <div>
                    <span className="text-[10px] font-semibold mr-2" style={{ color }}>{model}</span>
                    <span className="text-[#8a9099] text-[11px] leading-relaxed">{msg}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Model activity bar */}
            <div className="flex items-center gap-2 pt-3 border-t border-[rgba(110,231,249,0.06)]">
              <Activity className="w-3 h-3 text-[#8a9099] shrink-0" />
              <span className="text-[#8a9099] text-[10px] mr-2">Active:</span>
              {MODELS.map(({ label, color }) => (
                <div
                  key={label}
                  className="flex items-center gap-1 px-2 py-0.5 rounded text-[9px] font-medium"
                  style={{ background: `${color}10`, border: `1px solid ${color}20`, color }}
                >
                  <motion.div
                    className="w-1 h-1 rounded-full"
                    style={{ background: color }}
                    animate={{ opacity: [1, 0.3, 1] }}
                    transition={{ duration: 1.8, repeat: Infinity, delay: Math.random() * 1 }}
                  />
                  {label}
                </div>
              ))}
            </div>
          </GlassPanel>
        </motion.div>

        {/* Feature pills */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.7 }}
          className="mt-8 flex flex-wrap gap-3"
        >
          {[
            { icon: <Brain className="w-3.5 h-3.5" />,    label: "Multi-Model Ensemble",    desc: "Llama, Gemma, Mistral, GPT" },
            { icon: <Zap className="w-3.5 h-3.5" />,      label: "Sub-200ms Orchestration", desc: "Real-time deliberation" },
            { icon: <Shield className="w-3.5 h-3.5" />,   label: "Governance Layer",        desc: "Built-in integrity checks" },
          ].map((f) => (
            <div
              key={f.label}
              className="flex items-center gap-2.5 px-4 py-2.5 rounded-lg bg-[rgba(17,18,20,0.7)] border border-[rgba(110,231,249,0.08)] backdrop-blur-sm"
            >
              <div className="text-[#6ee7f9]">{f.icon}</div>
              <div>
                <div className="text-[#c7cbd1] text-[12px] font-semibold leading-none mb-0.5">{f.label}</div>
                <div className="text-[#8a9099] text-[10px]">{f.desc}</div>
              </div>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
