import { useState, useEffect } from "react";
import { motion } from "motion/react";
import { ArrowRight, Check, Cpu } from "lucide-react";
import { Link } from "react-router";
import { getLearningSummary, type LearningSummary } from "../api";
import { GlassPanel } from "./GlassPanel";

const models = [
  {
    id: "llama31",
    name: "Llama 3.1 8B",
    provider: "Groq",
    color: "#6366f1",
    description: "Fast, reliable anchor model with broad reasoning capability. Runs on Groq's LPU for ultra-low-latency inference.",
    features: ["128K context window", "Fast inference (Groq)", "Strong reasoning", "Free tier"],
    speed: "Instant",
    quality: "Excellent",
    badge: "Anchor",
  },
  {
    id: "gemma9b",
    name: "Gemma 2 9B IT",
    provider: "Google",
    color: "#f97316",
    description: "Google's efficient instruction-tuned model with strong analytical capabilities across diverse tasks.",
    features: ["Instruction-tuned", "Balanced reasoning", "Compact architecture", "Free tier"],
    speed: "Fast",
    quality: "Great",
    badge: "Anchor",
  },
  {
    id: "mistral7b",
    name: "Mistral 7B Instruct",
    provider: "Mistral AI",
    color: "#10b981",
    description: "European AI excellence with strong multilingual, coding, and instruction-following capabilities.",
    features: ["32K context window", "Multilingual fluency", "Code specialist", "Free tier"],
    speed: "Very Fast",
    quality: "Great",
    badge: "Debate",
  },
  {
    id: "sentinel-e",
    name: "Sentinel-E",
    provider: "NeuralOS",
    color: "#8b5cf6",
    description: "Aggregate intelligence combining Llama, Gemma, Mistral & Phi — built for structured debate and deep research using Chain-of-Thought and Tree-of-Thought reasoning.",
    features: ["Chain-of-Thought reasoning", "Tree-of-Thought exploration", "Debate & argumentation", "Multi-source research synthesis"],
    speed: "Adaptive",
    quality: "Excellent",
    badge: "Aggregate AI",
  },
];

export function ModelsPage() {
  const [learning, setLearning] = useState<LearningSummary | null>(null);

  useEffect(() => {
    getLearningSummary().then(setLearning).catch(() => {});
  }, []);

  return (
    <div className="min-h-screen pt-24">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-12"
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-[rgba(110,231,249,0.15)] bg-[rgba(110,231,249,0.05)] mb-5">
            <div className="w-1 h-1 rounded-full bg-[#6ee7f9]" />
            <span className="text-[#6ee7f9] text-[10px] font-medium tracking-[0.2em] uppercase">Model Ensemble</span>
          </div>
          <h1 className="text-[clamp(32px,5vw,54px)] font-bold tracking-[-0.03em] text-[#f3f5f7] leading-tight mb-4 text-balance">
            Cognitive engines
            <br />
            <span className="text-[#8a9099] font-light">powering the runtime</span>
          </h1>
          <p className="text-[#8a9099] max-w-lg text-sm leading-relaxed">
            Llama, Gemma, Mistral and Sentinel-E aggregate intelligence — all orchestrated for
            structured deliberation, debate, and research.
          </p>
          {learning && learning.total_feedback > 0 && (
            <div className="mt-4 inline-flex items-center gap-2 px-3 py-1.5 rounded-md bg-[rgba(52,211,153,0.08)] border border-[rgba(52,211,153,0.15)]">
              <div className="w-1.5 h-1.5 rounded-full bg-[#34d399]" />
              <span className="text-[#34d399] text-[11px] font-medium">
                Learning from {learning.total_feedback} feedback loops
                {learning.total_risk_profiles > 0 && ` · ${learning.total_risk_profiles} risk profiles`}
              </span>
            </div>
          )}
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {models.map((model, index) => (
            <motion.div
              key={model.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.08 }}
            >
              <GlassPanel className="p-6 h-full flex flex-col hover:-translate-y-0.5 transition-transform duration-300">
                <div className="flex items-start justify-between mb-4">
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center"
                    style={{ background: `${model.color}12`, border: `1px solid ${model.color}25` }}
                  >
                    <Cpu className="w-5 h-5" style={{ color: model.color }} />
                  </div>
                  <span
                    className="px-2.5 py-0.5 rounded text-[10px] font-semibold tracking-wide"
                    style={{
                      background: `${model.color}10`,
                      border: `1px solid ${model.color}25`,
                      color: model.color,
                    }}
                  >
                    {model.badge}
                  </span>
                </div>

                <h3 className="text-[#f3f5f7] font-semibold text-base mb-0.5 tracking-tight">
                  {model.name}
                </h3>
                <p className="text-[#8a9099] text-[11px] font-medium mb-3">
                  by {model.provider}
                </p>
                <p className="text-[#8a9099] text-xs leading-relaxed mb-4">
                  {model.description}
                </p>

                <div className="space-y-1.5 mb-5 flex-1">
                  {model.features.map((feature) => (
                    <div key={feature} className="flex items-center gap-2">
                      <Check className="w-3.5 h-3.5 text-[#34d399] shrink-0" />
                      <span className="text-[#c7cbd1] text-xs">{feature}</span>
                    </div>
                  ))}
                </div>

                <div className="flex items-center gap-2 mb-4">
                  <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[rgba(52,211,153,0.08)] border border-[rgba(52,211,153,0.15)]">
                    <div className="w-1 h-1 rounded-full bg-[#34d399]" />
                    <span className="text-[#34d399] text-[10px] font-medium">{model.speed}</span>
                  </div>
                  <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[rgba(110,231,249,0.08)] border border-[rgba(110,231,249,0.15)]">
                    <div className="w-1 h-1 rounded-full bg-[#6ee7f9]" />
                    <span className="text-[#6ee7f9] text-[10px] font-medium">{model.quality}</span>
                  </div>
                </div>

                <Link
                  to="/chat"
                  className="group/btn flex items-center justify-center gap-2 w-full py-2.5 rounded-lg text-xs font-semibold tracking-wide transition-all"
                  style={{
                    background: `${model.color}10`,
                    border: `1px solid ${model.color}20`,
                    color: model.color,
                  }}
                >
                  Try {model.name}
                  <ArrowRight className="w-3.5 h-3.5 group-hover/btn:translate-x-0.5 transition-transform" />
                </Link>
              </GlassPanel>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
