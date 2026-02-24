import { useState, useEffect } from "react";
import { motion } from "motion/react";
import { Sparkles, ArrowRight, Check } from "lucide-react";
import { Link } from "react-router";
import { getLearningSummary, type LearningSummary } from "../api";

const models = [
  {
    id: "qwen-vl",
    name: "Qwen VL 2.5 7B",
    provider: "Alibaba Cloud",
    color: "#6366f1",
    description: "Vision-language model with powerful multimodal understanding across text, images, and documents.",
    features: ["Vision & document parsing", "Multimodal reasoning", "7B efficient architecture", "Multilingual support"],
    speed: "Fast",
    quality: "Excellent",
    badge: "Multimodal",
  },
  {
    id: "mistral",
    name: "Mistral Large",
    provider: "Mistral AI",
    color: "#f97316",
    description: "European AI excellence with strong multilingual, coding, and instruction-following capabilities.",
    features: ["32K context window", "Multilingual fluency", "Code specialist", "Efficient inference"],
    speed: "Very Fast",
    quality: "Great",
    badge: "Fastest",
  },
  {
    id: "groq",
    name: "Groq LPU",
    provider: "Groq",
    color: "#10b981",
    description: "Hardware-accelerated inference delivering near-instant responses via custom LPU architecture.",
    features: ["Ultra-low latency", "LPU acceleration", "Real-time streaming", "High throughput"],
    speed: "Instant",
    quality: "Great",
    badge: "Speed King",
  },
  {
    id: "sentinel-e",
    name: "Sentinel-E",
    provider: "NeuralOS",
    color: "#8b5cf6",
    description: "Aggregate intelligence combining Qwen, Mistral & Groq — built for structured debate and deep research using Chain-of-Thought and Tree-of-Thought reasoning.",
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
    <div className="min-h-screen bg-[#f5f5f7] pt-14">
      <div className="max-w-7xl mx-auto px-6 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h1
            className="text-[#1d1d1f] mb-4"
            style={{
              fontFamily: "'Inter', -apple-system, sans-serif",
              fontSize: 'clamp(36px, 5vw, 56px)',
              fontWeight: 700,
              letterSpacing: '-0.03em',
              lineHeight: 1.1,
            }}
          >
            Powered by
            <br />
            <span className="bg-gradient-to-r from-[#8b5cf6] to-[#06b6d4] bg-clip-text text-transparent">
              Three Engines
            </span>
          </h1>
          <p
            className="text-[#6e6e73] max-w-lg mx-auto"
            style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '17px', lineHeight: 1.6, fontWeight: 400 }}
          >
            Qwen, Mistral & Groq as the foundation — Sentinel-E as the aggregate brain for debate and research with Chain-of-Thought and Tree-of-Thought reasoning.
          </p>
          {learning && learning.total_feedback > 0 && (
            <p
              className="text-[#aeaeb2] mt-3"
              style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '12px', fontWeight: 500 }}
            >
              Continuously learning from {learning.total_feedback} feedback loops
              {learning.total_risk_profiles > 0 && ` · ${learning.total_risk_profiles} risk profiles tracked`}
            </p>
          )}
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {models.map((model, index) => (
            <motion.div
              key={model.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.08 }}
              className="group p-6 rounded-3xl bg-white border border-black/5 hover:shadow-xl hover:shadow-black/5 transition-all duration-300 hover:-translate-y-1 flex flex-col bg-[#ffffff]"
            >
              <div className="flex items-start justify-between mb-4">
                <div
                  className="w-12 h-12 rounded-2xl flex items-center justify-center"
                  style={{ backgroundColor: model.color + "15" }}
                >
                  <Sparkles className="w-6 h-6" style={{ color: model.color }} />
                </div>
                <span
                  className="px-3 py-1 rounded-full border"
                  style={{
                    backgroundColor: model.color + "10",
                    borderColor: model.color + "25",
                    color: model.color,
                    fontFamily: "'Inter', -apple-system, sans-serif",
                    fontSize: '11px',
                    fontWeight: 600,
                  }}
                >
                  {model.badge}
                </span>
              </div>

              <h3
                className="text-[#1d1d1f] mb-1"
                style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '20px', fontWeight: 600 }}
              >
                {model.name}
              </h3>
              <p
                className="text-[#6e6e73] mb-1"
                style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '13px', fontWeight: 500 }}
              >
                by {model.provider}
              </p>
              <p
                className="text-[#6e6e73] mb-4"
                style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '14px', lineHeight: 1.5, fontWeight: 400 }}
              >
                {model.description}
              </p>

              <div className="space-y-2 mb-5 flex-1">
                {model.features.map((feature) => (
                  <div key={feature} className="flex items-center gap-2">
                    <Check className="w-4 h-4 text-[#34c759] flex-shrink-0" />
                    <span
                      className="text-[#1d1d1f]"
                      style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '13px', fontWeight: 400 }}
                    >
                      {feature}
                    </span>
                  </div>
                ))}
              </div>

              <div className="flex items-center gap-3 mb-4">
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#f5f5f7]">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#34c759]" />
                  <span
                    className="text-[#6e6e73]"
                    style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '12px', fontWeight: 500 }}
                  >
                    {model.speed}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#f5f5f7]">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#007aff]" />
                  <span
                    className="text-[#6e6e73]"
                    style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '12px', fontWeight: 500 }}
                  >
                    {model.quality}
                  </span>
                </div>
              </div>

              <Link
                to="/chat"
                className="group/btn flex items-center justify-center gap-2 w-full py-2.5 rounded-2xl transition-all"
                style={{
                  backgroundColor: model.color + "10",
                  color: model.color,
                  fontFamily: "'Inter', -apple-system, sans-serif",
                  fontSize: '14px',
                  fontWeight: 600,
                }}
              >
                Try {model.name}
                <ArrowRight className="w-4 h-4 group-hover/btn:translate-x-0.5 transition-transform" />
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}