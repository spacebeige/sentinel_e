import { useTheme } from "next-themes";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Link } from "react-router";
import { Moon, Sun } from "lucide-react";

const ENGINES = [
  { id: "gpt4", name: "GPT-4o", provider: "OpenAI", tag: "Flagship reasoning", accent: "#10a37f", description: "State-of-the-art multimodal reasoning with long-context capability and advanced tool use." },
  { id: "claude", name: "Claude 3.5", provider: "Anthropic", tag: "Semantic analysis", accent: "#cc785c", description: "Exceptional long-form reasoning, nuanced writing, and safety-conscious response generation." },
  { id: "gemini", name: "Gemini 1.5", provider: "Google", tag: "Multimodal", accent: "#4285f4", description: "Google's frontier model with native multimodal understanding and massive context window." },
  { id: "deepseek", name: "DeepSeek-R1", provider: "DeepSeek", tag: "Deep reasoning", accent: "#7c3aed", description: "Reinforcement-trained reasoning model with chain-of-thought transparency at every step." },
  { id: "mistral", name: "Mixtral 8×7B", provider: "Mistral", tag: "Efficient routing", accent: "#f97316", description: "Mixture-of-experts architecture delivers frontier-quality outputs at fraction of compute cost." },
  { id: "llama", name: "Llama 3.1", provider: "Meta", tag: "Open source", accent: "#3b82f6", description: "Meta's open-weights model offering full transparency, fine-tuning capability, and on-device inference." },
  { id: "sigma", name: "Sentinel Σ", provider: "Sentinel-E", tag: "Orchestration core", accent: "#1d1d1f", description: "Sentinel's internal orchestration engine — routes, synthesizes, and coordinates all model outputs." },
];

export function EnginesPage() {
  const { theme, setTheme } = useTheme();
  const isDark = theme === "dark";

  const toggleTheme = () => {
    setTheme(isDark ? "light" : "dark");
  };

  return (
    <div className={`min-h-screen transition-colors duration-500 pt-28 pb-24 px-6 ${isDark ? "bg-[#09090b] text-[#f5f5f7]" : "bg-[#f5f5f7] text-[#1d1d1f]"}`}>
      {/* Compact Navbar */}
      <div className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4" style={{ background: isDark ? "rgba(9,9,11,0.8)" : "rgba(245,245,247,0.8)", backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)", borderBottom: isDark ? "1px solid rgba(255,255,255,0.05)" : "1px solid rgba(0,0,0,0.05)" }}>
        <Link to="/" className="flex items-center transition-transform hover:scale-105">
          <img src="/logo.png" alt="Logo" className="h-6 w-auto" />
        </Link>
        <button onClick={toggleTheme} className="flex items-center justify-center w-9 h-9 rounded-full transition-colors duration-200 hover:bg-black/[0.05] dark:hover:bg-white/[0.07]">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div key={isDark ? "sun" : "moon"} initial={{ opacity: 0, rotate: -90, scale: 0.8 }} animate={{ opacity: 1, rotate: 0, scale: 1 }} exit={{ opacity: 0, rotate: 90, scale: 0.8 }} transition={{ duration: 0.2 }}>
              {isDark ? <Sun className="w-[15px] h-[15px] text-[rgba(255,255,255,0.55)]" /> : <Moon className="w-[15px] h-[15px] text-[rgba(0,0,0,0.4)]" />}
            </motion.div>
          </AnimatePresence>
        </button>
      </div>

      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
          className="mb-16"
        >
          <div
            className="inline-flex items-center gap-2 mb-5 px-3 py-1.5 rounded-full"
            style={{ background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)", border: isDark ? "1px solid rgba(255,255,255,0.07)" : "1px solid rgba(0,0,0,0.06)" }}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
            <span className="text-[10px] font-bold tracking-[0.22em] text-[#8e8e93] uppercase">Engine Layer</span>
          </div>
          <h1
            className={`mb-3 ${isDark ? "text-[#f5f5f7]" : "text-[#1d1d1f]"}`}
            style={{ fontFamily: "'Inter', sans-serif", fontSize: "clamp(36px, 6vw, 64px)", fontWeight: 800, letterSpacing: "-0.04em", lineHeight: 0.95 }}
          >
            Intelligence Engines
          </h1>
          <p className={`max-w-lg ${isDark ? "text-[#8e8e93]" : "text-[#636366]"}`} style={{ fontSize: "16px", lineHeight: 1.6 }}>
            Sentinel-E routes queries through the optimal engine based on task type, complexity, and semantic intent.
          </p>
        </motion.div>

        {/* Engine grid */}
        <div className="grid md:grid-cols-2 gap-4">
          {ENGINES.map((engine, i) => (
            <motion.div
              key={engine.id}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: i * 0.07, ease: [0.16, 1, 0.3, 1] }}
              className={`group p-6 rounded-3xl cursor-default transition-all duration-300 hover:scale-[1.01] ${engine.id === "sigma" ? "md:col-span-2" : ""}`}
              style={{
                background: `rgba(${engine.accent === "#1d1d1f" ? (isDark ? "255,255,255" : "29,29,31") : hexToRgb(engine.accent)}, ${isDark ? "0.1" : "0.05"})`,
                border: isDark ? "1px solid rgba(255,255,255,0.07)" : "1px solid rgba(0,0,0,0.05)",
              }}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-3">
                    <div
                      className={`w-9 h-9 rounded-2xl flex items-center justify-center font-bold text-[13px] flex-shrink-0 ${isDark ? "text-[#f5f5f7]" : "text-[#1d1d1f]"}`}
                      style={{ background: engine.accent }}
                    >
                      {engine.name[0]}
                    </div>
                    <div>
                      <div className={`font-semibold text-[15px] ${isDark ? "text-[#f5f5f7]" : "text-[#1d1d1f]"}`} style={{ letterSpacing: "-0.015em" }}>{engine.name}</div>
                      <div className="text-[#8e8e93] text-[12px]">{engine.provider}</div>
                    </div>
                  </div>
                  <p className={`text-[13px] leading-relaxed ${isDark ? "text-[#8e8e93]" : "text-[#636366]"}`}>{engine.description}</p>
                </div>
                <span
                  className="flex-shrink-0 px-2.5 py-1 rounded-full text-[10px] font-bold tracking-wide uppercase"
                  style={{ background: `rgba(${hexToRgb(engine.accent)},0.1)`, color: engine.accent }}
                >
                  {engine.tag}
                </span>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

function hexToRgb(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r},${g},${b}`;
}
