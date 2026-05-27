import { useTheme } from "next-themes";
import { useState, useEffect } from "react";
import { Link } from "react-router";
import { motion } from "motion/react";
import { HeroSection } from "./HeroSection";
import { Footer } from "./Footer";

const CAPABILITIES = [
  {
    id: "orchestration",
    badge: "01",
    tag: "Orchestration",
    title: "Multi-model cognitive routing",
    body: "Sentinel-E dynamically routes queries through a semantic orchestration layer, selecting the optimal model chain for each task type in real time.",
    accent: "#3b82f6",
    icon: "⬡",
  },
  {
    id: "reasoning",
    badge: "02",
    tag: "Reasoning",
    title: "Multi-step inference architecture",
    body: "Each response passes through layered reasoning checks — evidence weighting, confidence scoring, and semantic coherence validation before delivery.",
    accent: "#8b5cf6",
    icon: "◈",
  },
  {
    id: "memory",
    badge: "03",
    tag: "Memory",
    title: "Persistent semantic memory",
    body: "Cross-session context retention allows the system to build a cumulative understanding of your cognitive patterns and query architecture.",
    accent: "#06b6d4",
    icon: "◉",
  },
];

const STATS = [
  { value: "8+", label: "AI Engines", sub: "Integrated" },
  { value: "6", label: "Orchestration", sub: "Modes" },
  { value: "<1s", label: "Response", sub: "Latency" },
  { value: "∞", label: "Context", sub: "Depth" },
];

const ARCHITECTURE_LAYERS = [
  { label: "Semantic Parser", status: "ACTIVE", color: "#3b82f6" },
  { label: "Omega Kernel", status: "RUNNING", color: "#8b5cf6" },
  { label: "Evidence Engine", status: "STANDBY", color: "#06b6d4" },
  { label: "Boundary Guard", status: "ACTIVE", color: "#10b981" },
  { label: "Memory Matrix", status: "SYNCED", color: "#f59e0b" },
];

export default function HomePage() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);
  const isDark = theme === "dark";

  const toggleTheme = () => {
    setTheme(isDark ? "light" : "dark");
  };

  const textPrimary = isDark ? "#f5f5f7" : "#1d1d1f";
  const textSecondary = isDark ? "rgba(255,255,255,0.38)" : "rgba(0,0,0,0.45)";
  const borderColor = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.07)";
  const surfaceBg = isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)";

  if (!mounted) return null;
  return (
    <div
      className="w-full"
      style={{ background: isDark ? "#08090e" : "#f7f8fc" }}
    >
      {/* ── Hero ──────────────────────────────────────────────────────── */}
      <HeroSection />

      {/* ── Stats bar ─────────────────────────────────────────────────── */}
      <section
        className="relative py-12 px-6 overflow-hidden"
        style={{ borderTop: `1px solid ${borderColor}`, borderBottom: `1px solid ${borderColor}` }}
      >
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: isDark
              ? "rgba(12,14,20,0.6)"
              : "rgba(255,255,255,0.7)",
            backdropFilter: "blur(8px)",
          }}
        />
        <div className="relative max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-10">
          {STATS.map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.7, delay: i * 0.08, ease: [0.16, 1, 0.3, 1] }}
              className="text-center"
            >
              <div
                style={{
                  fontFamily: "'Inter', sans-serif",
                  fontSize: "clamp(30px, 4.5vw, 46px)",
                  fontWeight: 800,
                  letterSpacing: "-0.045em",
                  lineHeight: 1,
                  color: textPrimary,
                  marginBottom: "4px",
                }}
              >
                {s.value}
              </div>
              <div style={{ fontSize: "12px", fontWeight: 600, color: textSecondary }}>{s.label}</div>
              <div style={{ fontSize: "11px", fontWeight: 400, color: isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.28)" }}>{s.sub}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── Semantic Architecture Visualization ────────────────────────── */}
      <section className="py-24 px-6 overflow-hidden">
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-16 items-center">

            {/* Left: Text */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
            >
              <div
                className="inline-flex items-center gap-2 mb-6 px-3 py-1.5 rounded-full"
                style={{ background: surfaceBg, border: `1px solid ${borderColor}` }}
              >
                <span
                  style={{
                    fontSize: "10px",
                    fontWeight: 700,
                    letterSpacing: "0.2em",
                    color: isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.35)",
                    textTransform: "uppercase",
                  }}
                >
                  System Architecture
                </span>
              </div>
              <h2
                style={{
                  fontFamily: "'Inter', sans-serif",
                  fontSize: "clamp(32px, 4.5vw, 52px)",
                  fontWeight: 700,
                  letterSpacing: "-0.04em",
                  lineHeight: 1.05,
                  color: textPrimary,
                  marginBottom: "20px",
                }}
              >
                Hidden cognition.<br />
                <span style={{ color: isDark ? "rgba(255,255,255,0.35)" : "rgba(0,0,0,0.3)" }}>
                  Visible intelligence.
                </span>
              </h2>
              <p style={{ fontSize: "16px", lineHeight: 1.65, color: textSecondary, maxWidth: "440px" }}>
                Sentinel-E operates multiple reasoning layers simultaneously — most of which remain invisible to the user. The output you see is the surface of a deep cognitive stack.
              </p>
            </motion.div>

            {/* Right: Architecture panel */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1], delay: 0.1 }}
            >
              <div
                className="rounded-3xl overflow-hidden"
                style={{
                  background: isDark ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.9)",
                  border: `1px solid ${borderColor}`,
                  backdropFilter: "blur(20px)",
                  boxShadow: isDark
                    ? "0 24px 64px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05)"
                    : "0 24px 64px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.9)",
                }}
              >
                {/* Panel header */}
                <div
                  className="px-5 py-3.5 flex items-center gap-2"
                  style={{ borderBottom: `1px solid ${borderColor}` }}
                >
                  <div className="flex gap-1.5">
                    {["#ef4444", "#f59e0b", "#22c55e"].map((c) => (
                      <div key={c} className="w-2.5 h-2.5 rounded-full" style={{ background: c }} />
                    ))}
                  </div>
                  <span
                    style={{
                      fontFamily: "monospace",
                      fontSize: "11px",
                      color: isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.25)",
                      marginLeft: "6px",
                    }}
                  >
                    sentinel-e / cognitive-kernel
                  </span>
                </div>

                {/* Architecture layers */}
                <div className="p-4 space-y-2">
                  {ARCHITECTURE_LAYERS.map((layer, i) => (
                    <motion.div
                      key={layer.label}
                      initial={{ opacity: 0, x: 10 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.5, delay: 0.05 * i }}
                      className="flex items-center justify-between px-4 py-2.5 rounded-xl"
                      style={{
                        background: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.02)",
                        border: `1px solid ${borderColor}`,
                      }}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className="w-1.5 h-1.5 rounded-full"
                          style={{ background: layer.color, boxShadow: `0 0 6px ${layer.color}80` }}
                        />
                        <span style={{ fontFamily: "monospace", fontSize: "12px", color: textPrimary, fontWeight: 500 }}>
                          {layer.label}
                        </span>
                      </div>
                      <span
                        className="px-2 py-0.5 rounded-md"
                        style={{
                          fontFamily: "monospace",
                          fontSize: "9px",
                          fontWeight: 700,
                          letterSpacing: "0.12em",
                          color: layer.color,
                          background: `${layer.color}14`,
                        }}
                      >
                        {layer.status}
                      </span>
                    </motion.div>
                  ))}
                </div>

                {/* Bottom status */}
                <div
                  className="px-5 py-3 flex items-center justify-between"
                  style={{ borderTop: `1px solid ${borderColor}` }}
                >
                  <span style={{ fontFamily: "monospace", fontSize: "10px", color: isDark ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.2)" }}>
                    omega.kernel v4.5
                  </span>
                  <div className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                    <span style={{ fontFamily: "monospace", fontSize: "10px", color: "#22c55e" }}>ALL SYSTEMS NOMINAL</span>
                  </div>
                </div>
              </div>
            </motion.div>

          </div>
        </div>
      </section>

      {/* ── Capabilities ──────────────────────────────────────────────── */}
      <section className="py-24 px-6" style={{ borderTop: `1px solid ${borderColor}` }}>
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="text-center mb-16"
          >
            <div
              className="inline-flex items-center gap-2 mb-5 px-3 py-1.5 rounded-full"
              style={{ background: surfaceBg, border: `1px solid ${borderColor}` }}
            >
              <span style={{ fontSize: "10px", fontWeight: 700, letterSpacing: "0.2em", color: textSecondary, textTransform: "uppercase" }}>
                Core Capabilities
              </span>
            </div>
            <h2
              style={{
                fontFamily: "'Inter', sans-serif",
                fontSize: "clamp(30px, 4vw, 48px)",
                fontWeight: 700,
                letterSpacing: "-0.04em",
                lineHeight: 1.1,
                color: textPrimary,
              }}
            >
              Intelligence at every layer.
            </h2>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-5">
            {CAPABILITIES.map((cap, i) => (
              <motion.div
                key={cap.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-40px" }}
                transition={{ duration: 0.75, delay: i * 0.1, ease: [0.16, 1, 0.3, 1] }}
                className="group relative p-6 rounded-3xl transition-all duration-500 cursor-default overflow-hidden"
                style={{
                  background: isDark ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.85)",
                  border: `1px solid ${borderColor}`,
                  backdropFilter: "blur(12px)",
                }}
                whileHover={{ scale: 1.01, y: -3 }}
              >
                {/* Accent glow on hover */}
                <div
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none rounded-3xl"
                  style={{
                    background: `radial-gradient(ellipse 80% 60% at 50% 0%, ${cap.accent}10, transparent 70%)`,
                  }}
                />

                {/* Badge + number */}
                <div className="flex items-center gap-2 mb-5">
                  <span
                    className="text-[32px]"
                    style={{ opacity: 0.7, lineHeight: 1 }}
                  >
                    {cap.icon}
                  </span>
                  <div>
                    <div
                      style={{
                        fontFamily: "monospace",
                        fontSize: "9px",
                        fontWeight: 700,
                        letterSpacing: "0.18em",
                        color: cap.accent,
                        textTransform: "uppercase",
                      }}
                    >
                      {cap.badge} — {cap.tag}
                    </div>
                  </div>
                </div>

                <h3
                  style={{
                    fontFamily: "'Inter', sans-serif",
                    fontSize: "17px",
                    fontWeight: 600,
                    letterSpacing: "-0.025em",
                    lineHeight: 1.25,
                    color: textPrimary,
                    marginBottom: "10px",
                  }}
                >
                  {cap.title}
                </h3>
                <p style={{ fontSize: "14px", lineHeight: 1.65, color: textSecondary }}>
                  {cap.body}
                </p>

                {/* Bottom accent line */}
                <div
                  className="absolute bottom-0 left-6 right-6 h-px opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                  style={{ background: `linear-gradient(to right, transparent, ${cap.accent}40, transparent)` }}
                />
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA Banner ────────────────────────────────────────────────── */}
      <section className="px-6 pb-28 pt-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
          className="max-w-6xl mx-auto rounded-3xl overflow-hidden relative"
          style={{
            background: isDark
              ? "linear-gradient(135deg, #0d1117 0%, #111827 50%, #0d1117 100%)"
              : "linear-gradient(135deg, #1d1d1f 0%, #2d2d30 50%, #1d1d1f 100%)",
            boxShadow: "0 32px 80px rgba(0,0,0,0.25)",
          }}
        >
          {/* Subtle grid */}
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              backgroundImage: "linear-gradient(rgba(255,255,255,1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,1) 1px, transparent 1px)",
              backgroundSize: "44px 44px",
              opacity: 0.025,
              maskImage: "radial-gradient(ellipse at center, black 20%, transparent 70%)",
            }}
          />
          {/* Atmospheric glow */}
          <div
            className="absolute top-0 left-1/2 -translate-x-1/2 w-[60%] h-32 pointer-events-none"
            style={{
              background: "radial-gradient(ellipse at top, rgba(99,102,241,0.15), transparent 70%)",
              filter: "blur(20px)",
            }}
          />

          <div className="relative px-10 py-16 flex flex-col md:flex-row items-center justify-between gap-10">
            <div>
              <div
                style={{
                  fontSize: "10px",
                  fontWeight: 700,
                  letterSpacing: "0.22em",
                  textTransform: "uppercase",
                  color: "rgba(255,255,255,0.3)",
                  marginBottom: "12px",
                  fontFamily: "monospace",
                }}
              >
                Ready to Initialize?
              </div>
              <h3
                style={{
                  fontFamily: "'Inter', sans-serif",
                  fontSize: "clamp(26px, 3.5vw, 38px)",
                  fontWeight: 700,
                  letterSpacing: "-0.035em",
                  lineHeight: 1.12,
                  color: "#f5f5f7",
                }}
              >
                Start your first<br />cognitive session.
              </h3>
            </div>
            <div className="flex gap-3 flex-shrink-0">
              <Link
                to="/chat"
                className={`px-8 py-3.5 rounded-2xl font-semibold text-[14px] transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] ${isDark ? "bg-[#f5f5f7] text-[#1d1d1f]" : "bg-[#1d1d1f] text-[#f5f5f7]"}`}
                style={{ letterSpacing: "-0.01em" }}
              >
                Initialize System
              </Link>
              <Link
                to="/pricing"
                className="px-8 py-3.5 rounded-2xl font-medium text-[14px] text-white transition-all hover:scale-[1.02] active:scale-[0.98]"
                style={{
                  background: "rgba(255,255,255,0.08)",
                  border: "1px solid rgba(255,255,255,0.12)",
                  letterSpacing: "-0.01em",
                }}
              >
                View Plans
              </Link>
            </div>
          </div>
        </motion.div>
      </section>

      <Footer />
    </div>
  );
}
