import { Link } from "react-router";
import { motion } from "motion/react";
import { HeroSection } from "./HeroSection";
import { Footer } from "./Footer";

// ── Scroll section: AI Infrastructure ─────────────────────────────────────
const CAPABILITIES = [
  {
    id: "orchestration",
    badge: "01 — Orchestration",
    title: "Multi-model cognitive routing",
    body: "Sentinel-E dynamically routes queries through a semantic orchestration layer, selecting the optimal model chain for each task type.",
    accent: "rgba(59,130,246,0.06)",
  },
  {
    id: "reasoning",
    badge: "02 — Reasoning",
    title: "Multi-step inference architecture",
    body: "Each response passes through layered reasoning checks — evidence weighting, confidence scoring, and semantic coherence validation.",
    accent: "rgba(99,102,241,0.06)",
  },
  {
    id: "memory",
    badge: "03 — Memory",
    title: "Persistent semantic memory",
    body: "Cross-session context retention allows the system to build a cumulative understanding of your cognitive patterns and preferences.",
    accent: "rgba(168,85,247,0.06)",
  },
];

const STATS = [
  { value: "8", label: "AI Engines", sub: "Integrated" },
  { value: "6", label: "Orchestration", sub: "Modes" },
  { value: "<1s", label: "Response", sub: "Latency" },
  { value: "∞", label: "Context", sub: "Depth" },
];

export default function HomePage() {
  return (
    <div className="w-full bg-white dark:bg-[#090b0f]">
      {/* ── Hero ──────────────────────────────────────────────────────── */}
      <HeroSection />

      {/* ── Stats bar ─────────────────────────────────────────────────── */}
      <section className="relative border-y border-black/[0.06] dark:border-white/[0.05] py-10 px-6 overflow-hidden">
        <div
          className="absolute inset-0 dark:opacity-0 transition-opacity duration-700"
          style={{ background: "rgba(248,250,252,0.8)" }}
        />
        <div
          className="absolute inset-0 opacity-0 dark:opacity-100 transition-opacity duration-700"
          style={{ background: "rgba(12,14,18,0.8)" }}
        />
        <div className="relative max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">
          {STATS.map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 8 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: i * 0.08, ease: [0.16, 1, 0.3, 1] }}
              className="text-center"
            >
              <div
                className="text-[#1d1d1f] dark:text-white mb-0.5"
                style={{ fontFamily: "'Inter', sans-serif", fontSize: "clamp(28px, 4vw, 40px)", fontWeight: 800, letterSpacing: "-0.04em", lineHeight: 1 }}
              >
                {s.value}
              </div>
              <div className="text-[#8e8e93] dark:text-[#636366] text-[12px] font-medium">{s.label}</div>
              <div className="text-[#8e8e93] dark:text-[#636366] text-[11px]">{s.sub}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── Capabilities ──────────────────────────────────────────────── */}
      <section className="py-28 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="mb-20 max-w-xl"
          >
            <div className="inline-flex items-center gap-2 mb-4 px-3 py-1 rounded-full" style={{ background: "rgba(0,0,0,0.04)", border: "1px solid rgba(0,0,0,0.06)" }}>
              <span className="text-[10px] font-bold tracking-[0.2em] text-[#8e8e93] uppercase">System Architecture</span>
            </div>
            <h2
              className="text-[#1d1d1f] dark:text-white mb-3"
              style={{ fontFamily: "'Inter', sans-serif", fontSize: "clamp(30px, 4.5vw, 48px)", fontWeight: 700, letterSpacing: "-0.035em", lineHeight: 1.1 }}
            >
              Hidden cognition.<br />Visible intelligence.
            </h2>
            <p className="text-[#8e8e93] dark:text-[#636366]" style={{ fontSize: "16px", lineHeight: 1.6 }}>
              Sentinel-E operates multiple reasoning layers simultaneously — most of which remain invisible to the user.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-5">
            {CAPABILITIES.map((cap, i) => (
              <motion.div
                key={cap.id}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.7, delay: i * 0.1, ease: [0.16, 1, 0.3, 1] }}
                className="group p-6 rounded-3xl cursor-default transition-all duration-300 hover:scale-[1.01]"
                style={{
                  background: cap.accent,
                  border: "1px solid rgba(0,0,0,0.05)",
                }}
              >
                <div className="text-[10px] font-bold tracking-[0.18em] text-[#8e8e93] uppercase mb-4">{cap.badge}</div>
                <h3
                  className="text-[#1d1d1f] dark:text-white mb-3"
                  style={{ fontSize: "17px", fontWeight: 600, letterSpacing: "-0.02em", lineHeight: 1.25 }}
                >
                  {cap.title}
                </h3>
                <p className="text-[#8e8e93] dark:text-[#636366] text-[14px] leading-relaxed">{cap.body}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA Banner ────────────────────────────────────────────────── */}
      <section className="px-6 pb-28">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="max-w-5xl mx-auto rounded-3xl overflow-hidden relative"
          style={{
            background: "#1d1d1f",
            boxShadow: "0 24px 80px rgba(0,0,0,0.2)",
          }}
        >
          {/* Subtle grid inside CTA */}
          <div
            className="absolute inset-0 pointer-events-none opacity-[0.04]"
            style={{
              backgroundImage: "linear-gradient(rgba(255,255,255,1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,1) 1px, transparent 1px)",
              backgroundSize: "40px 40px",
              maskImage: "radial-gradient(ellipse at center, black 20%, transparent 70%)",
            }}
          />
          <div className="relative px-10 py-16 flex flex-col md:flex-row items-center justify-between gap-8">
            <div>
              <div className="text-white/40 text-[11px] font-bold tracking-[0.2em] uppercase mb-3">Ready to Initialize?</div>
              <h3
                className="text-white"
                style={{ fontSize: "clamp(24px, 3.5vw, 36px)", fontWeight: 700, letterSpacing: "-0.03em", lineHeight: 1.15 }}
              >
                Start your first<br />cognitive session.
              </h3>
            </div>
            <div className="flex gap-3 flex-shrink-0">
              <Link
                to="/chat"
                className="px-7 py-3 rounded-2xl font-semibold text-[14px] bg-white text-[#1d1d1f] transition-all hover:scale-[1.02] active:scale-[0.98]"
              >
                Initialize System
              </Link>
              <Link
                to="/pricing"
                className="px-7 py-3 rounded-2xl font-medium text-[14px] text-white transition-all hover:scale-[1.02] active:scale-[0.98]"
                style={{ background: "rgba(255,255,255,0.1)", border: "1px solid rgba(255,255,255,0.12)" }}
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
