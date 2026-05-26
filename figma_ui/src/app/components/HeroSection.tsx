import { Link } from "react-router";
import { motion, useMotionTemplate, useMotionValue } from "motion/react";
import { useChatInteraction } from "../context/ChatInteractionContext";
import type { MouseEvent } from "react";

const FLOAT_PLANES = [
  { top: "18%", left: "7%", w: "220px", h: "220px", delay: 0, duration: 22 },
  { bottom: "20%", right: "7%", w: "280px", h: "160px", delay: 4, duration: 28 },
  { top: "58%", left: "72%", w: "120px", h: "120px", delay: 8, duration: 18 },
];

export function HeroSection() {
  const { isProMode } = useChatInteraction();
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  function onMouseMove({ currentTarget, clientX, clientY }: MouseEvent<HTMLElement>) {
    const { left, top } = currentTarget.getBoundingClientRect();
    mouseX.set(clientX - left);
    mouseY.set(clientY - top);
  }

  return (
    <section
      className="relative min-h-screen overflow-hidden flex items-center justify-center group"
      onMouseMove={onMouseMove}
    >
      {/* ── L1: ATMOSPHERE ─────────────────────────────────────────────── */}
      <div className="absolute inset-0 pointer-events-none" aria-hidden>
        <div
          className="absolute inset-0 dark:opacity-0 transition-opacity duration-700"
          style={{ background: "radial-gradient(ellipse 80% 60% at 50% 40%, #e8edf5 0%, #f8fafc 50%, #ffffff 100%)" }}
        />
        <div
          className="absolute inset-0 opacity-0 dark:opacity-100 transition-opacity duration-700"
          style={{ background: "radial-gradient(ellipse 80% 60% at 50% 40%, #0d1117 0%, #0a0d12 60%, #090b0f 100%)" }}
        />
        {/* Ambient blue accent */}
        <div
          className="absolute w-[50vw] h-[50vw] rounded-full pointer-events-none"
          style={{
            top: "10%", left: "55%",
            background: "radial-gradient(circle, rgba(59,130,246,0.05) 0%, transparent 70%)",
            filter: "blur(60px)",
          }}
        />
        <div
          className="absolute w-[35vw] h-[35vw] rounded-full pointer-events-none"
          style={{
            bottom: "15%", left: "15%",
            background: "radial-gradient(circle, rgba(99,102,241,0.04) 0%, transparent 70%)",
            filter: "blur(80px)",
          }}
        />
      </div>

      {/* ── L2: COMPUTATIONAL TOPOLOGY ─────────────────────────────────── */}
      <div
        className="absolute inset-0 pointer-events-none dark:opacity-0 transition-opacity duration-700"
        style={{
          backgroundImage: "linear-gradient(rgba(0,0,0,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.035) 1px, transparent 1px)",
          backgroundSize: "52px 52px",
          maskImage: "radial-gradient(ellipse 60% 50% at 50% 50%, black, transparent)",
          WebkitMaskImage: "radial-gradient(ellipse 60% 50% at 50% 50%, black, transparent)",
        }}
      />
      <div
        className="absolute inset-0 pointer-events-none opacity-0 dark:opacity-100 transition-opacity duration-700"
        style={{
          backgroundImage: "linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)",
          backgroundSize: "52px 52px",
          maskImage: "radial-gradient(ellipse 60% 50% at 50% 50%, black, transparent)",
          WebkitMaskImage: "radial-gradient(ellipse 60% 50% at 50% 50%, black, transparent)",
        }}
      />

      {/* ── L3: FLOATING DEPTH PLANES ──────────────────────────────────── */}
      {FLOAT_PLANES.map((p, i) => (
        <motion.div
          key={i}
          className="absolute rounded-3xl hidden xl:block pointer-events-none"
          style={{
            top: p.top, bottom: (p as any).bottom, left: p.left, right: (p as any).right,
            width: p.w, height: p.h,
            background: "rgba(255,255,255,0.28)",
            backdropFilter: "blur(12px)",
            WebkitBackdropFilter: "blur(12px)",
            border: "1px solid rgba(0,0,0,0.05)",
            boxShadow: "0 4px 24px rgba(0,0,0,0.03)",
          }}
          animate={{
            y: [0, i % 2 === 0 ? -12 : 12, 0],
            rotate: [0, i % 2 === 0 ? -0.6 : 0.6, 0],
          }}
          transition={{ duration: p.duration, repeat: Infinity, ease: "easeInOut", delay: p.delay }}
        />
      ))}

      {/* ── L4: MOUSE SEMANTIC PULSE ────────────────────────────────────── */}
      <motion.div
        className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
        style={{
          background: useMotionTemplate`radial-gradient(650px circle at ${mouseX}px ${mouseY}px, rgba(59,130,246,0.035), transparent 65%)`,
        }}
      />

      {/* ── L5: HERO COMPOSITION ────────────────────────────────────────── */}
      <div className="relative z-10 flex flex-col items-center text-center px-6 max-w-[900px] pt-24 pb-16">

        {/* Semantic badge */}
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
          className="mb-8 inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full"
          style={{
            background: "rgba(0,0,0,0.04)",
            border: "1px solid rgba(0,0,0,0.07)",
          }}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse flex-shrink-0" />
          <span className="text-[10px] font-bold tracking-[0.22em] text-[#6e6e73] dark:text-[#8e8e93] uppercase">
            Semantic OS {isProMode ? "· Pro Active" : "· Active"}
          </span>
        </motion.div>

        {/* Wordmark */}
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1], delay: 0.07 }}
          className="text-[#1d1d1f] dark:text-white mb-5"
          style={{
            fontFamily: "'Inter', sans-serif",
            fontSize: "clamp(58px, 10vw, 108px)",
            fontWeight: 800,
            lineHeight: 0.93,
            letterSpacing: "-0.04em",
          }}
        >
          Sentinel-E
        </motion.h1>

        {/* Descriptors */}
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1], delay: 0.18 }}
          className="flex flex-col items-center gap-1.5 mb-11"
        >
          <p
            className="text-[#1d1d1f] dark:text-white font-semibold"
            style={{ fontSize: "clamp(17px, 2.2vw, 22px)", letterSpacing: "-0.025em" }}
          >
            The Cognitive Operating System.
          </p>
          <p
            className="text-[#8e8e93] dark:text-[#636366]"
            style={{ fontSize: "clamp(14px, 1.8vw, 18px)", letterSpacing: "-0.01em" }}
          >
            Military-grade intelligence architecture.
          </p>
        </motion.div>

        {/* CTA cluster */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1], delay: 0.3 }}
          className="flex flex-wrap items-center justify-center gap-3"
        >
          <Link
            to="/chat"
            className="px-7 py-3 rounded-2xl font-semibold text-[14px] transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] hover:shadow-xl"
            style={{
              background: "#1d1d1f",
              color: "white",
              boxShadow: "0 2px 16px rgba(0,0,0,0.15)",
              letterSpacing: "-0.01em",
            }}
          >
            Initialize System
          </Link>
          <Link
            to="/engines"
            className="px-7 py-3 rounded-2xl font-medium text-[14px] text-[#1d1d1f] dark:text-white transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
            style={{
              background: "rgba(0,0,0,0.05)",
              border: "1px solid rgba(0,0,0,0.08)",
              letterSpacing: "-0.01em",
            }}
          >
            Explore Engines
          </Link>
        </motion.div>

        {/* Pro telemetry strip */}
        {isProMode && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.5 }}
            className="mt-14 flex items-center gap-8"
          >
            {["NET: OPTIMAL", "SYNTH: ACTIVE", "ROUTING: PRO"].map((s) => (
              <span key={s} className="text-[9px] font-mono tracking-[0.2em] text-black/20 dark:text-white/15">
                {s}
              </span>
            ))}
          </motion.div>
        )}
      </div>
    </section>
  );
}