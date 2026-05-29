import { useRef, useEffect, useState } from "react";
import { Link } from "react-router";
import { useTheme } from "next-themes";
import { motion, useMotionTemplate, useMotionValue, useSpring } from "motion/react";
import type { MouseEvent } from "react";
import { useAuthContext } from "../providers/AuthProvider";

// ── Floating glass depth planes ─────────────────────────────────────────────
const FLOAT_PLANES = [
  { top: "16%", left: "5%",   w: "260px", h: "200px", delay: 0,  dur: 24, rx: -1.2 },
  { top: "60%", right: "5%",  w: "320px", h: "180px", delay: 5,  dur: 30, rx: 0.8  },
  { top: "30%", right: "8%",  w: "140px", h: "140px", delay: 9,  dur: 20, rx: -0.6 },
  { bottom: "14%", left: "8%", w: "200px", h: "120px", delay: 3, dur: 26, rx: 1.0  },
];

// ── Neural topology SVG paths ────────────────────────────────────────────────
const NEURAL_PATHS = [
  "M 120 200 Q 300 80 500 250 T 880 200",
  "M 50 350 Q 220 180 450 320 T 780 280",
  "M 200 450 Q 420 300 620 420 T 950 380",
  "M 80 150 L 280 250 L 500 180 L 720 280 L 900 200",
  "M 150 500 Q 380 380 600 480 T 900 450",
  "M 0 300 Q 250 450 500 300 T 1000 300", // New horizontal pathway
];

const NEURAL_NODES = [
  { cx: 280, cy: 250 }, { cx: 500, cy: 180 }, { cx: 720, cy: 280 },
  { cx: 380, cy: 380 }, { cx: 600, cy: 310 }, { cx: 820, cy: 350 },
  { cx: 200, cy: 420 }, { cx: 460, cy: 450 }, { cx: 680, cy: 390 },
  { cx: 500, cy: 300 }, { cx: 880, cy: 200 }, { cx: 120, cy: 200 }
];


function CinematicRevealLayer({ isDark }: { isDark: boolean }) {
  const layerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;

    let targetX = window.innerWidth / 2;
    let targetY = window.innerHeight / 2;
    let currentX = targetX;
    let currentY = targetY;
    let rafId: number;

    const onPointerMove = (e: PointerEvent) => {
      targetX = e.clientX;
      targetY = e.clientY;
    };

    window.addEventListener("pointermove", onPointerMove, { passive: true });

    const animate = () => {
      currentX += (targetX - currentX) * 0.1;
      currentY += (targetY - currentY) * 0.1;
      
      layer.style.setProperty("--x", `${currentX}px`);
      layer.style.setProperty("--y", `${currentY}px`);
      rafId = requestAnimationFrame(animate);
    };
    rafId = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      cancelAnimationFrame(rafId);
    };
  }, []);

  return (
    <div
      ref={layerRef}
      className="pointer-events-none absolute inset-0 overflow-hidden"
      style={{ zIndex: 5 }}
      aria-hidden
    >
      {/* 1. Main Ambient Illumination */}
      <div 
        className="absolute inset-0 transition-opacity duration-500"
        style={{
          background: isDark
            ? "radial-gradient(circle 240px at var(--x, 50%) var(--y, 50%), rgba(255,255,255,0.08), rgba(120,140,255,0.05), transparent 72%)"
            : "radial-gradient(circle 220px at var(--x, 50%) var(--y, 50%), rgba(255,255,255,0.22), rgba(240,240,255,0.12), transparent 70%)"
        }}
      />
      
      {/* 2. Hidden Semantic Mesh (Only visible near cursor) */}
      <div 
        className="absolute inset-0 opacity-[0.06] transition-opacity duration-500"
        style={{
          backgroundImage: isDark
            ? "linear-gradient(rgba(255,255,255,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.3) 1px, transparent 1px)"
            : "linear-gradient(rgba(0,0,0,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.3) 1px, transparent 1px)",
          backgroundSize: "40px 40px",
          WebkitMaskImage: "radial-gradient(circle 240px at var(--x, 50%) var(--y, 50%), black, transparent 80%)",
          maskImage: "radial-gradient(circle 240px at var(--x, 50%) var(--y, 50%), black, transparent 80%)"
        }}
      />
    </div>
  );
}

export function HeroSection() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const { isAuthenticated } = useAuthContext();

  useEffect(() => {
    setMounted(true);
  }, []);
  const isDark = theme === "dark";

  const sectionRef = useRef<HTMLElement>(null);
  // Smooth mouse tracking
  const rawX = useMotionValue(0);
  const rawY = useMotionValue(0);
  const mouseX = useSpring(rawX, { damping: 50, stiffness: 300 });
  const mouseY = useSpring(rawY, { damping: 50, stiffness: 300 });

  // Magnetic secondary orb (smaller, offset)
  const rawX2 = useMotionValue(0);
  const rawY2 = useMotionValue(0);
  const mouse2X = useSpring(rawX2, { damping: 80, stiffness: 150 });
  const mouse2Y = useSpring(rawY2, { damping: 80, stiffness: 150 });
  function onMouseMove({ currentTarget, clientX, clientY }: MouseEvent<HTMLElement>) {
    const { left, top } = currentTarget.getBoundingClientRect();
    rawX.set(clientX - left);
    rawY.set(clientY - top);
    rawX2.set((clientX - left) * 0.85 + 60);
    rawY2.set((clientY - top) * 0.85 + 80);
  }


  const maskImageTemplate = useMotionTemplate`radial-gradient(circle 450px at ${mouseX}px ${mouseY}px, black 15%, transparent 70%), radial-gradient(ellipse 70% 60% at 50% 50%, black 10%, transparent 80%)`;
  const bgTemplate1 = useMotionTemplate`radial-gradient(800px circle at ${mouseX}px ${mouseY}px, ${isDark ? "rgba(99,102,241,0.08)" : "rgba(59,130,246,0.06)"}, transparent 60%)`;
  const bgTemplate2 = useMotionTemplate`radial-gradient(500px circle at ${mouse2X}px ${mouse2Y}px, ${isDark ? "rgba(139,92,246,0.07)" : "rgba(99,102,241,0.05)"}, transparent 60%)`;

  if (!mounted) return null;
  return (
    <section
      ref={sectionRef}
      className="relative min-h-screen overflow-hidden flex items-center justify-center group"
      onMouseMove={onMouseMove}
      style={{ background: isDark ? "#08090e" : "#f7f8fc" }}
    >

      <CinematicRevealLayer isDark={isDark} />

      {/* ── L1: ATMOSPHERIC BASE ────────────────────────────────────────── */}
      <div className="absolute inset-0 pointer-events-none" aria-hidden>

        {/* Light mode: soft computational haze */}
        <div
          className="absolute inset-0 transition-opacity duration-700"
          style={{
            opacity: isDark ? 0 : 1,
            background: `
              radial-gradient(ellipse 90% 70% at 50% 30%, rgba(228,234,248,0.9) 0%, rgba(245,247,252,0.7) 45%, rgba(247,248,252,0) 75%),
              radial-gradient(ellipse 50% 50% at 80% 60%, rgba(209,219,255,0.35) 0%, transparent 60%),
              radial-gradient(ellipse 40% 40% at 20% 70%, rgba(224,230,255,0.3) 0%, transparent 60%)
            `,
          }}
        />

        {/* Dark mode: deep cosmic atmosphere */}
        <div
          className="absolute inset-0 transition-opacity duration-700"
          style={{
            opacity: isDark ? 1 : 0,
            background: `
              radial-gradient(ellipse 90% 70% at 50% 30%, rgba(13,17,30,0.95) 0%, rgba(8,9,14,0.8) 50%, rgba(8,9,14,0) 80%),
              radial-gradient(ellipse 55% 50% at 75% 55%, rgba(30,40,90,0.4) 0%, transparent 60%),
              radial-gradient(ellipse 45% 45% at 25% 65%, rgba(20,30,80,0.35) 0%, transparent 60%)
            `,
          }}
        />

        {/* Blue accent orb — top right */}
        <motion.div
          className="absolute rounded-full pointer-events-none"
          style={{
            top: "5%", right: "10%",
            width: "45vw", height: "45vw",
            background: isDark
              ? "radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 65%)"
              : "radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 65%)",
            filter: "blur(60px)",
          }}
          animate={{ scale: [1, 1.08, 1], opacity: [0.7, 1, 0.7] }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        />

        {/* Violet accent orb — bottom left */}
        <motion.div
          className="absolute rounded-full pointer-events-none"
          style={{
            bottom: "10%", left: "5%",
            width: "40vw", height: "40vw",
            background: isDark
              ? "radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 65%)"
              : "radial-gradient(circle, rgba(59,130,246,0.04) 0%, transparent 65%)",
            filter: "blur(80px)",
          }}
          animate={{ scale: [1, 1.12, 1], opacity: [0.5, 0.8, 0.5] }}
          transition={{ duration: 10, repeat: Infinity, ease: "easeInOut", delay: 3 }}
        />
      </div>

      {/* ── L2: HIDDEN COMPUTATIONAL TOPOLOGY ──────────────────────────── */}
      {/* 
        Hover Interaction System:
        We use motion templates based on mouseX/mouseY to create a spotlight effect.
        This "illuminates" the hidden topology underneath, creating a sense of deep interaction.
      */}
      <motion.div
        className="absolute inset-0 pointer-events-none overflow-hidden transition-opacity duration-1000"
        style={{
          maskImage: maskImageTemplate,
          WebkitMaskImage: maskImageTemplate,
        }}
        aria-hidden
      >
        <svg
          className="absolute inset-0 w-full h-full"
          viewBox="0 0 1000 600"
          preserveAspectRatio="xMidYMid slice"
          fill="none"
          style={{ opacity: isDark ? 0.75 : 0.5 }}
        >
          {/* Neural paths */}
          {NEURAL_PATHS.map((d, i) => (
            <path
              key={i}
              d={d}
              stroke={isDark ? "rgba(99,102,241,0.25)" : "rgba(59,130,246,0.2)"}
              strokeWidth="1.5"
              className="neural-pulse"
              style={{ animationDelay: `${i * 0.8}s` }}
            />
          ))}

          {/* Neural nodes */}
          {NEURAL_NODES.map((n, i) => (
            <circle
              key={i}
              cx={n.cx} cy={n.cy} r="3"
              fill={isDark ? "rgba(139,92,246,0.4)" : "rgba(59,130,246,0.3)"}
              className="neural-pulse"
              style={{ animationDelay: `${i * 0.4}s` }}
            />
          ))}

          {/* Semantic grid overlay */}
          <defs>
            <pattern id="semanticGrid" width="60" height="60" patternUnits="userSpaceOnUse">
              <path
                d="M 60 0 L 0 0 0 60"
                fill="none"
                stroke={isDark ? "rgba(255,255,255,0.035)" : "rgba(0,0,0,0.04)"}
                strokeWidth="1"
              />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#semanticGrid)" />
        </svg>
      </motion.div>

      {/* ── L3: FLOATING GLASS DEPTH PLANES ────────────────────────────── */}
      {FLOAT_PLANES.map((p, i) => {
        const style: React.CSSProperties = {
          top: p.top,
          left: p.left,
          right: (p as any).right,
          bottom: (p as any).bottom,
          width: p.w,
          height: p.h,
          background: isDark
            ? "rgba(255,255,255,0.015)"
            : "rgba(255,255,255,0.45)",
          backdropFilter: "blur(24px) saturate(180%)",
          WebkitBackdropFilter: "blur(24px) saturate(180%)",
          border: isDark
            ? "1px solid rgba(255,255,255,0.04)"
            : "1px solid rgba(0,0,0,0.04)",
          boxShadow: isDark
            ? "0 12px 48px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05)"
            : "0 12px 48px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.8)",
          borderRadius: "24px",
        };
        return (
          <motion.div
            key={i}
            className="absolute hidden xl:block pointer-events-none"
            style={style}
            animate={{
              y: [0, i % 2 === 0 ? -18 : 18, 0],
              rotate: [0, p.rx, 0],
              scale: [1, 1.015, 1],
            }}
            transition={{
              duration: p.dur,
              repeat: Infinity,
              ease: "easeInOut",
              delay: p.delay,
            }}
          >
            {/* Subtle inner grid for architectural feel */}
            <div
              className="absolute inset-0 opacity-[0.03] dark:opacity-[0.05] rounded-[24px]"
              style={{
                backgroundImage: "linear-gradient(rgba(0,0,0,1) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,1) 1px, transparent 1px)",
                backgroundSize: "20px 20px",
              }}
            />
          </motion.div>
        );
      })}

      {/* ── L4: DUAL MOUSE SEMANTIC PULSE ──────────────────────────────── */}
      {/* Primary orb — follows mouse directly */}
      <motion.div
        className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-700"
        style={{
          background: bgTemplate1,
        }}
      />
      {/* Secondary orb — slightly lagged, different color */}
      <motion.div
        className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-1000"
        style={{
          background: bgTemplate2,
        }}
      />

      {/* ── L5: HERO COMPOSITION ────────────────────────────────────────── */}
      <div className="relative z-10 flex flex-col items-center text-center px-6 max-w-[980px] pt-28 pb-20">

        {/* System status badge */}
        <motion.div
          initial={{ opacity: 0, y: 8, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="mb-9 inline-flex items-center gap-2.5 px-4 py-2 rounded-full"
          style={{
            background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)",
            border: isDark ? "1px solid rgba(255,255,255,0.09)" : "1px solid rgba(0,0,0,0.07)",
            backdropFilter: "blur(12px)",
          }}
        >
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-60" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500" />
          </span>
          <span
            style={{
              fontFamily: "'Inter', sans-serif",
              fontSize: "10px",
              fontWeight: 700,
              letterSpacing: "0.2em",
              color: isDark ? "rgba(255,255,255,0.45)" : "rgba(0,0,0,0.4)",
              textTransform: "uppercase",
            }}
          >
            Semantic OS · Cognitive Kernel Active
          </span>
        </motion.div>

        {/* Wordmark — massive cinematic title */}
        <motion.div
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          whileHover={{ scale: 1.015, y: -2 }}
          transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1], delay: 0.06 }}
          className="relative cursor-default"
        >
          {/* Ambient text glow */}
          <div
            className="absolute inset-0 blur-3xl opacity-40 pointer-events-none transition-opacity duration-500"
            style={{
              background: isDark
                ? "radial-gradient(circle, rgba(180,200,255,0.3) 0%, rgba(255,255,255,0.1) 40%, transparent 70%)"
                : "radial-gradient(circle, rgba(255,255,255,0.8) 0%, rgba(99,102,241,0.15) 40%, transparent 70%)",
            }}
          />
          <h1
            style={{
              fontFamily: "'Inter', sans-serif",
              fontSize: "clamp(64px, 13vw, 132px)",
              fontWeight: 800,
              lineHeight: 0.88,
              letterSpacing: "-0.05em",
              color: "transparent",
              backgroundImage: isDark
                ? "linear-gradient(180deg, #ffffff 0%, rgba(255,255,255,0.65) 100%)"
                : "linear-gradient(180deg, #1d1d1f 0%, rgba(29,29,31,0.55) 100%)",
              WebkitBackgroundClip: "text",
              backgroundClip: "text",
              marginBottom: "clamp(18px, 2.5vw, 28px)",
              filter: isDark
                ? "drop-shadow(0px 12px 24px rgba(0,0,0,0.8)) drop-shadow(0px 0px 48px rgba(180,200,255,0.3)) drop-shadow(0px 0px 12px rgba(255,255,255,0.4))"
                : "drop-shadow(0px 8px 16px rgba(0,0,0,0.2)) drop-shadow(0px 0px 36px rgba(99,102,241,0.25)) drop-shadow(0px 0px 10px rgba(255,255,255,0.8))",
            }}
          >
            Sentinel-E
          </h1>
        </motion.div>

        {/* Tagline */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.0, ease: [0.16, 1, 0.3, 1], delay: 0.16 }}
          className="flex flex-col items-center gap-2 mb-12"
        >
          <p
            style={{
              fontFamily: "'Inter', sans-serif",
              fontSize: "clamp(18px, 2.5vw, 26px)",
              fontWeight: 600,
              letterSpacing: "-0.03em",
              color: isDark ? "rgba(245,245,247,0.9)" : "#1d1d1f",
            }}
          >
            The Cognitive Operating System.
          </p>
          <p
            style={{
              fontFamily: "'Inter', sans-serif",
              fontSize: "clamp(14px, 1.8vw, 18px)",
              fontWeight: 400,
              letterSpacing: "-0.01em",
              color: isDark ? "rgba(255,255,255,0.45)" : "rgba(0,0,0,0.5)",
              maxWidth: "600px",
              lineHeight: 1.6,
            }}
          >
            A cinematic cognitive operating system with hidden machine intelligence beneath glass, orchestrating semantic reasoning, adaptive cognition, and living multi-model intelligence through a responsive architectural surface.
          </p>
        </motion.div>

        {/* CTA Cluster */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1], delay: 0.28 }}
          className="flex flex-wrap items-center justify-center gap-3"
        >
          {/* Primary CTA */}
          <Link
            to={isAuthenticated ? "/chat" : "/signup"}
            className="relative overflow-hidden px-8 py-3.5 rounded-full font-semibold text-[15px] transition-all duration-300 hover:scale-[1.03] active:scale-[0.97]"
            style={{
              background: isDark ? "#f5f5f7" : "#1d1d1f",
              color: isDark ? "#1d1d1f" : "#ffffff",
              letterSpacing: "-0.01em",
              boxShadow: isDark
                ? "0 0 0 1px rgba(255,255,255,0.1), 0 8px 24px rgba(0,0,0,0.5)"
                : "0 2px 16px rgba(0,0,0,0.18), 0 1px 4px rgba(0,0,0,0.12)",
            }}
          >
            {isAuthenticated ? "Chat" : "Sign Up"}
          </Link>

          {/* Secondary CTA */}
          <Link
            to={isAuthenticated ? "/engines" : "/login"}
            className="px-8 py-3.5 rounded-full font-medium text-[15px] transition-all duration-300 hover:scale-[1.03] active:scale-[0.97]"
            style={{
              background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)",
              color: isDark ? "rgba(255,255,255,0.8)" : "#1d1d1f",
              border: isDark ? "1px solid rgba(255,255,255,0.09)" : "1px solid rgba(0,0,0,0.09)",
              letterSpacing: "-0.01em",
              backdropFilter: "blur(8px)",
            }}
          >
            {isAuthenticated ? "Explore Engines" : "Login"}
          </Link>
        </motion.div>

        {/* Telemetry strip */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.4, delay: 0.7 }}
          className="mt-16 flex items-center gap-8 flex-wrap justify-center"
        >
          {["NET: OPTIMAL", "SYNTH: ACTIVE", "COGNITION: ONLINE", "ROUTING: READY"].map((s, i) => (
            <span
              key={s}
              className="neural-pulse"
              style={{
                fontFamily: "monospace",
                fontSize: "9px",
                fontWeight: 600,
                letterSpacing: "0.22em",
                color: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.13)",
                animationDelay: `${i * 0.5}s`,
              }}
            >
              {s}
            </span>
          ))}
        </motion.div>
      </div>

      {/* ── SCROLL INDICATOR ────────────────────────────────────────────── */}
      <motion.div
        className="absolute bottom-8 left-1/2 -translate-x-1/2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
      >
        <motion.div
          animate={{ y: [0, 6, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          className="flex flex-col items-center gap-1.5"
        >
          <span
            style={{
              fontFamily: "'Inter', sans-serif",
              fontSize: "9px",
              fontWeight: 600,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              color: isDark ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.15)",
            }}
          >
            Scroll
          </span>
          <div
            className="w-px h-8 rounded-full"
            style={{
              background: isDark
                ? "linear-gradient(to bottom, rgba(255,255,255,0.15), transparent)"
                : "linear-gradient(to bottom, rgba(0,0,0,0.12), transparent)",
            }}
          />
        </motion.div>
      </motion.div>
    </section>
  );
}