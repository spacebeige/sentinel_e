import { Link } from "react-router";
import { motion } from "framer-motion";
import {
  ArrowRight, Brain, Shield, Activity, Cpu, Swords,
  GitBranch, Eye, Network, CheckCircle2, ChevronRight,
  Zap, Lock, BarChart3, Globe,
} from "lucide-react";
import { GlassPanel } from "./GlassPanel";
import { ConsensusMeter } from "./ConsensusMeter";

// ─── Fade-up animation helper ───────────────────────────────────────────────
const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 24 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true },
  transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1], delay },
});

// ─── Animated council node ──────────────────────────────────────────────────
function CouncilNode({
  label, role, color, x, y, delay,
}: { label: string; role: string; color: string; x: string; y: string; delay: number }) {
  const colors: Record<string, { ring: string; bg: string; dot: string }> = {
    cyan:    { ring: "rgba(110,231,249,0.25)", bg: "rgba(110,231,249,0.08)", dot: "#6ee7f9" },
    emerald: { ring: "rgba(52,211,153,0.25)",  bg: "rgba(52,211,153,0.08)",  dot: "#34d399" },
    violet:  { ring: "rgba(139,92,246,0.25)",  bg: "rgba(139,92,246,0.08)",  dot: "#8b5cf6" },
    amber:   { ring: "rgba(245,158,11,0.25)",  bg: "rgba(245,158,11,0.08)",  dot: "#f59e0b" },
  };
  const c = colors[color] ?? colors.cyan;
  return (
    <motion.div
      className="absolute flex flex-col items-center gap-1.5"
      style={{ left: x, top: y, transform: "translate(-50%,-50%)" }}
      initial={{ opacity: 0, scale: 0.7 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      <motion.div
        className="w-12 h-12 rounded-full flex items-center justify-center text-xs font-bold"
        style={{ background: c.bg, border: `1px solid ${c.ring}`, color: c.dot }}
        animate={{ boxShadow: [`0 0 0px ${c.dot}00`, `0 0 18px ${c.dot}40`, `0 0 0px ${c.dot}00`] }}
        transition={{ duration: 2.8, repeat: Infinity, delay }}
      >
        {label}
      </motion.div>
      <span className="text-[10px] text-[#8a9099] tracking-wider">{role}</span>
    </motion.div>
  );
}

// ─── Section label ────────────────────────────────────────────────────────────
function SectionLabel({ children }: { children: string }) {
  return (
    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-[rgba(110,231,249,0.15)] bg-[rgba(110,231,249,0.05)] mb-5">
      <div className="w-1 h-1 rounded-full bg-[#6ee7f9]" />
      <span className="text-[#6ee7f9] text-[10px] font-medium tracking-[0.2em] uppercase">{children}</span>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
export function HomePage() {
  return (
    <div className="min-h-screen">
      <HeroSection />
      <CouncilSection />
      <CapabilitiesSection />
      <ReasoningSection />
      <GovernanceSection />
      <MissionPreviewSection />
      <EnterpriseSection />
      <FooterSection />
    </div>
  );
}

// ─── 1. HERO ──────────────────────────────────────────────────────────────────
function HeroSection() {
  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-28 pb-20 overflow-hidden">
      {/* Atmospheric glow */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] rounded-full bg-[rgba(110,231,249,0.03)] blur-[120px] pointer-events-none" />

      <div className="relative z-10 max-w-5xl mx-auto text-center">
        <motion.div {...fadeUp(0.1)} className="flex justify-center mb-8">
          <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-[rgba(110,231,249,0.18)] bg-[rgba(110,231,249,0.05)]">
            <div className="w-1.5 h-1.5 rounded-full bg-[#34d399] animate-pulse" />
            <span className="text-[#34d399] text-[11px] font-medium tracking-[0.2em] uppercase">Collective Machine Cognition · Active</span>
          </div>
        </motion.div>

        <motion.h1
          {...fadeUp(0.2)}
          className="text-[clamp(48px,8vw,96px)] font-bold leading-[0.95] tracking-[-0.03em] text-[#f3f5f7] mb-6 text-balance"
        >
          Visible Machine
          <br />
          <span className="text-[#6ee7f9]">Cognition</span>
        </motion.h1>

        <motion.p
          {...fadeUp(0.35)}
          className="text-[clamp(15px,2vw,19px)] text-[#8a9099] max-w-2xl mx-auto mb-10 leading-relaxed font-light"
        >
          Sentinel-E orchestrates multi-agent AI deliberation in real time — surfacing consensus,
          conflict, and semantic reasoning across an ensemble of cognitive systems.
        </motion.p>

        <motion.div {...fadeUp(0.45)} className="flex flex-wrap justify-center gap-3">
          <Link
            to="/chat"
            className="inline-flex items-center gap-2 px-7 py-3 rounded-lg bg-[#6ee7f9] text-[#060708] font-semibold text-sm tracking-wide hover:bg-[rgba(110,231,249,0.85)] transition-all hover:scale-[1.02]"
          >
            Enter the Runtime
            <ArrowRight className="w-4 h-4" />
          </Link>
          <Link
            to="/debate"
            className="inline-flex items-center gap-2 px-7 py-3 rounded-lg border border-[rgba(110,231,249,0.18)] bg-[rgba(110,231,249,0.05)] text-[#c7cbd1] font-medium text-sm tracking-wide hover:text-[#f3f5f7] hover:bg-[rgba(110,231,249,0.08)] transition-all"
          >
            Watch Deliberation
          </Link>
        </motion.div>

        {/* Live orchestration preview strip */}
        <motion.div {...fadeUp(0.6)} className="mt-20 max-w-3xl mx-auto">
          <GlassPanel className="p-4" glow="cyan">
            <div className="flex items-center gap-3 mb-3 pb-3 border-b border-[rgba(110,231,249,0.08)]">
              <div className="flex gap-1.5">
                {["#6ee7f9","#34d399","#f59e0b"].map((c) => (
                  <div key={c} className="w-2 h-2 rounded-full" style={{ background: c, opacity: 0.8 }} />
                ))}
              </div>
              <span className="text-[#8a9099] text-[10px] font-mono tracking-wider">sentinel-e / orchestration-runtime · session #4821</span>
              <div className="ml-auto flex items-center gap-1.5">
                <motion.div
                  className="w-1.5 h-1.5 rounded-full bg-[#34d399]"
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ duration: 1.4, repeat: Infinity }}
                />
                <span className="text-[#34d399] text-[10px]">deliberating</span>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <div className="flex-1 space-y-2">
                {[
                  { model: "GPT-5o", msg: "The primary vector is semantic coherence across token boundaries...", color: "#6ee7f9" },
                  { model: "Claude-4", msg: "Divergence detected on causal attribution. Evidence weight: 0.73.", color: "#8b5cf6" },
                  { model: "Gemini-3", msg: "Synthesising via orthogonal reasoning path — consensus threshold reached.", color: "#34d399" },
                ].map(({ model, msg, color }) => (
                  <div key={model} className="flex items-start gap-2.5">
                    <div className="mt-0.5 w-5 h-5 rounded flex items-center justify-center text-[8px] font-bold shrink-0" style={{ background: `${color}15`, border: `1px solid ${color}30`, color }}>
                      {model.slice(0, 2)}
                    </div>
                    <p className="text-[11px] text-[#8a9099] leading-relaxed">{msg}</p>
                  </div>
                ))}
              </div>
              <div className="flex flex-col gap-3 items-center shrink-0">
                <ConsensusMeter value={87} color="cyan" size="sm" label="Consensus" />
                <ConsensusMeter value={34} color="amber" size="sm" label="Conflict" />
              </div>
            </div>
          </GlassPanel>
        </motion.div>
      </div>
    </section>
  );
}

// ─── 2. AI COUNCIL ────────────────────────────────────────────────────────────
function CouncilSection() {
  return (
    <section className="relative py-32 px-6 overflow-hidden">
      <div className="max-w-6xl mx-auto">
        <motion.div {...fadeUp(0)} className="text-center mb-16">
          <SectionLabel>AI Council Chamber</SectionLabel>
          <h2 className="text-[clamp(32px,5vw,56px)] font-bold tracking-[-0.025em] text-[#f3f5f7] leading-tight text-balance">
            An ensemble deliberates
            <br />
            <span className="text-[#8a9099] font-light">so you see every angle</span>
          </h2>
        </motion.div>

        {/* Council visualisation */}
        <motion.div {...fadeUp(0.2)} className="relative h-[340px] mb-16">
          <GlassPanel className="absolute inset-0" glow="cyan">
            {/* Centre pulse */}
            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
              <motion.div
                className="w-16 h-16 rounded-full border border-[rgba(110,231,249,0.3)] flex items-center justify-center"
                animate={{ scale: [1, 1.08, 1] }}
                transition={{ duration: 3, repeat: Infinity }}
              >
                <motion.div
                  className="w-10 h-10 rounded-full bg-[rgba(110,231,249,0.1)] border border-[rgba(110,231,249,0.4)] flex items-center justify-center"
                  animate={{ boxShadow: ["0 0 0px #6ee7f900", "0 0 28px #6ee7f940", "0 0 0px #6ee7f900"] }}
                  transition={{ duration: 2.4, repeat: Infinity }}
                >
                  <Brain className="w-4 h-4 text-[#6ee7f9]" />
                </motion.div>
              </motion.div>
            </div>
            <CouncilNode label="GPT" role="Synthesis"  color="cyan"    x="20%"  y="30%" delay={0.1} />
            <CouncilNode label="CLN" role="Critique"   color="violet"  x="80%"  y="30%" delay={0.2} />
            <CouncilNode label="GEM" role="Evidence"   color="emerald" x="20%"  y="70%" delay={0.3} />
            <CouncilNode label="MIS" role="Governance" color="amber"   x="80%"  y="70%" delay={0.4} />
            <CouncilNode label="LLM" role="Reasoning"  color="cyan"    x="50%"  y="12%" delay={0.5} />
          </GlassPanel>
        </motion.div>

        {/* Stats row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { value: "12+",   label: "Active Models",     color: "text-[#6ee7f9]" },
            { value: "97%",   label: "Consensus Rate",    color: "text-[#34d399]" },
            { value: "<180ms",label: "Orchestration Lag", color: "text-[#f59e0b]" },
            { value: "∞",     label: "Reasoning Depth",  color: "text-[#8b5cf6]" },
          ].map(({ value, label, color }) => (
            <motion.div key={label} {...fadeUp(0.1)}>
              <GlassPanel className="p-5 text-center">
                <div className={`text-3xl font-bold tracking-tight mb-1 ${color}`}>{value}</div>
                <div className="text-[#8a9099] text-xs tracking-wide">{label}</div>
              </GlassPanel>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ─── 3. CAPABILITIES ─────────────────────────────────────────────────────────
function CapabilitiesSection() {
  const cards = [
    { icon: Brain,      title: "Multi-Agent Deliberation", desc: "Multiple cognitive systems reason in parallel, forming and contesting positions in real time.", color: "cyan"    as const },
    { icon: Eye,        title: "Visible Reasoning",         desc: "Every inference step, confidence score, and semantic link is surfaced — nothing is hidden.",    color: "emerald" as const },
    { icon: Swords,     title: "Adversarial Debate",        desc: "AI models argue opposing positions. Conflict is engineered to stress-test every conclusion.",     color: "amber"   as const },
    { icon: GitBranch,  title: "Semantic Routing",          desc: "Queries are parsed, intent extracted, and dynamically routed to the optimal cognitive path.",     color: "violet"  as const },
    { icon: Shield,     title: "Governance Layer",          desc: "Continuous hallucination detection, boundary enforcement, and adversarial integrity checks.",     color: "cyan"    as const },
    { icon: Network,    title: "Topology Mapping",          desc: "Live semantic graph of agent relationships, evidence chains, and consensus emergence.",           color: "emerald" as const },
  ];
  return (
    <section className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div {...fadeUp(0)} className="text-center mb-14">
          <SectionLabel>Orchestration Capabilities</SectionLabel>
          <h2 className="text-[clamp(28px,4vw,48px)] font-bold tracking-[-0.025em] text-[#f3f5f7] text-balance">
            Every layer of machine cognition, visible
          </h2>
        </motion.div>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {cards.map(({ icon: Icon, title, desc, color }, i) => (
            <motion.div key={title} {...fadeUp(i * 0.07)}>
              <GlassPanel glow={color} className="p-6 h-full hover:translate-y-[-2px] transition-transform duration-300">
                <div className={`w-9 h-9 rounded-lg flex items-center justify-center mb-4 ${
                  color === "cyan"    ? "bg-[rgba(110,231,249,0.1)] text-[#6ee7f9]" :
                  color === "emerald" ? "bg-[rgba(52,211,153,0.1)]  text-[#34d399]" :
                  color === "violet"  ? "bg-[rgba(139,92,246,0.1)]  text-[#8b5cf6]" :
                                        "bg-[rgba(245,158,11,0.1)]  text-[#f59e0b]"
                }`}>
                  <Icon className="w-4 h-4" />
                </div>
                <h3 className="text-[#f3f5f7] font-semibold mb-2 text-sm tracking-tight">{title}</h3>
                <p className="text-[#8a9099] text-xs leading-relaxed">{desc}</p>
              </GlassPanel>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ─── 4. REASONING ────────────────────────────────────────────────────────────
function ReasoningSection() {
  const steps = [
    { phase: "01 · Ingest",     label: "Query parsed, intent extracted, context compiled",            color: "#6ee7f9", pct: 100 },
    { phase: "02 · Route",      label: "Semantic routing to optimal agent ensemble",                  color: "#8b5cf6", pct: 88  },
    { phase: "03 · Deliberate", label: "Multi-model parallel reasoning with adversarial stress-test", color: "#34d399", pct: 74  },
    { phase: "04 · Synthesise", label: "Cross-model consensus formation and conflict resolution",     color: "#f59e0b", pct: 91  },
    { phase: "05 · Govern",     label: "Hallucination detection, boundary enforcement, sign-off",     color: "#6ee7f9", pct: 97  },
  ];
  return (
    <section className="py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <motion.div {...fadeUp(0)} className="mb-14 max-w-xl">
          <SectionLabel>Reasoning Pipeline</SectionLabel>
          <h2 className="text-[clamp(28px,4vw,48px)] font-bold tracking-[-0.025em] text-[#f3f5f7] leading-tight text-balance">
            Five-phase cognition — fully observable
          </h2>
        </motion.div>
        <div className="space-y-3">
          {steps.map(({ phase, label, color, pct }, i) => (
            <motion.div key={phase} {...fadeUp(i * 0.09)}>
              <GlassPanel className="p-4 flex items-center gap-5">
                <div className="shrink-0 w-28">
                  <span className="text-[10px] font-semibold tracking-widest uppercase" style={{ color }}>{phase}</span>
                </div>
                <div className="flex-1">
                  <p className="text-[#c7cbd1] text-xs mb-2">{label}</p>
                  <div className="h-0.5 bg-[rgba(255,255,255,0.05)] rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ background: color }}
                      initial={{ width: 0 }}
                      whileInView={{ width: `${pct}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 1.1, delay: i * 0.1, ease: "easeOut" }}
                    />
                  </div>
                </div>
                <div className="shrink-0 text-xs font-semibold" style={{ color }}>{pct}%</div>
              </GlassPanel>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ─── 5. GOVERNANCE ───────────────────────────────────────────────────────────
function GovernanceSection() {
  const items = [
    { icon: Shield, title: "Hallucination Detection",  desc: "Forensic-grade verification at every inference step." },
    { icon: Lock,   title: "Boundary Enforcement",     desc: "Semantic and governance boundaries enforced in real time." },
    { icon: Eye,    title: "Adversarial Stress Testing",desc: "Each output challenged before it reaches you." },
    { icon: Zap,    title: "Live Integrity Scoring",   desc: "Continuous confidence and integrity metrics per response." },
  ];
  return (
    <section className="py-24 px-6">
      <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
        <motion.div {...fadeUp(0)}>
          <SectionLabel>Governance &amp; Forensics</SectionLabel>
          <h2 className="text-[clamp(28px,4vw,46px)] font-bold tracking-[-0.025em] text-[#f3f5f7] leading-tight mb-5 text-balance">
            Trustworthy AI starts with visible verification
          </h2>
          <p className="text-[#8a9099] text-sm leading-relaxed mb-8">
            Every response passes through a multi-layer governance pipeline —
            hallucination gates, semantic integrity checks, and adversarial probes — before
            being delivered.
          </p>
          <Link
            to="/governance"
            className="inline-flex items-center gap-2 text-[#6ee7f9] text-sm font-medium hover:gap-3 transition-all"
          >
            Explore Governance Layer <ChevronRight className="w-4 h-4" />
          </Link>
        </motion.div>
        <div className="grid grid-cols-2 gap-3">
          {items.map(({ icon: Icon, title, desc }, i) => (
            <motion.div key={title} {...fadeUp(i * 0.08)}>
              <GlassPanel className="p-4 h-full" glow="cyan">
                <Icon className="w-4 h-4 text-[#6ee7f9] mb-3" />
                <h4 className="text-[#f3f5f7] text-xs font-semibold mb-1">{title}</h4>
                <p className="text-[#8a9099] text-[11px] leading-relaxed">{desc}</p>
              </GlassPanel>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ─── 6. MISSION PREVIEW ──────────────────────────────────────────────────────
function MissionPreviewSection() {
  return (
    <section className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div {...fadeUp(0)} className="text-center mb-14">
          <SectionLabel>Mission Control</SectionLabel>
          <h2 className="text-[clamp(28px,4vw,48px)] font-bold tracking-[-0.025em] text-[#f3f5f7] text-balance">
            Orchestration operations at a glance
          </h2>
        </motion.div>
        <motion.div {...fadeUp(0.2)}>
          <GlassPanel className="p-6" glow="cyan">
            <div className="grid md:grid-cols-3 gap-6">
              {/* Left: metrics */}
              <div className="space-y-4">
                <h4 className="text-[#8a9099] text-[10px] tracking-widest uppercase font-medium mb-2">Cognition Streams</h4>
                {[
                  { label: "Active Sessions",    value: "248",   change: "+12",  color: "#6ee7f9" },
                  { label: "Tokens/sec",         value: "94.2K", change: "+5.3K",color: "#34d399" },
                  { label: "Consensus Rate",     value: "96.1%", change: "+0.4%",color: "#8b5cf6" },
                  { label: "Governance Alerts",  value: "3",     change: "-11",  color: "#f59e0b" },
                ].map(({ label, value, change, color }) => (
                  <div key={label} className="flex items-center justify-between border-b border-[rgba(255,255,255,0.04)] pb-3">
                    <span className="text-[#8a9099] text-xs">{label}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold" style={{ color }}>{value}</span>
                      <span className="text-[10px] text-[#34d399]">{change}</span>
                    </div>
                  </div>
                ))}
              </div>
              {/* Centre: topology map placeholder */}
              <div className="relative rounded-lg overflow-hidden border border-[rgba(110,231,249,0.08)] bg-[rgba(6,7,8,0.6)] flex items-center justify-center min-h-[200px]">
                <div className="absolute inset-0 opacity-20">
                  {Array.from({ length: 6 }).map((_, i) => (
                    <motion.div
                      key={i}
                      className="absolute w-2 h-2 rounded-full bg-[#6ee7f9]"
                      style={{
                        left: `${15 + i * 14}%`,
                        top: `${20 + (i % 3) * 25}%`,
                      }}
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 2, repeat: Infinity, delay: i * 0.35 }}
                    />
                  ))}
                </div>
                <div className="text-center z-10">
                  <Network className="w-8 h-8 text-[#6ee7f9] mx-auto mb-2 opacity-60" />
                  <p className="text-[#8a9099] text-xs">Topology Active</p>
                </div>
              </div>
              {/* Right: model activity */}
              <div className="space-y-3">
                <h4 className="text-[#8a9099] text-[10px] tracking-widest uppercase font-medium mb-2">Model Activity</h4>
                {[
                  { name: "GPT-5o",        load: 82, color: "#6ee7f9" },
                  { name: "Claude-4",      load: 67, color: "#8b5cf6" },
                  { name: "Gemini-3 Flash",load: 91, color: "#34d399" },
                  { name: "Mistral-L",     load: 54, color: "#f59e0b" },
                ].map(({ name, load, color }) => (
                  <div key={name}>
                    <div className="flex justify-between text-[11px] mb-1">
                      <span className="text-[#c7cbd1]">{name}</span>
                      <span style={{ color }}>{load}%</span>
                    </div>
                    <div className="h-1 rounded-full bg-[rgba(255,255,255,0.06)]">
                      <motion.div
                        className="h-full rounded-full"
                        style={{ background: color }}
                        initial={{ width: 0 }}
                        whileInView={{ width: `${load}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 1 }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </GlassPanel>
        </motion.div>
        <motion.div {...fadeUp(0.35)} className="text-center mt-6">
          <Link
            to="/mission-control"
            className="inline-flex items-center gap-2 text-[#6ee7f9] text-sm font-medium hover:gap-3 transition-all"
          >
            Open Mission Control <ChevronRight className="w-4 h-4" />
          </Link>
        </motion.div>
      </div>
    </section>
  );
}

// ─── 7. ENTERPRISE ───────────────────────────────────────────────────────────
function EnterpriseSection() {
  const features = [
    "SOC 2 Type II compliant infrastructure",
    "Private cloud and on-premise deployment",
    "Custom model integration and fine-tuning",
    "Dedicated governance policy engine",
    "Enterprise SLA with 99.9% uptime guarantee",
    "Audit logs and forensic trail export",
  ];
  return (
    <section className="py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <GlassPanel className="p-10 md:p-14" glow="violet">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <SectionLabel>Enterprise Orchestration</SectionLabel>
              <h2 className="text-[clamp(26px,3.5vw,42px)] font-bold tracking-[-0.025em] text-[#f3f5f7] leading-tight mb-5 text-balance">
                Built for the most demanding intelligence operations
              </h2>
              <p className="text-[#8a9099] text-sm leading-relaxed mb-8">
                Sentinel-E scales from a single researcher to a global enterprise with
                hundreds of concurrent orchestration sessions, governed by a dedicated safety layer.
              </p>
              <div className="flex flex-wrap gap-3">
                <Link
                  to="/pricing"
                  className="px-6 py-2.5 rounded-lg bg-[#6ee7f9] text-[#060708] text-sm font-semibold tracking-wide hover:bg-[rgba(110,231,249,0.85)] transition-colors"
                >
                  View Enterprise Plans
                </Link>
                <Link
                  to="/governance"
                  className="px-6 py-2.5 rounded-lg border border-[rgba(110,231,249,0.18)] text-[#c7cbd1] text-sm font-medium hover:text-[#f3f5f7] hover:bg-[rgba(110,231,249,0.05)] transition-colors"
                >
                  Security Overview
                </Link>
              </div>
            </div>
            <div className="space-y-3">
              {features.map((f) => (
                <div key={f} className="flex items-center gap-3">
                  <CheckCircle2 className="w-4 h-4 text-[#34d399] shrink-0" />
                  <span className="text-[#c7cbd1] text-sm">{f}</span>
                </div>
              ))}
            </div>
          </div>
        </GlassPanel>
      </div>
    </section>
  );
}

// ─── 8. FOOTER ───────────────────────────────────────────────────────────────
function FooterSection() {
  return (
    <footer className="border-t border-[rgba(110,231,249,0.07)] py-12 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2.5">
            <div className="w-6 h-6 rounded-md bg-[rgba(110,231,249,0.1)] border border-[rgba(110,231,249,0.2)] flex items-center justify-center">
              <BarChart3 className="w-3 h-3 text-[#6ee7f9]" />
            </div>
            <span className="text-[#f3f5f7] font-semibold text-sm tracking-tight">SENTINEL-E</span>
            <span className="text-[#8a9099] text-xs ml-1">Visible Machine Cognition</span>
          </div>
          <div className="flex items-center gap-6">
            {[
              { to: "/chat",            label: "Deliberation"    },
              { to: "/debate",          label: "Debate"          },
              { to: "/mission-control", label: "Mission Control" },
              { to: "/governance",      label: "Governance"      },
              { to: "/models",          label: "Models"          },
              { to: "/pricing",         label: "Pricing"         },
            ].map(({ to, label }) => (
              <Link key={to} to={to} className="text-[#8a9099] text-xs hover:text-[#c7cbd1] transition-colors">
                {label}
              </Link>
            ))}
          </div>
          <div className="flex items-center gap-1.5">
            <Globe className="w-3.5 h-3.5 text-[#8a9099]" />
            <span className="text-[#8a9099] text-xs">© 2025 Sentinel-E</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
