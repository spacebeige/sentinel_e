import { motion } from "motion/react";
import { Activity, Cpu, Network, Shield, Zap, BarChart3, GitBranch, Globe, ArrowUpRight } from "lucide-react";
import { GlassPanel } from "./GlassPanel";
import { ConsensusMeter } from "./ConsensusMeter";

const STREAMS = [
  { id: "s1", session: "#4821", query: "Causal inference analysis",       model: "GPT-5o + Claude 4", phase: "Synthesis",  progress: 82, color: "#6ee7f9" },
  { id: "s2", session: "#4820", query: "Climate policy deliberation",     model: "Ensemble × 4",      phase: "Evidence",   progress: 61, color: "#34d399" },
  { id: "s3", session: "#4819", query: "Adversarial stress test",         model: "Mistral + Gemini",  phase: "Debate",     progress: 47, color: "#f59e0b" },
  { id: "s4", session: "#4818", query: "Governance boundary probe",       model: "Claude 4",          phase: "Governance", progress: 94, color: "#8b5cf6" },
  { id: "s5", session: "#4817", query: "Semantic routing optimisation",   model: "GPT-5o",            phase: "Routing",    progress: 33, color: "#6ee7f9" },
];

const TOP_METRICS = [
  { label: "Active Sessions",     value: "248",   delta: "+12",   color: "#6ee7f9", icon: Activity  },
  { label: "Tokens / sec",         value: "94.2K", delta: "+5.3K", color: "#34d399", icon: Zap       },
  { label: "Consensus Rate",       value: "96.1%", delta: "+0.4%", color: "#8b5cf6", icon: BarChart3 },
  { label: "Governance Alerts",    value: "3",     delta: "-11",   color: "#f59e0b", icon: Shield    },
  { label: "Orchestration Nodes",  value: "18",    delta: "+2",    color: "#6ee7f9", icon: Network   },
  { label: "Avg. Response (ms)",   value: "174",   delta: "-23",   color: "#34d399", icon: Cpu       },
];

const MODEL_LOADS = [
  { name: "GPT-5o",          load: 82, region: "US-East",    color: "#6ee7f9" },
  { name: "Claude Opus 4",   load: 67, region: "EU-West",    color: "#8b5cf6" },
  { name: "Gemini 3 Flash",  load: 91, region: "Asia-Pac",   color: "#34d399" },
  { name: "Mistral Large",   load: 54, region: "US-West",    color: "#f59e0b" },
  { name: "Llama 3.3 70B",   load: 38, region: "US-East",    color: "#6ee7f9" },
];

const ROUTING_EVENTS = [
  { time: "14:32:11", event: "Query routed → GPT-5o + Claude",   type: "route"    },
  { time: "14:32:09", event: "Consensus threshold reached (91%)", type: "success"  },
  { time: "14:32:06", event: "Adversarial probe triggered",       type: "warning"  },
  { time: "14:31:58", event: "Evidence cache hit × 6",            type: "info"     },
  { time: "14:31:54", event: "New session #4821 opened",          type: "route"    },
  { time: "14:31:47", event: "Governance flag resolved",          type: "success"  },
];

const typeColors: Record<string, string> = {
  route:   "#6ee7f9",
  success: "#34d399",
  warning: "#f59e0b",
  info:    "#8a9099",
};

export function MissionControlPage() {
  return (
    <div className="min-h-screen pt-24 pb-16 px-6">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <div className="w-6 h-6 rounded-md bg-[rgba(110,231,249,0.1)] border border-[rgba(110,231,249,0.2)] flex items-center justify-center">
                <Activity className="w-3 h-3 text-[#6ee7f9]" />
              </div>
              <span className="text-[#6ee7f9] text-[10px] font-medium tracking-[0.2em] uppercase">Mission Control</span>
            </div>
            <h1 className="text-[clamp(22px,3vw,36px)] font-bold text-[#f3f5f7] tracking-tight">
              Orchestration Operations Center
            </h1>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[rgba(52,211,153,0.08)] border border-[rgba(52,211,153,0.15)]">
            <motion.div className="w-2 h-2 rounded-full bg-[#34d399]" animate={{ opacity: [1, 0.3, 1] }} transition={{ duration: 1.4, repeat: Infinity }} />
            <span className="text-[#34d399] text-xs font-medium tracking-widest uppercase">All Systems Nominal</span>
          </div>
        </div>

        {/* Metrics grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
          {TOP_METRICS.map(({ label, value, delta, color, icon: Icon }, i) => (
            <motion.div key={label} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
              <GlassPanel className="p-4">
                <Icon className="w-3.5 h-3.5 mb-2" style={{ color }} />
                <div className="text-2xl font-bold tracking-tight mb-0.5" style={{ color }}>{value}</div>
                <div className="text-[#8a9099] text-[10px] leading-tight mb-1">{label}</div>
                <div className="text-[10px] text-[#34d399] font-medium">{delta}</div>
              </GlassPanel>
            </motion.div>
          ))}
        </div>

        {/* Main 3-column layout */}
        <div className="grid lg:grid-cols-3 gap-5">

          {/* Left: Active cognition streams */}
          <div className="lg:col-span-2 space-y-3">
            <div className="flex items-center justify-between mb-1">
              <h3 className="text-[#f3f5f7] text-sm font-semibold">Active Cognition Streams</h3>
              <span className="text-[#8a9099] text-xs">{STREAMS.length} running</span>
            </div>
            {STREAMS.map(({ id, session, query, model, phase, progress, color }, i) => (
              <motion.div key={id} initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.07 }}>
                <GlassPanel className="p-4 hover:translate-y-[-1px] transition-transform duration-200">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-[10px] font-mono" style={{ color }}>{session}</span>
                        <span className="text-[10px] px-2 py-0.5 rounded-full font-medium" style={{ background: `${color}12`, color }}>
                          {phase}
                        </span>
                      </div>
                      <p className="text-[#c7cbd1] text-sm truncate">{query}</p>
                      <p className="text-[#8a9099] text-[11px] mt-0.5">{model}</p>
                    </div>
                    <div className="text-right shrink-0">
                      <div className="text-sm font-semibold" style={{ color }}>{progress}%</div>
                      <div className="text-[10px] text-[#8a9099]">complete</div>
                    </div>
                  </div>
                  <div className="mt-3 h-0.5 rounded-full bg-[rgba(255,255,255,0.06)]">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ background: color }}
                      initial={{ width: 0 }}
                      animate={{ width: `${progress}%` }}
                      transition={{ duration: 1.2, delay: i * 0.1 }}
                    />
                  </div>
                </GlassPanel>
              </motion.div>
            ))}

            {/* Topology map */}
            <div className="mt-4">
              <h3 className="text-[#f3f5f7] text-sm font-semibold mb-3">Orchestration Topology</h3>
              <GlassPanel className="p-5" glow="cyan">
                <div className="relative h-48">
                  {/* Backbone lines */}
                  <svg className="absolute inset-0 w-full h-full" viewBox="0 0 400 200">
                    {[
                      [200,100, 80,40], [200,100, 320,40], [200,100, 80,160],
                      [200,100, 320,160],[200,100, 50,100],[200,100, 350,100],
                    ].map(([x1,y1,x2,y2], i) => (
                      <motion.line
                        key={i}
                        x1={x1} y1={y1} x2={x2} y2={y2}
                        stroke="rgba(110,231,249,0.15)" strokeWidth="1"
                        initial={{ pathLength: 0, opacity: 0 }}
                        animate={{ pathLength: 1, opacity: 1 }}
                        transition={{ duration: 1, delay: i * 0.15 }}
                      />
                    ))}
                  </svg>
                  {/* Central node */}
                  <motion.div
                    className="absolute w-14 h-14 rounded-full bg-[rgba(110,231,249,0.1)] border border-[rgba(110,231,249,0.35)] flex items-center justify-center"
                    style={{ left: "50%", top: "50%", transform: "translate(-50%,-50%)" }}
                    animate={{ boxShadow: ["0 0 0px #6ee7f900", "0 0 30px #6ee7f930", "0 0 0px #6ee7f900"] }}
                    transition={{ duration: 2.8, repeat: Infinity }}
                  >
                    <Network className="w-5 h-5 text-[#6ee7f9]" />
                  </motion.div>
                  {/* Peripheral nodes */}
                  {[
                    { label:"GPT-5o", x:"20%", y:"20%", color:"#6ee7f9" },
                    { label:"CLD-4",  x:"80%", y:"20%", color:"#8b5cf6" },
                    { label:"GEM-3",  x:"20%", y:"80%", color:"#34d399" },
                    { label:"MST-L",  x:"80%", y:"80%", color:"#f59e0b" },
                    { label:"GOV",    x:"10%", y:"50%", color:"#6ee7f9" },
                    { label:"SYNC",   x:"90%", y:"50%", color:"#34d399" },
                  ].map(({ label, x, y, color }, i) => (
                    <motion.div
                      key={label}
                      className="absolute flex flex-col items-center gap-1"
                      style={{ left: x, top: y, transform: "translate(-50%,-50%)" }}
                      initial={{ opacity: 0, scale: 0.6 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.5 + i * 0.1 }}
                    >
                      <div
                        className="w-8 h-8 rounded-full flex items-center justify-center text-[8px] font-bold"
                        style={{ background: `${color}12`, border: `1px solid ${color}30`, color }}
                      >
                        {label.slice(0, 2)}
                      </div>
                      <span className="text-[8px] text-[#8a9099]">{label}</span>
                    </motion.div>
                  ))}
                </div>
              </GlassPanel>
            </div>
          </div>

          {/* Right column */}
          <div className="space-y-4">
            {/* Model load */}
            <GlassPanel className="p-4">
              <h4 className="text-[#8a9099] text-[9px] tracking-[0.2em] uppercase font-medium mb-3">Model Load</h4>
              {MODEL_LOADS.map(({ name, load, region, color }) => (
                <div key={name} className="mb-3">
                  <div className="flex items-center justify-between text-[11px] mb-1">
                    <div>
                      <span className="text-[#c7cbd1]">{name}</span>
                      <span className="text-[#8a9099] ml-2">{region}</span>
                    </div>
                    <span style={{ color }} className="font-semibold">{load}%</span>
                  </div>
                  <div className="h-1 rounded-full bg-[rgba(255,255,255,0.06)]">
                    <motion.div className="h-full rounded-full" style={{ background: color }}
                      initial={{ width: 0 }} animate={{ width: `${load}%` }} transition={{ duration: 1 }} />
                  </div>
                </div>
              ))}
            </GlassPanel>

            {/* Live event log */}
            <GlassPanel className="p-4">
              <h4 className="text-[#8a9099] text-[9px] tracking-[0.2em] uppercase font-medium mb-3">Routing Events</h4>
              <div className="space-y-2">
                {ROUTING_EVENTS.map(({ time, event, type }, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: 8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.08 }}
                    className="flex items-start gap-2"
                  >
                    <div className="w-1 h-1 rounded-full mt-1.5 shrink-0" style={{ background: typeColors[type] }} />
                    <div className="flex-1 min-w-0">
                      <p className="text-[11px] text-[#c7cbd1] leading-snug truncate">{event}</p>
                      <p className="text-[9px] font-mono text-[#8a9099]">{time}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </GlassPanel>

            {/* Global consensus */}
            <GlassPanel className="p-4" glow="cyan">
              <h4 className="text-[#8a9099] text-[9px] tracking-[0.2em] uppercase font-medium mb-4">Global Metrics</h4>
              <div className="flex justify-around">
                <ConsensusMeter value={96} color="cyan"    size="sm" label="Consensus" />
                <ConsensusMeter value={8}  color="amber"   size="sm" label="Alerts"    />
                <ConsensusMeter value={99} color="emerald" size="sm" label="Uptime"    />
              </div>
            </GlassPanel>
          </div>
        </div>
      </div>
    </div>
  );
}
