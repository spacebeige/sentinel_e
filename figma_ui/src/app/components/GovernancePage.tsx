import { motion } from "framer-motion";
import { Shield, AlertTriangle, CheckCircle2, Eye, Lock, Zap, BarChart3, GitBranch, Search } from "lucide-react";
import { GlassPanel } from "./GlassPanel";
import { ConsensusMeter } from "./ConsensusMeter";

const AUDIT_ENTRIES = [
  { id: "G-4821", session: "#4821", type: "Hallucination Probe",    result: "PASS",    severity: "none",   model: "GPT-5o",        time: "14:32:11" },
  { id: "G-4820", session: "#4820", type: "Semantic Boundary",      result: "PASS",    severity: "none",   model: "Claude 4",      time: "14:31:58" },
  { id: "G-4819", session: "#4819", type: "Adversarial Stress",     result: "FLAG",    severity: "medium", model: "Gemini 3",      time: "14:31:47" },
  { id: "G-4818", session: "#4818", type: "Evidence Integrity",     result: "PASS",    severity: "none",   model: "Mistral L",     time: "14:31:32" },
  { id: "G-4817", session: "#4817", type: "Drift Detection",        result: "FLAG",    severity: "low",    model: "GPT-5o",        time: "14:30:58" },
  { id: "G-4816", session: "#4816", type: "Logical Consistency",    result: "PASS",    severity: "none",   model: "Ensemble",      time: "14:30:44" },
  { id: "G-4815", session: "#4815", type: "Factual Grounding",      result: "FAIL",    severity: "high",   model: "Llama 3.3",     time: "14:29:21" },
];

const GOVERNANCE_METRICS = [
  { label: "Hallucination Rate",    value: "0.4%",   status: "nominal",  trend: "down"  },
  { label: "Boundary Violations",   value: "2",      status: "nominal",  trend: "down"  },
  { label: "Adversarial Flags",     value: "7",      status: "elevated", trend: "up"    },
  { label: "Evidence Pass Rate",    value: "97.8%",  status: "nominal",  trend: "up"    },
  { label: "Integrity Score",       value: "9.4/10", status: "nominal",  trend: "stable"},
  { label: "Drift Index",           value: "0.12",   status: "nominal",  trend: "down"  },
];

const INTEGRITY_LAYERS = [
  { layer: "Hallucination Gate",    desc: "Detects fabricated facts via cross-reference sampling",         pass: true  },
  { layer: "Semantic Boundary",     desc: "Enforces topical scope and detects off-domain generation",      pass: true  },
  { layer: "Adversarial Probe",     desc: "Stress-tests responses against known attack vectors",           pass: false },
  { layer: "Evidence Validation",   desc: "Verifies claim-to-source traceability chain",                   pass: true  },
  { layer: "Confidence Calibration",desc: "Ensures expressed confidence aligns with evidence strength",    pass: true  },
  { layer: "Drift Monitor",         desc: "Tracks semantic drift across multi-turn sessions",              pass: true  },
];

const RESULT_STYLES: Record<string, { color: string; bg: string }> = {
  PASS: { color: "#34d399", bg: "rgba(52,211,153,0.1)"  },
  FLAG: { color: "#f59e0b", bg: "rgba(245,158,11,0.1)"  },
  FAIL: { color: "#ef4444", bg: "rgba(239,68,68,0.1)"   },
};

const SEVERITY_COLORS: Record<string, string> = {
  none:   "transparent",
  low:    "#f59e0b",
  medium: "#f97316",
  high:   "#ef4444",
};

export function GovernancePage() {
  return (
    <div className="min-h-screen pt-24 pb-16 px-6">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-10">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <div className="w-6 h-6 rounded-md bg-[rgba(139,92,246,0.1)] border border-[rgba(139,92,246,0.2)] flex items-center justify-center">
                <Shield className="w-3 h-3 text-[#8b5cf6]" />
              </div>
              <span className="text-[#8b5cf6] text-[10px] font-medium tracking-[0.2em] uppercase">Governance &amp; Forensics</span>
            </div>
            <h1 className="text-[clamp(22px,3vw,36px)] font-bold text-[#f3f5f7] tracking-tight">
              AI Safety Verification Layer
            </h1>
            <p className="text-[#8a9099] text-sm mt-1">Continuous forensic analysis across all active sessions</p>
          </div>
          <div className="flex items-center gap-4">
            <ConsensusMeter value={97} color="emerald" size="md" label="Integrity" />
            <ConsensusMeter value={4}  color="amber"   size="md" label="Alerts"   />
          </div>
        </div>

        {/* Top metrics */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-8">
          {GOVERNANCE_METRICS.map(({ label, value, status, trend }, i) => (
            <motion.div key={label} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.06 }}>
              <GlassPanel className="p-4" glow={status === "elevated" ? "amber" : "none"}>
                <div className={`text-xl font-bold tracking-tight mb-0.5 ${
                  status === "elevated" ? "text-[#f59e0b]" : "text-[#f3f5f7]"
                }`}>{value}</div>
                <div className="text-[#8a9099] text-[10px] leading-tight mb-1">{label}</div>
                <div className={`text-[9px] font-medium ${
                  trend === "up" && status === "elevated" ? "text-[#f59e0b]"
                  : trend === "up" ? "text-[#34d399]"
                  : trend === "down" ? "text-[#34d399]"
                  : "text-[#8a9099]"
                }`}>
                  {trend === "up" ? "↑" : trend === "down" ? "↓" : "→"} {trend}
                </div>
              </GlassPanel>
            </motion.div>
          ))}
        </div>

        {/* Two-column layout */}
        <div className="grid lg:grid-cols-3 gap-5">

          {/* Left: audit log */}
          <div className="lg:col-span-2">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-[#f3f5f7] text-sm font-semibold">Forensic Audit Log</h3>
              <div className="flex items-center gap-1.5 px-3 py-1 rounded-lg border border-[rgba(255,255,255,0.07)] text-[#8a9099] text-xs">
                <Search className="w-3 h-3" />
                Filter
              </div>
            </div>

            {/* Table header */}
            <div className="grid grid-cols-6 gap-2 px-4 pb-2 mb-1">
              {["ID", "Session", "Check Type", "Model", "Time", "Result"].map((h) => (
                <div key={h} className="text-[9px] tracking-[0.15em] uppercase font-medium text-[#8a9099]">{h}</div>
              ))}
            </div>

            <div className="space-y-1.5">
              {AUDIT_ENTRIES.map(({ id, session, type, result, severity, model, time }, i) => {
                const rs = RESULT_STYLES[result];
                return (
                  <motion.div
                    key={id}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.06 }}
                  >
                    <GlassPanel className={`px-4 py-2.5 ${severity !== "none" ? "border-l-2" : ""}`}
                      style={{ borderLeftColor: severity !== "none" ? SEVERITY_COLORS[severity] : undefined } as React.CSSProperties}
                    >
                      <div className="grid grid-cols-6 gap-2 items-center">
                        <span className="text-[11px] font-mono text-[#8a9099]">{id}</span>
                        <span className="text-[11px] text-[#6ee7f9] font-mono">{session}</span>
                        <span className="text-[11px] text-[#c7cbd1] col-span-1">{type}</span>
                        <span className="text-[11px] text-[#8a9099]">{model}</span>
                        <span className="text-[10px] font-mono text-[#8a9099]">{time}</span>
                        <span className="inline-flex items-center gap-1">
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold tracking-wide" style={{ background: rs.bg, color: rs.color }}>
                            {result}
                          </span>
                        </span>
                      </div>
                    </GlassPanel>
                  </motion.div>
                );
              })}
            </div>

            {/* Hallucination detection viz */}
            <div className="mt-6">
              <h3 className="text-[#f3f5f7] text-sm font-semibold mb-3">Hallucination Detection — 24h Window</h3>
              <GlassPanel className="p-5">
                <div className="flex items-end gap-2 h-28">
                  {Array.from({ length: 24 }, (_, i) => {
                    const h = 10 + Math.random() * 60;
                    const flagged = Math.random() > 0.85;
                    return (
                      <motion.div
                        key={i}
                        className="flex-1 rounded-sm"
                        style={{ height: `${h}%`, background: flagged ? "rgba(239,68,68,0.4)" : "rgba(110,231,249,0.2)" }}
                        initial={{ scaleY: 0, originY: 1 }}
                        animate={{ scaleY: 1 }}
                        transition={{ duration: 0.5, delay: i * 0.03 }}
                      />
                    );
                  })}
                </div>
                <div className="flex items-center gap-4 mt-3 pt-3 border-t border-[rgba(255,255,255,0.05)]">
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-sm bg-[rgba(110,231,249,0.4)]" />
                    <span className="text-[10px] text-[#8a9099]">Normal</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-sm bg-[rgba(239,68,68,0.5)]" />
                    <span className="text-[10px] text-[#8a9099]">Flagged</span>
                  </div>
                  <span className="ml-auto text-[10px] text-[#8a9099]">Hourly buckets</span>
                </div>
              </GlassPanel>
            </div>
          </div>

          {/* Right: integrity layers */}
          <div className="space-y-4">
            <h3 className="text-[#f3f5f7] text-sm font-semibold">Verification Stack</h3>
            <div className="space-y-2">
              {INTEGRITY_LAYERS.map(({ layer, desc, pass }, i) => (
                <motion.div key={layer} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.08 }}>
                  <GlassPanel className={`p-3.5 ${!pass ? "border-l-2 border-[#f59e0b]" : ""}`} glow={!pass ? "amber" : "none"}>
                    <div className="flex items-start gap-3">
                      <div className={`mt-0.5 w-5 h-5 rounded flex items-center justify-center shrink-0 ${pass ? "bg-[rgba(52,211,153,0.12)]" : "bg-[rgba(245,158,11,0.12)]"}`}>
                        {pass
                          ? <CheckCircle2 className="w-3 h-3 text-[#34d399]" />
                          : <AlertTriangle className="w-3 h-3 text-[#f59e0b]" />
                        }
                      </div>
                      <div>
                        <p className={`text-xs font-semibold mb-0.5 ${pass ? "text-[#f3f5f7]" : "text-[#f59e0b]"}`}>{layer}</p>
                        <p className="text-[#8a9099] text-[11px] leading-relaxed">{desc}</p>
                      </div>
                    </div>
                  </GlassPanel>
                </motion.div>
              ))}
            </div>

            {/* Risk profile */}
            <GlassPanel className="p-4" glow="violet">
              <p className="text-[#8a9099] text-[9px] tracking-[0.2em] uppercase font-medium mb-3">Risk Profile</p>
              {[
                { label: "Factual Risk",      value: 4,  color: "#34d399" },
                { label: "Adversarial Risk",  value: 18, color: "#f59e0b" },
                { label: "Drift Risk",        value: 9,  color: "#8b5cf6" },
                { label: "Boundary Risk",     value: 6,  color: "#6ee7f9" },
              ].map(({ label, value, color }) => (
                <div key={label} className="mb-2.5">
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-[#8a9099]">{label}</span>
                    <span style={{ color }}>{value}%</span>
                  </div>
                  <div className="h-1 rounded-full bg-[rgba(255,255,255,0.06)]">
                    <motion.div className="h-full rounded-full" style={{ background: color }}
                      initial={{ width: 0 }} whileInView={{ width: `${value}%` }} viewport={{ once: true }} transition={{ duration: 0.8 }} />
                  </div>
                </div>
              ))}
            </GlassPanel>
          </div>
        </div>
      </div>
    </div>
  );
}
