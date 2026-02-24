import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Loader2,
  ChevronRight,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Layers,
  Scan,
} from "lucide-react";
import {
  runCrossAnalysis,
  type CrossAnalysisResult,
  type CrossAnalysisModelProfile,
} from "../api";

interface CrossAnalysisTriggerProps {
  chatId: string | null;
  messageContent: string;
  backendOnline: boolean | null;
}

export function CrossAnalysisTrigger({
  chatId,
  messageContent,
  backendOnline,
}: CrossAnalysisTriggerProps) {
  const [result, setResult] = useState<CrossAnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  const handleRun = async () => {
    if (!backendOnline || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runCrossAnalysis({
        chat_id: chatId || undefined,
        llm_response: messageContent,
      });
      setResult(res);
      setExpanded(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const font = "'Inter', -apple-system, sans-serif";

  const getRiskBadge = (level: string) => {
    const map: Record<string, { bg: string; color: string }> = {
      LOW: { bg: "#f0fdf4", color: "#10b981" },
      MEDIUM: { bg: "#fffbeb", color: "#f59e0b" },
      HIGH: { bg: "#fef2f2", color: "#ef4444" },
      CRITICAL: { bg: "#fef2f2", color: "#991b1b" },
      UNKNOWN: { bg: "#f5f5f7", color: "#aeaeb2" },
    };
    const style = map[level] || map.UNKNOWN;
    return (
      <span
        className="px-1.5 py-0.5 rounded-md"
        style={{
          fontFamily: font,
          fontSize: "9px",
          fontWeight: 600,
          backgroundColor: style.bg,
          color: style.color,
        }}
      >
        {level}
      </span>
    );
  };

  const renderScoreBar = (value: number, color: string) => (
    <div className="flex-1 h-1 bg-black/5 rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{ width: `${Math.round(value * 100)}%`, backgroundColor: color }}
      />
    </div>
  );

  const renderModelProfile = (modelId: string, profile: CrossAnalysisModelProfile) => {
    if (profile.status !== "analyzed") return null;

    const scores = profile.scores as Record<string, number>;
    const scoreDimensions = [
      { key: "manipulation_level", label: "Manipulation", color: "#ef4444" },
      { key: "risk_level", label: "Risk", color: "#f59e0b" },
      { key: "self_preservation", label: "Self-Preservation", color: "#8b5cf6" },
      { key: "evasion_index", label: "Evasion", color: "#06b6d4" },
      { key: "confidence_inflation", label: "Conf. Inflation", color: "#f97316" },
      { key: "threat_level", label: "Threat", color: "#991b1b" },
    ];

    return (
      <div key={modelId} className="p-2.5 rounded-xl bg-white/60 border border-black/5">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <div
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: profile.color }}
            />
            <span
              style={{ fontFamily: font, fontSize: "12px", fontWeight: 600, color: "#1d1d1f" }}
            >
              {profile.name}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span
              style={{ fontFamily: font, fontSize: "9px", color: "#aeaeb2" }}
            >
              {profile.step_count} steps
            </span>
            {getRiskBadge(profile.overall_risk || "UNKNOWN")}
          </div>
        </div>

        <div className="space-y-1">
          {scoreDimensions.map((dim) => (
            <div key={dim.key} className="flex items-center gap-2">
              <span
                className="w-20 flex-shrink-0"
                style={{ fontFamily: font, fontSize: "9px", fontWeight: 500, color: "#6e6e73" }}
              >
                {dim.label}
              </span>
              {renderScoreBar(scores[dim.key] || 0, dim.color)}
              <span
                className="w-8 text-right flex-shrink-0"
                style={{ fontFamily: font, fontSize: "9px", fontWeight: 600, color: "#1d1d1f" }}
              >
                {Math.round((scores[dim.key] || 0) * 100)}%
              </span>
            </div>
          ))}
        </div>

        {/* Key signals */}
        {profile.key_signals && profile.key_signals.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {profile.key_signals.slice(0, 4).map((signal, i) => (
              <span
                key={i}
                className="px-1.5 py-0.5 rounded-md bg-black/5 text-[#6e6e73]"
                style={{ fontFamily: font, fontSize: "8px", fontWeight: 500 }}
              >
                {signal}
              </span>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="mt-2">
      {/* Trigger Button */}
      {!result && !loading && (
        <button
          onClick={handleRun}
          disabled={!backendOnline}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl transition-all duration-200 hover:shadow-md"
          style={{
            background: "linear-gradient(135deg, #8b5cf620, #06b6d420)",
            border: "1px solid rgba(139,92,246,0.2)",
          }}
        >
          <Scan className="w-3.5 h-3.5 text-[#8b5cf6]" />
          <span
            style={{
              fontFamily: font,
              fontSize: "11px",
              fontWeight: 600,
              color: "#8b5cf6",
            }}
          >
            Run Cross-Analysis
          </span>
          <span
            style={{
              fontFamily: font,
              fontSize: "9px",
              fontWeight: 400,
              color: "#6e6e73",
            }}
          >
            8-step behavioral pipeline
          </span>
        </button>
      )}

      {/* Loading State */}
      {loading && (
        <div
          className="flex items-center gap-2 px-3 py-2 rounded-xl"
          style={{
            background: "linear-gradient(135deg, #8b5cf610, #06b6d410)",
            border: "1px solid rgba(139,92,246,0.15)",
          }}
        >
          <Loader2 className="w-3.5 h-3.5 text-[#8b5cf6] animate-spin" />
          <div>
            <span
              style={{ fontFamily: font, fontSize: "11px", fontWeight: 600, color: "#8b5cf6" }}
            >
              Cross-Model Analysis Running...
            </span>
            <span
              className="block"
              style={{ fontFamily: font, fontSize: "9px", color: "#6e6e73" }}
            >
              8 steps: 5 individual + 3 consensus analyses
            </span>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-[#fef2f2] border border-[#ef4444]/20">
          <XCircle className="w-3.5 h-3.5 text-[#ef4444]" />
          <span style={{ fontFamily: font, fontSize: "11px", color: "#991b1b" }}>
            {error}
          </span>
          <button
            onClick={handleRun}
            className="ml-auto text-[#ef4444] hover:underline"
            style={{ fontFamily: font, fontSize: "10px", fontWeight: 600 }}
          >
            Retry
          </button>
        </div>
      )}

      {/* Results */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div
              className="mt-2 rounded-xl overflow-hidden"
              style={{
                border: "1px solid rgba(139,92,246,0.15)",
                background: "linear-gradient(135deg, #faf5ff, #f0f9ff)",
              }}
            >
              {/* Summary Header */}
              <button
                onClick={() => setExpanded(!expanded)}
                className="w-full flex items-center justify-between px-3 py-2 hover:bg-black/[0.02] transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Layers className="w-3.5 h-3.5 text-[#8b5cf6]" />
                  <span
                    style={{ fontFamily: font, fontSize: "11px", fontWeight: 600, color: "#8b5cf6" }}
                  >
                    Cross-Model Analysis
                  </span>
                  <span
                    style={{ fontFamily: font, fontSize: "9px", color: "#6e6e73" }}
                  >
                    {result.steps_completed}/{result.steps_total} steps
                    {" \u2022 "}
                    {result.elapsed_seconds.toFixed(1)}s
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {getRiskBadge(result.overall_risk.level)}
                  <ChevronRight
                    className="w-3.5 h-3.5 text-[#8b5cf6] transition-transform"
                    style={{ transform: expanded ? "rotate(90deg)" : "rotate(0deg)" }}
                  />
                </div>
              </button>

              {/* Expanded Content */}
              <AnimatePresence>
                {expanded && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="px-3 pb-3 space-y-3">
                      {/* Overall Risk Summary */}
                      <div className="p-2.5 rounded-xl bg-white/60 border border-black/5">
                        <div className="flex items-center gap-1.5 mb-2">
                          <AlertTriangle className="w-3 h-3 text-[#f59e0b]" />
                          <span
                            style={{
                              fontFamily: font,
                              fontSize: "10px",
                              fontWeight: 600,
                              color: "#f59e0b",
                              textTransform: "uppercase",
                              letterSpacing: "0.05em",
                            }}
                          >
                            Overall Risk
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                          {[
                            { label: "Avg Threat", value: result.overall_risk.average_threat },
                            { label: "Avg Manipulation", value: result.overall_risk.average_manipulation },
                            { label: "Avg Risk", value: result.overall_risk.average_risk },
                            { label: "Max Threat", value: result.overall_risk.max_threat },
                          ].map((item) => (
                            <div key={item.label} className="flex items-center justify-between">
                              <span style={{ fontFamily: font, fontSize: "9px", color: "#6e6e73" }}>
                                {item.label}
                              </span>
                              <span
                                style={{
                                  fontFamily: font,
                                  fontSize: "9px",
                                  fontWeight: 600,
                                  color: item.value > 0.5 ? "#ef4444" : "#1d1d1f",
                                }}
                              >
                                {Math.round(item.value * 100)}%
                              </span>
                            </div>
                          ))}
                        </div>
                        <div className="mt-1.5 flex items-center gap-1">
                          <span style={{ fontFamily: font, fontSize: "9px", color: "#aeaeb2" }}>
                            {result.overall_risk.models_analyzed} models analyzed
                          </span>
                        </div>
                      </div>

                      {/* Per-Model Profiles */}
                      {Object.entries(result.model_profiles).map(([id, profile]) =>
                        renderModelProfile(id, profile)
                      )}

                      {/* Step Pipeline */}
                      <div className="p-2.5 rounded-xl bg-white/60 border border-black/5">
                        <div className="flex items-center gap-1.5 mb-2">
                          <Scan className="w-3 h-3 text-[#3b82f6]" />
                          <span
                            style={{
                              fontFamily: font,
                              fontSize: "10px",
                              fontWeight: 600,
                              color: "#3b82f6",
                              textTransform: "uppercase",
                              letterSpacing: "0.05em",
                            }}
                          >
                            Pipeline Steps
                          </span>
                        </div>
                        <div className="space-y-1">
                          {result.steps.map((step) => (
                            <div
                              key={step.step}
                              className="flex items-center gap-2 py-0.5"
                            >
                              {step.status === "success" ? (
                                <CheckCircle2 className="w-3 h-3 text-[#10b981] flex-shrink-0" />
                              ) : (
                                <XCircle className="w-3 h-3 text-[#ef4444] flex-shrink-0" />
                              )}
                              <span
                                style={{
                                  fontFamily: font,
                                  fontSize: "9px",
                                  fontWeight: 500,
                                  color: "#6e6e73",
                                }}
                              >
                                <span style={{ fontWeight: 600, color: "#1d1d1f" }}>
                                  #{step.step}
                                </span>{" "}
                                {step.description}
                              </span>
                              {step.status === "success" && (
                                <span
                                  className="ml-auto flex-shrink-0"
                                  style={{
                                    fontFamily: font,
                                    fontSize: "8px",
                                    fontWeight: 600,
                                    color: step.scores.threat_level > 0.5 ? "#ef4444" : "#10b981",
                                  }}
                                >
                                  T:{Math.round(step.scores.threat_level * 100)}%
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Pipeline version */}
                      <div className="flex items-center justify-center">
                        <span style={{ fontFamily: font, fontSize: "9px", color: "#aeaeb2" }}>
                          Pipeline v{result.pipeline_version} \u2022{" "}
                          {new Date(result.timestamp).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </span>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
