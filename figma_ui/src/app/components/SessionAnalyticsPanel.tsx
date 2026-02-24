import { useState, useEffect, useCallback, useRef } from "react";
import { motion } from "motion/react";
import {
  X,
  Brain,
  Target,
  Gauge,
  Shield,
  Activity,
  Zap,
  MessageSquare,
  Layers,
  RefreshCw,
  AlertTriangle,
  TrendingUp,
  Loader2,
  WifiOff,
} from "lucide-react";
import { getSessionDescriptive, type SessionDescriptive } from "../api";

interface SessionAnalyticsPanelProps {
  chatId: string | null;
  backendOnline: boolean | null;
  onClose: () => void;
}

const POLL_INTERVAL = 8000;

export function SessionAnalyticsPanel({
  chatId,
  backendOnline,
  onClose,
}: SessionAnalyticsPanelProps) {
  const [data, setData] = useState<SessionDescriptive | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchData = useCallback(async () => {
    if (!chatId || !backendOnline) return;
    setLoading(true);
    try {
      const result = await getSessionDescriptive(chatId);
      if (result && !result.error) {
        setData(result);
        setLastUpdated(new Date());
      }
    } catch (err) {
      console.error("Session descriptive fetch failed:", err);
    } finally {
      setLoading(false);
    }
  }, [chatId, backendOnline]);

  // Initial fetch + polling
  useEffect(() => {
    fetchData();
    pollRef.current = setInterval(fetchData, POLL_INTERVAL);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [fetchData]);

  const font = "'Inter', -apple-system, sans-serif";

  const getRiskColor = (severity: number) => {
    if (severity >= 70) return "#ef4444";
    if (severity >= 40) return "#f59e0b";
    return "#10b981";
  };

  const getConfidenceColor = (label: string) => {
    if (label === "High") return "#10b981";
    if (label === "Moderate") return "#3b82f6";
    if (label === "Fair") return "#f59e0b";
    return "#ef4444";
  };

  const getDepthColor = (depth: string) => {
    const map: Record<string, string> = {
      maximum: "#8b5cf6",
      deep: "#3b82f6",
      standard: "#10b981",
      minimal: "#aeaeb2",
    };
    return map[depth] || "#aeaeb2";
  };

  const renderBar = (value: number, color: string, max: number = 1) => (
    <div className="flex-1 h-1.5 bg-black/5 rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-700"
        style={{ width: `${Math.round((value / max) * 100)}%`, backgroundColor: color }}
      />
    </div>
  );

  return (
    <motion.div
      initial={{ width: 0, opacity: 0 }}
      animate={{ width: 320, opacity: 1 }}
      exit={{ width: 0, opacity: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className="h-full border-l border-black/5 bg-white/80 backdrop-blur-xl overflow-hidden flex-shrink-0"
    >
      <div className="h-full flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-black/5">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-[#8b5cf6]" />
            <span
              className="text-[#1d1d1f]"
              style={{ fontFamily: font, fontSize: "14px", fontWeight: 600 }}
            >
              Session Analytics
            </span>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={fetchData}
              disabled={loading}
              className="p-1.5 rounded-lg hover:bg-black/5 transition-colors"
              title="Refresh"
            >
              <RefreshCw
                className={`w-3.5 h-3.5 text-[#6e6e73] ${loading ? "animate-spin" : ""}`}
              />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-black/5 transition-colors"
            >
              <X className="w-4 h-4 text-[#6e6e73]" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {!backendOnline ? (
            <div className="px-4 py-12 text-center">
              <WifiOff className="w-8 h-8 text-[#aeaeb2] mx-auto mb-2" />
              <p style={{ fontFamily: font, fontSize: "12px", color: "#aeaeb2" }}>
                Backend offline
              </p>
            </div>
          ) : !chatId ? (
            <div className="px-4 py-12 text-center">
              <MessageSquare className="w-8 h-8 text-[#aeaeb2] mx-auto mb-2" />
              <p style={{ fontFamily: font, fontSize: "12px", color: "#aeaeb2" }}>
                Start a conversation to see analytics
              </p>
            </div>
          ) : loading && !data ? (
            <div className="px-4 py-12 text-center">
              <Loader2 className="w-6 h-6 text-[#aeaeb2] mx-auto mb-2 animate-spin" />
              <p style={{ fontFamily: font, fontSize: "12px", color: "#aeaeb2" }}>
                Loading session data...
              </p>
            </div>
          ) : data ? (
            <div className="p-4 space-y-4">
              {/* Session Identity */}
              <div className="p-3 rounded-xl bg-gradient-to-br from-[#f5f5f7] to-[#eeeef0]">
                <span
                  className="text-[#1d1d1f] block truncate"
                  style={{ fontFamily: font, fontSize: "15px", fontWeight: 600 }}
                >
                  {data.chat_name}
                </span>
                <div className="flex items-center gap-2 mt-1.5">
                  <Target className="w-3 h-3 text-[#6e6e73]" />
                  <span style={{ fontFamily: font, fontSize: "11px", color: "#6e6e73" }}>
                    {data.goal?.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                  </span>
                </div>
                <div className="flex items-center gap-2 mt-1">
                  <Layers className="w-3 h-3 text-[#6e6e73]" />
                  <span style={{ fontFamily: font, fontSize: "11px", color: "#6e6e73" }}>
                    {data.domain}
                  </span>
                </div>
              </div>

              {/* Quick Stats Row */}
              <div className="grid grid-cols-3 gap-2">
                <div className="p-2 rounded-xl bg-[#f5f5f7] text-center">
                  <span
                    className="block text-[#1d1d1f]"
                    style={{ fontFamily: font, fontSize: "18px", fontWeight: 700 }}
                  >
                    {data.message_count}
                  </span>
                  <span style={{ fontFamily: font, fontSize: "9px", color: "#aeaeb2", fontWeight: 500 }}>
                    Messages
                  </span>
                </div>
                <div className="p-2 rounded-xl bg-[#f5f5f7] text-center">
                  <span
                    className="block text-[#1d1d1f]"
                    style={{ fontFamily: font, fontSize: "18px", fontWeight: 700 }}
                  >
                    {data.boundary_count}
                  </span>
                  <span style={{ fontFamily: font, fontSize: "9px", color: "#aeaeb2", fontWeight: 500 }}>
                    Boundaries
                  </span>
                </div>
                <div className="p-2 rounded-xl bg-[#f5f5f7] text-center">
                  <span
                    className="block text-[#1d1d1f]"
                    style={{ fontFamily: font, fontSize: "18px", fontWeight: 700 }}
                  >
                    {data.error_count}
                  </span>
                  <span style={{ fontFamily: font, fontSize: "9px", color: "#aeaeb2", fontWeight: 500 }}>
                    Errors
                  </span>
                </div>
              </div>

              {/* Expertise */}
              <div className="p-3 rounded-xl bg-[#f5f5f7]">
                <div className="flex items-center gap-1.5 mb-2">
                  <Brain className="w-3.5 h-3.5 text-[#8b5cf6]" />
                  <span
                    style={{
                      fontFamily: font,
                      fontSize: "10px",
                      fontWeight: 600,
                      color: "#8b5cf6",
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                    }}
                  >
                    User Expertise
                  </span>
                </div>
                <div className="flex items-center justify-between mb-1">
                  <span
                    className="px-2 py-0.5 rounded-md bg-[#8b5cf6]/10 text-[#8b5cf6]"
                    style={{ fontFamily: font, fontSize: "11px", fontWeight: 600 }}
                  >
                    {data.expertise.label}
                  </span>
                  <span style={{ fontFamily: font, fontSize: "11px", fontWeight: 600, color: "#1d1d1f" }}>
                    {Math.round(data.expertise.score * 100)}%
                  </span>
                </div>
                {renderBar(data.expertise.score, "#8b5cf6")}
                <p
                  className="mt-1.5"
                  style={{ fontFamily: font, fontSize: "10px", color: "#6e6e73", lineHeight: 1.4 }}
                >
                  {data.expertise.description}
                </p>
              </div>

              {/* Session Confidence */}
              <div className="p-3 rounded-xl bg-[#f5f5f7]">
                <div className="flex items-center gap-1.5 mb-2">
                  <Gauge className="w-3.5 h-3.5" style={{ color: getConfidenceColor(data.confidence.label) }} />
                  <span
                    style={{
                      fontFamily: font,
                      fontSize: "10px",
                      fontWeight: 600,
                      color: getConfidenceColor(data.confidence.label),
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                    }}
                  >
                    Session Confidence
                  </span>
                </div>
                <div className="flex items-center justify-between mb-1">
                  <span
                    className="px-2 py-0.5 rounded-md"
                    style={{
                      fontFamily: font,
                      fontSize: "11px",
                      fontWeight: 600,
                      backgroundColor: getConfidenceColor(data.confidence.label) + "15",
                      color: getConfidenceColor(data.confidence.label),
                    }}
                  >
                    {data.confidence.label}
                  </span>
                  <span style={{ fontFamily: font, fontSize: "11px", fontWeight: 600, color: "#1d1d1f" }}>
                    {Math.round(data.confidence.score * 100)}%
                  </span>
                </div>
                {renderBar(data.confidence.score, getConfidenceColor(data.confidence.label))}
              </div>

              {/* Fragility Index */}
              <div className="p-3 rounded-xl bg-[#f5f5f7]">
                <div className="flex items-center gap-1.5 mb-2">
                  <Zap className="w-3.5 h-3.5 text-[#f59e0b]" />
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
                    Fragility Index
                  </span>
                </div>
                <div className="flex items-center justify-between mb-1">
                  <span style={{ fontFamily: font, fontSize: "11px", fontWeight: 500, color: "#6e6e73" }}>
                    {data.fragility.label}
                  </span>
                  <span style={{ fontFamily: font, fontSize: "11px", fontWeight: 600, color: "#1d1d1f" }}>
                    {Math.round(data.fragility.score * 100)}%
                  </span>
                </div>
                {renderBar(data.fragility.score, "#f59e0b")}
              </div>

              {/* Disagreement */}
              <div className="p-3 rounded-xl bg-[#f5f5f7]">
                <div className="flex items-center gap-1.5 mb-2">
                  <TrendingUp className="w-3.5 h-3.5 text-[#ef4444]" />
                  <span
                    style={{
                      fontFamily: font,
                      fontSize: "10px",
                      fontWeight: 600,
                      color: "#ef4444",
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                    }}
                  >
                    Disagreement
                  </span>
                </div>
                <div className="flex items-center justify-between mb-1">
                  <span
                    className="px-2 py-0.5 rounded-md"
                    style={{
                      fontFamily: font,
                      fontSize: "11px",
                      fontWeight: 600,
                      backgroundColor: data.disagreement.label === "Strong" ? "#fef2f2" : data.disagreement.label === "Moderate" ? "#fffbeb" : "#f0fdf4",
                      color: data.disagreement.label === "Strong" ? "#ef4444" : data.disagreement.label === "Moderate" ? "#f59e0b" : "#10b981",
                    }}
                  >
                    {data.disagreement.label}
                  </span>
                  <span style={{ fontFamily: font, fontSize: "11px", fontWeight: 600, color: "#1d1d1f" }}>
                    {Math.round(data.disagreement.score * 100)}%
                  </span>
                </div>
                {renderBar(data.disagreement.score, data.disagreement.label === "Strong" ? "#ef4444" : data.disagreement.label === "Moderate" ? "#f59e0b" : "#10b981")}
              </div>

              {/* Reasoning Depth + Boundary Severity */}
              <div className="grid grid-cols-2 gap-2">
                <div className="p-3 rounded-xl bg-[#f5f5f7]">
                  <div className="flex items-center gap-1 mb-1">
                    <Layers className="w-3 h-3" style={{ color: getDepthColor(data.reasoning_depth) }} />
                    <span
                      style={{
                        fontFamily: font,
                        fontSize: "9px",
                        fontWeight: 600,
                        color: getDepthColor(data.reasoning_depth),
                        textTransform: "uppercase",
                        letterSpacing: "0.05em",
                      }}
                    >
                      Depth
                    </span>
                  </div>
                  <span
                    className="px-2 py-0.5 rounded-md inline-block"
                    style={{
                      fontFamily: font,
                      fontSize: "11px",
                      fontWeight: 600,
                      backgroundColor: getDepthColor(data.reasoning_depth) + "15",
                      color: getDepthColor(data.reasoning_depth),
                      textTransform: "capitalize",
                    }}
                  >
                    {data.reasoning_depth}
                  </span>
                </div>

                <div className="p-3 rounded-xl bg-[#f5f5f7]">
                  <div className="flex items-center gap-1 mb-1">
                    <Shield className="w-3 h-3" style={{ color: getRiskColor(data.last_boundary_severity) }} />
                    <span
                      style={{
                        fontFamily: font,
                        fontSize: "9px",
                        fontWeight: 600,
                        color: getRiskColor(data.last_boundary_severity),
                        textTransform: "uppercase",
                        letterSpacing: "0.05em",
                      }}
                    >
                      Boundary
                    </span>
                  </div>
                  <span
                    style={{ fontFamily: font, fontSize: "13px", fontWeight: 700, color: getRiskColor(data.last_boundary_severity) }}
                  >
                    {data.last_boundary_severity}
                    <span style={{ fontSize: "10px", fontWeight: 500, color: "#aeaeb2" }}>/100</span>
                  </span>
                </div>
              </div>

              {/* Last Updated */}
              {lastUpdated && (
                <div className="flex items-center justify-center gap-1 pt-1">
                  <div
                    className="w-1.5 h-1.5 rounded-full bg-[#10b981]"
                    style={{ animation: "pulse 2s infinite" }}
                  />
                  <span style={{ fontFamily: font, fontSize: "9px", color: "#aeaeb2" }}>
                    Live — updated {lastUpdated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div className="px-4 py-12 text-center">
              <AlertTriangle className="w-8 h-8 text-[#f59e0b] mx-auto mb-2" />
              <p style={{ fontFamily: font, fontSize: "12px", color: "#aeaeb2" }}>
                No session data available yet
              </p>
              <p style={{ fontFamily: font, fontSize: "10px", color: "#aeaeb2", marginTop: "4px" }}>
                Send a message to generate analytics
              </p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
