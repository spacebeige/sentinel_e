import React, { Component, ErrorInfo, ReactNode } from "react";
import { motion } from "motion/react";
import { RefreshCw, AlertTriangle, Terminal } from "lucide-react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class CinematicErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ errorInfo });
    console.error("[SENTINEL-E ARCHITECTURE FAULT]", error, errorInfo);
  }

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <div className="min-h-screen bg-white dark:bg-[#0a0d12] flex items-center justify-center p-6 relative overflow-hidden">
        {/* Atmospheric red glow */}
        <div className="absolute inset-0 pointer-events-none">
          <div
            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[60vw] h-[60vw] rounded-full"
            style={{
              background: "radial-gradient(circle, rgba(220,38,38,0.06) 0%, transparent 70%)",
              filter: "blur(80px)",
            }}
          />
        </div>

        {/* Topology grid */}
        <div
          className="absolute inset-0 opacity-[0.03] pointer-events-none"
          style={{
            backgroundImage:
              "linear-gradient(rgba(220,38,38,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(220,38,38,0.5) 1px, transparent 1px)",
            backgroundSize: "40px 40px",
            maskImage: "radial-gradient(ellipse at center, black 20%, transparent 70%)",
          }}
        />

        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.97 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="relative w-full max-w-2xl"
        >
          {/* Main error card */}
          <div
            className="rounded-3xl border border-red-200/40 dark:border-red-900/40 p-8"
            style={{
              background: "rgba(255,255,255,0.7)",
              backdropFilter: "blur(24px)",
              WebkitBackdropFilter: "blur(24px)",
              boxShadow: "0 8px 64px rgba(220,38,38,0.08), 0 1px 2px rgba(0,0,0,0.04)",
            }}
          >
            {/* Header */}
            <div className="flex items-start gap-4 mb-8">
              <div
                className="w-12 h-12 rounded-2xl flex items-center justify-center flex-shrink-0"
                style={{ background: "rgba(220,38,38,0.1)" }}
              >
                <AlertTriangle className="w-6 h-6 text-red-500" />
              </div>
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className="text-[10px] font-bold tracking-[0.25em] text-red-400 uppercase"
                  >
                    ARCHITECTURE FAULT
                  </span>
                  <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
                </div>
                <h1
                  className="text-[#1d1d1f] dark:text-white font-bold"
                  style={{ fontSize: "clamp(20px, 3vw, 26px)", lineHeight: 1.2 }}
                >
                  Cognitive Layer Failure
                </h1>
                <p className="text-[#6e6e73] dark:text-[#a1a1aa] text-sm mt-1">
                  A fault was detected in the rendering architecture. The system has isolated the affected layer.
                </p>
              </div>
            </div>

            {/* Divider */}
            <div className="w-full h-px bg-red-100 dark:bg-red-900/30 mb-6" />

            {/* Error diagnostic */}
            <div className="mb-6">
              <div className="flex items-center gap-2 mb-3">
                <Terminal className="w-3.5 h-3.5 text-[#6e6e73] dark:text-[#a1a1aa]" />
                <span className="text-[11px] font-bold tracking-[0.2em] text-[#6e6e73] dark:text-[#a1a1aa] uppercase">
                  Diagnostic Output
                </span>
              </div>
              <div
                className="rounded-2xl p-4 font-mono text-sm"
                style={{
                  background: "rgba(0,0,0,0.04)",
                  border: "1px solid rgba(0,0,0,0.06)",
                }}
              >
                <div className="text-red-500 font-semibold mb-1 text-[13px]">
                  {this.state.error?.name || "Error"}
                </div>
                <div className="text-[#1d1d1f] dark:text-white text-[12px] leading-relaxed break-all">
                  {this.state.error?.message || "Unknown rendering fault"}
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => {
                  this.setState({ hasError: false, error: null, errorInfo: null });
                  window.location.href = "/";
                }}
                className="flex items-center gap-2 px-6 py-3 rounded-2xl bg-[#1d1d1f] dark:bg-white text-white dark:text-[#1d1d1f] font-bold text-[14px] flex-1 justify-center"
              >
                <RefreshCw className="w-4 h-4" />
                Reinitialize Layer
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => window.location.reload()}
                className="px-6 py-3 rounded-2xl border border-black/10 dark:border-white/10 text-[#1d1d1f] dark:text-white font-medium text-[14px]"
              >
                Force Reload
              </motion.button>
            </div>
          </div>

          {/* System ID badge */}
          <div className="flex justify-center mt-4">
            <span className="text-[11px] font-mono text-[#6e6e73] dark:text-[#a1a1aa] tracking-widest opacity-50">
              SENTINEL-E / SYS.ERR / LAYER.ISOLATED
            </span>
          </div>
        </motion.div>
      </div>
    );
  }
}
