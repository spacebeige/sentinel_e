/**
 * OmegaInsightPanel — semantic cognition insights overlay
 * Renders the Omega Kernel's internal reasoning metadata
 * in a minimal, cinematic glass panel beside assistant messages.
 */

interface OmegaInsightPanelProps {
  isExpanded?: boolean;
  onToggle?: () => void;
  isDark?: boolean;
}

export function OmegaInsightPanel({
  isExpanded = false,
  onToggle,
  isDark = false,
}: OmegaInsightPanelProps) {
  if (!isExpanded) return null;

  return (
    <div
      className="mt-2 p-3 rounded-xl"
      style={{
        background: isDark ? "rgba(139,92,246,0.06)" : "rgba(139,92,246,0.04)",
        border: "1px solid rgba(139,92,246,0.15)",
        backdropFilter: "blur(12px)",
      }}
    >
      <div className="flex items-center gap-2">
        <div
          className="w-1.5 h-1.5 rounded-full animate-pulse"
          style={{ background: "#8b5cf6" }}
        />
        <span
          style={{
            fontFamily: "monospace",
            fontSize: "10px",
            fontWeight: 600,
            letterSpacing: "0.1em",
            color: "#8b5cf6",
            textTransform: "uppercase",
          }}
        >
          Omega Kernel Active
        </span>
      </div>
    </div>
  );
}
