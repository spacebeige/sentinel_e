import { cn } from "../components/ui/utils";
import { ReactNode } from "react";

interface GlassPanelProps {
  children: ReactNode;
  className?: string;
  glow?: "cyan" | "emerald" | "violet" | "amber" | "none";
  border?: boolean;
}

export function GlassPanel({
  children,
  className,
  glow = "none",
  border = true,
}: GlassPanelProps) {
  const glowStyles = {
    cyan: "shadow-[0_0_30px_rgba(110,231,249,0.06)] hover:shadow-[0_0_40px_rgba(110,231,249,0.10)]",
    emerald: "shadow-[0_0_30px_rgba(52,211,153,0.06)] hover:shadow-[0_0_40px_rgba(52,211,153,0.10)]",
    violet: "shadow-[0_0_30px_rgba(139,92,246,0.06)] hover:shadow-[0_0_40px_rgba(139,92,246,0.10)]",
    amber: "shadow-[0_0_30px_rgba(245,158,11,0.06)] hover:shadow-[0_0_40px_rgba(245,158,11,0.10)]",
    none: "",
  };

  return (
    <div
      className={cn(
        "relative rounded-lg backdrop-blur-md transition-all duration-300",
        "bg-[rgba(17,18,20,0.65)]",
        border && "border border-[rgba(110,231,249,0.08)]",
        glowStyles[glow],
        className
      )}
    >
      {children}
    </div>
  );
}
