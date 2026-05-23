import { motion } from "framer-motion";

interface ConsensusMeterProps {
  value: number; // 0–100
  label?: string;
  color?: "cyan" | "emerald" | "violet" | "amber";
  size?: "sm" | "md" | "lg";
}

const colorMap = {
  cyan: { stroke: "#6ee7f9", bg: "rgba(110,231,249,0.08)", text: "text-[#6ee7f9]" },
  emerald: { stroke: "#34d399", bg: "rgba(52,211,153,0.08)", text: "text-[#34d399]" },
  violet: { stroke: "#8b5cf6", bg: "rgba(139,92,246,0.08)", text: "text-[#8b5cf6]" },
  amber: { stroke: "#f59e0b", bg: "rgba(245,158,11,0.08)", text: "text-[#f59e0b]" },
};

export function ConsensusMeter({
  value,
  label = "Consensus",
  color = "cyan",
  size = "md",
}: ConsensusMeterProps) {
  const c = colorMap[color];
  const radius = size === "sm" ? 20 : size === "lg" ? 36 : 28;
  const stroke = size === "sm" ? 2.5 : 3;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  const svgSize = (radius + stroke + 4) * 2;

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative" style={{ width: svgSize, height: svgSize }}>
        <svg width={svgSize} height={svgSize} viewBox={`0 0 ${svgSize} ${svgSize}`}>
          {/* Background track */}
          <circle
            cx={svgSize / 2}
            cy={svgSize / 2}
            r={radius}
            fill="none"
            stroke={c.bg}
            strokeWidth={stroke}
          />
          {/* Progress */}
          <motion.circle
            cx={svgSize / 2}
            cy={svgSize / 2}
            r={radius}
            fill="none"
            stroke={c.stroke}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.2, ease: "easeOut" }}
            transform={`rotate(-90 ${svgSize / 2} ${svgSize / 2})`}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={`font-semibold ${c.text} ${size === "sm" ? "text-[10px]" : size === "lg" ? "text-base" : "text-xs"}`}>
            {value}%
          </span>
        </div>
      </div>
      {label && (
        <span className="text-[#8a9099] text-[10px] tracking-widest uppercase font-medium">
          {label}
        </span>
      )}
    </div>
  );
}
