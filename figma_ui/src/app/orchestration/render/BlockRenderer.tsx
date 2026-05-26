import React from "react";

interface BlockProps {
  filled: boolean;
  color?: string;
  isGhost?: boolean;
  pulsing?: boolean;
}

export const BlockRenderer: React.FC<BlockProps> = ({ filled, color = "#00ffcc", isGhost = false, pulsing = false }) => {
  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        backgroundColor: filled ? (isGhost ? "transparent" : color) : "transparent",
        border: filled ? `1px solid ${color}` : "1px solid rgba(0, 255, 204, 0.05)",
        boxShadow: filled && !isGhost ? `0 0 ${pulsing ? '15px' : '10px'} ${color}, inset 0 0 ${pulsing ? '10px' : '5px'} ${color}` : "none",
        opacity: isGhost ? 0.3 : (pulsing ? 0.8 : 1),
        transition: "background-color 0.1s ease, box-shadow 0.1s ease"
      }}
    />
  );
};
