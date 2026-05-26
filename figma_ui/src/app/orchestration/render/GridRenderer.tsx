import React from "react";
import { useGameStore } from "../core/GameStore";
import { MATRIX_WIDTH, MATRIX_HEIGHT } from "../core/Board";
import { BlockRenderer } from "./BlockRenderer";
import { getSpiritLocation } from "../core/Pieces";

const getTypeIdColor = (typeId?: string) => {
  switch (typeId) {
    case "I": return "#00ffff"; // Cyan
    case "J": return "#0000ff"; // Blue
    case "L": return "#ff7f00"; // Orange
    case "O": return "#ffff00"; // Yellow
    case "S": return "#00ff00"; // Green
    case "T": return "#800080"; // Purple
    case "Z": return "#ff0000"; // Red
    default: return "#00ffcc";
  }
}

export const GridRenderer: React.FC = () => {
  const { bricks, spirit, divergencePressure } = useGameStore();

  const activeSpiritLocs = getSpiritLocation(spirit);
  const activeColor = getTypeIdColor(spirit.typeId);
  const isPulsing = divergencePressure > 60;

  // Flatten the board into a 1D array of [MATRIX_WIDTH * MATRIX_HEIGHT]
  const cells = Array.from({ length: MATRIX_WIDTH * MATRIX_HEIGHT }, (_, i) => {
    const x = i % MATRIX_WIDTH;
    const y = Math.floor(i / MATRIX_WIDTH);

    const isLocked = bricks.some(b => b.location.x === x && b.location.y === y);
    const isActive = activeSpiritLocs.some(p => p.x === x && p.y === y);

    return {
      x,
      y,
      filled: isLocked || isActive,
      color: isActive ? activeColor : isLocked ? "#aaaaaa" : undefined,
    };
  });

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: `repeat(${MATRIX_WIDTH}, 1fr)`,
        gridTemplateRows: `repeat(${MATRIX_HEIGHT}, 1fr)`,
        width: "300px",
        height: "600px",
        backgroundColor: "rgba(0, 0, 0, 0.8)",
        border: "2px solid #00ffcc",
        boxShadow: isPulsing ? "0 0 40px rgba(255, 0, 85, 0.4)" : "0 0 20px rgba(0, 255, 204, 0.2)",
        padding: "2px",
        gap: "1px",
        transition: "box-shadow 0.3s ease"
      }}
    >
      {cells.map((cell, idx) => (
        <BlockRenderer key={idx} filled={cell.filled} color={cell.color} pulsing={isPulsing && cell.filled} />
      ))}
    </div>
  );
};
