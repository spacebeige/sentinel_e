import React from "react";
import { useGameStore, GameStatus } from "../core/GameStore";
import { BlockRenderer } from "../render/BlockRenderer";

export const StatusPanel: React.FC = () => {
  const { 
    gameStatus, 
    spiritReserve, 
    divergencePressure, 
    hallucinationRisk, 
    memorySaturation,
    line,
    failureReason
  } = useGameStore();
  
  const nextSpirit = spiritReserve[0];

  const renderMeter = (label: string, value: number, color: string) => {
    return (
      <div style={{ marginBottom: "0.8rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", opacity: 0.8, marginBottom: "0.2rem" }}>
          <span>{label}</span>
          <span>{Math.floor(value)}%</span>
        </div>
        <div style={{ width: "100%", height: "10px", backgroundColor: "rgba(0,0,0,0.5)", border: `1px solid ${color}` }}>
          <div style={{ 
            width: `${Math.min(100, Math.max(0, value))}%`, 
            height: "100%", 
            backgroundColor: color,
            boxShadow: `0 0 8px ${color}`,
            transition: "width 0.2s ease-out, background-color 0.2s"
          }} />
        </div>
      </div>
    );
  };

  const renderNextPiece = () => {
    if (!nextSpirit) return null;
    const locs = nextSpirit.shape; 
    const minX = Math.min(...locs.map(p => p.x));
    const maxX = Math.max(...locs.map(p => p.x));
    const minY = Math.min(...locs.map(p => p.y));
    const maxY = Math.max(...locs.map(p => p.y));

    const w = maxX - minX + 1;
    const h = maxY - minY + 1;

    const cells = Array.from({ length: 4 * 4 }, (_, i) => {
      const x = (i % 4) + minX - Math.floor((4 - w)/2);
      const y = Math.floor(i / 4) + minY - Math.floor((4 - h)/2);
      const isFilled = locs.some(p => p.x === x && p.y === y);
      return { filled: isFilled };
    });

    return (
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gridTemplateRows: "repeat(4, 1fr)",
          width: "80px",
          height: "80px",
          gap: "1px",
          margin: "0.5rem 0"
        }}
      >
        {cells.map((c, i) => (
          <BlockRenderer key={i} filled={c.filled} />
        ))}
      </div>
    );
  };

  const getPhase = () => {
    if (line < 5) return "Phase 1: Single-model";
    if (line < 10) return "Phase 2: Multi-model";
    if (line < 15) return "Phase 3: Debate emergence";
    if (line < 20) return "Phase 4: Semantic divergence";
    if (line < 25) return "Phase 5: Council stabilization";
    return "Phase 6: Recursive orchestration";
  };

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      border: "1px solid rgba(0, 255, 204, 0.3)",
      padding: "1rem",
      width: "250px",
      backgroundColor: "rgba(0, 20, 15, 0.8)",
      textTransform: "uppercase"
    }}>
      <h2 style={{ margin: "0 0 1rem 0", fontSize: "1.2rem", borderBottom: "1px solid #00ffcc", paddingBottom: "0.5rem" }}>
        System Status
      </h2>
      
      <div style={{ marginBottom: "1.5rem" }}>
        <div style={{ opacity: 0.7, fontSize: "0.8rem", color: "#ffff00" }}>Orchestration Stage</div>
        <div style={{ fontSize: "0.9rem", fontWeight: "bold" }}>{getPhase()}</div>
        <div style={{ fontSize: "0.7rem", opacity: 0.5 }}>Stabilized Pipelines: {line}</div>
      </div>

      <div style={{ marginBottom: "1rem" }}>
        {renderMeter("Divergence Pressure", divergencePressure, divergencePressure > 80 ? "#ff0000" : "#ff0055")}
        {renderMeter("Hallucination Risk", hallucinationRisk, hallucinationRisk > 80 ? "#ff0000" : "#aa00ff")}
        {renderMeter("Memory Saturation", memorySaturation, memorySaturation > 80 ? "#ff0000" : "#00aaff")}
      </div>

      <div style={{ marginBottom: "1rem" }}>
        <div style={{ opacity: 0.7, fontSize: "0.8rem" }}>State</div>
        <div style={{ fontSize: "1.2rem", color: gameStatus === GameStatus.GameOver ? "#ff0055" : "#00ffcc" }}>
          {GameStatus[gameStatus]}
        </div>
      </div>

      <div style={{ marginTop: "1rem" }}>
        <div style={{ opacity: 0.7, fontSize: "0.8rem" }}>Next Fragment</div>
        <div style={{ fontSize: "0.6rem", color: "#00ffcc" }}>{nextSpirit?.cognitiveType}</div>
        {renderNextPiece()}
      </div>
      
      {gameStatus === GameStatus.Onboard && (
         <div style={{ marginTop: "auto", fontSize: "0.8rem", color: "#ffff00" }}>
           Press Enter to boot sequence.
         </div>
      )}
      
      {gameStatus === GameStatus.GameOver && (
         <div style={{ marginTop: "auto", fontSize: "0.8rem", color: "#ff0055" }}>
           <div style={{ fontWeight: "bold", marginBottom: "0.5rem" }}>{failureReason || "SYSTEM HALT"}</div>
           Press Enter to reset.
         </div>
      )}
    </div>
  );
};
