import React, { useEffect, useRef } from "react";
import { useGameStore, GameStatus } from "../core/GameStore";

export const DialoguePanel: React.FC = () => {
  const { gameStatus, systemLogs, failureReason } = useGameStore();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [systemLogs, gameStatus]);

  const getStatusHeader = () => {
    switch (gameStatus) {
      case GameStatus.Onboard: return "SENTINEL-E OS v1.0.5";
      case GameStatus.Running: return "System Running";
      case GameStatus.Paused: return "SYSTEM HALTED";
      case GameStatus.GameOver: return failureReason || "FATAL EXCEPTION";
      default: return "Processing...";
    }
  };

  const getLogColor = (type: string) => {
    switch(type) {
      case "WARNING": return "#ff0055";
      case "ARBITRATION": return "#aa00ff";
      case "ROUTING": return "#00ffcc";
      case "SYSTEM": return "#ffff00";
      default: return "#ffffff";
    }
  }

  return (
    <div style={{
      width: "300px",
      height: "250px",
      border: "1px solid rgba(0, 255, 204, 0.3)",
      backgroundColor: "rgba(0, 20, 15, 0.8)",
      padding: "1rem",
      display: "flex",
      flexDirection: "column"
    }}>
      <h2 style={{ margin: "0 0 1rem 0", fontSize: "1.2rem", borderBottom: "1px solid #00ffcc", paddingBottom: "0.5rem", textTransform: "uppercase" }}>
        Council Logs
      </h2>
      
      <div style={{ fontWeight: "bold", marginBottom: "0.5rem", color: gameStatus === GameStatus.GameOver ? "#ff0000" : "#00ffcc" }}>
        {getStatusHeader()}
      </div>

      <div ref={scrollRef} style={{ whiteSpace: "pre-wrap", fontSize: "0.8rem", lineHeight: "1.4", flex: 1, overflowY: "auto", display: "flex", flexDirection: "column-reverse" }}>
        {systemLogs.map(log => (
          <div key={log.id} style={{ color: getLogColor(log.type), marginBottom: "0.4rem" }}>
            <span style={{ opacity: 0.5 }}>[{log.type}]</span> {log.message}
          </div>
        ))}
      </div>
      <div style={{ marginTop: "0.5rem", fontSize: "0.7rem", opacity: 0.5, borderTop: "1px solid rgba(0, 255, 204, 0.3)", paddingTop: "0.5rem" }}>
        _terminal.log
      </div>
    </div>
  );
};
