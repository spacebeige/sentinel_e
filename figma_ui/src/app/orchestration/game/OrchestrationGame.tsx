import React, { useEffect } from "react";
import { TerminalHUD } from "../ui/TerminalHUD";
import { GridRenderer } from "../render/GridRenderer";
import { StatusPanel } from "../ui/StatusPanel";
import { DialoguePanel } from "../ui/DialoguePanel";
import { useGameInput, useGameTick } from "../core/Input";
import { useGameStore, GameStatus } from "../core/GameStore";

export default function OrchestrationGame() {
  useGameInput();
  useGameTick();

  const { dispatchReset } = useGameStore();

  useEffect(() => {
    const handleEnter = (e: KeyboardEvent) => {
      if (e.key === "Enter") {
        dispatchReset();
      }
    };
    window.addEventListener("keydown", handleEnter);
    return () => window.removeEventListener("keydown", handleEnter);
  }, [dispatchReset]);

  return (
    <TerminalHUD>
      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        <StatusPanel />
        <DialoguePanel />
      </div>
      <GridRenderer />
      
      {/* Input Legend */}
      <div style={{ 
        position: "absolute", bottom: "2rem", left: "2rem", 
        fontSize: "0.8rem", opacity: 0.5, border: "1px solid rgba(0, 255, 204, 0.3)", padding: "1rem" 
      }}>
        <div>CONTROLS</div>
        <div>[W/UP] Rotate</div>
        <div>[A/LEFT] Move Left</div>
        <div>[D/RIGHT] Move Right</div>
        <div>[S/DOWN] Soft Drop</div>
        <div>[SPACE] Hard Drop</div>
        <div>[ESC] Pause</div>
        <div>[ENTER] Reset/Start</div>
      </div>
    </TerminalHUD>
  );
}
