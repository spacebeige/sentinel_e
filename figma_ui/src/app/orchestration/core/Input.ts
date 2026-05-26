import { useEffect } from "react";
import { useGameStore } from "./GameStore";

export function useGameInput() {
  const { dispatchMove, dispatchRotate, dispatchDrop, dispatchPause, dispatchResume } = useGameStore();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case "ArrowLeft":
        case "A":
        case "a":
          dispatchMove(-1, 0);
          break;
        case "ArrowRight":
        case "D":
        case "d":
          dispatchMove(1, 0);
          break;
        case "ArrowDown":
        case "S":
        case "s":
          dispatchMove(0, 1);
          break;
        case "ArrowUp":
        case "W":
        case "w":
          dispatchRotate();
          break;
        case " ":
          dispatchDrop();
          break;
        case "Escape":
          const state = useGameStore.getState();
          if (state.gameStatus === 1) dispatchPause();
          else dispatchResume();
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [dispatchMove, dispatchRotate, dispatchDrop, dispatchPause, dispatchResume]);
}

export function useGameTick() {
  const { dispatchGameTick, gameStatus } = useGameStore();

  useEffect(() => {
    if (gameStatus !== 1) return; // 1 = Running

    const interval = setInterval(() => {
      dispatchGameTick();
    }, 800); // 800ms base tick rate

    return () => clearInterval(interval);
  }, [gameStatus, dispatchGameTick]);
}
