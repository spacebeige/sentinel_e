import { create } from "zustand";
import { Brick, MATRIX_WIDTH, MATRIX_HEIGHT, createBrick, offsetBrick } from "./Board";
import { Spirit, EmptySpirit, generateSpiritReserve, isValidInMatrix, moveSpirit, rotateSpirit, adjustSpiritOffset, getSpiritLocation } from "./Pieces";

export enum GameStatus {
  Onboard,
  Running,
  LineClearing,
  Paused,
  ScreenClearing,
  GameOver
}

export interface LogEntry {
  id: string;
  type: "WARNING" | "ROUTING" | "COUNCIL" | "ARBITRATION" | "SYSTEM";
  message: string;
}

export interface GameState {
  bricks: Brick[];
  spirit: Spirit;
  spiritReserve: Spirit[];
  gameStatus: GameStatus;
  score: number;
  line: number;
  isMute: boolean;

  // Semantic Pressures
  divergencePressure: number; // 0-100
  hallucinationRisk: number; // 0-100
  memorySaturation: number; // 0-100
  failureReason: string | null;
  systemLogs: LogEntry[];

  // Actions
  dispatchReset: () => void;
  dispatchPause: () => void;
  dispatchResume: () => void;
  dispatchMove: (dx: number, dy: number) => void;
  dispatchRotate: () => void;
  dispatchDrop: () => void;
  dispatchGameTick: () => void;
}

const SCORE_EVERY_SPIRIT = 10;
const calculateScore = (lines: number) => {
  if (lines === 1) return 100;
  if (lines === 2) return 300;
  if (lines === 3) return 500;
  if (lines >= 4) return 800;
  return 0;
};

export const useGameStore = create<GameState>((set, get) => ({
  bricks: [],
  spirit: EmptySpirit,
  spiritReserve: [],
  gameStatus: GameStatus.Onboard,
  score: 0,
  line: 0,
  isMute: false,

  divergencePressure: 0,
  hallucinationRisk: 0,
  memorySaturation: 0,
  failureReason: null,
  systemLogs: [],

  dispatchReset: () => set((state) => {
    if (state.gameStatus === GameStatus.Onboard || state.gameStatus === GameStatus.GameOver) {
      return { 
        gameStatus: GameStatus.Running, 
        spiritReserve: generateSpiritReserve(), 
        spirit: EmptySpirit, 
        bricks: [], 
        score: 0, 
        line: 0,
        divergencePressure: 0,
        hallucinationRisk: 0,
        memorySaturation: 0,
        failureReason: null,
        systemLogs: [{ id: Date.now().toString(), type: "SYSTEM", message: "Sequence Initialized." }]
      };
    }
    return { gameStatus: GameStatus.GameOver, failureReason: "MANUAL_ABORT" }; 
  }),

  dispatchPause: () => set((state) => state.gameStatus === GameStatus.Running ? { gameStatus: GameStatus.Paused } : {}),
  
  dispatchResume: () => set((state) => state.gameStatus === GameStatus.Paused ? { gameStatus: GameStatus.Running } : {}),

  dispatchMove: (dx, dy) => set((state) => {
    if (state.gameStatus !== GameStatus.Running) return {};
    const newSpirit = moveSpirit(state.spirit, dx, dy);
    if (isValidInMatrix(newSpirit, state.bricks)) return { spirit: newSpirit };
    return {};
  }),

  dispatchRotate: () => set((state) => {
    if (state.gameStatus !== GameStatus.Running) return {};
    let newSpirit = rotateSpirit(state.spirit);
    newSpirit = adjustSpiritOffset(newSpirit);
    if (isValidInMatrix(newSpirit, state.bricks)) return { spirit: newSpirit };
    return {};
  }),

  dispatchDrop: () => set((state) => {
    if (state.gameStatus !== GameStatus.Running) return {};
    let i = 0;
    while (isValidInMatrix(moveSpirit(state.spirit, 0, i + 1), state.bricks)) {
      i++;
    }
    return { spirit: moveSpirit(state.spirit, 0, i) };
  }),

  dispatchGameTick: () => set((state) => {
    if (state.gameStatus !== GameStatus.Running) return {};

    // Generate first piece if empty
    if (state.spirit === EmptySpirit) {
      const next = state.spiritReserve[0] || generateSpiritReserve()[0];
      let reserve = state.spiritReserve.slice(1);
      if (reserve.length === 0) reserve = generateSpiritReserve();
      return { spirit: next, spiritReserve: reserve };
    }

    const fallSpirit = moveSpirit(state.spirit, 0, 1);
    if (isValidInMatrix(fallSpirit, state.bricks)) {
      return { spirit: fallSpirit };
    }

    // Collision! Lock piece.
    if (!isValidInMatrix(state.spirit, state.bricks)) {
      // If it overlaps immediately upon spawn -> GameOver
      return { gameStatus: GameStatus.GameOver, failureReason: "STRUCTURAL_COLLAPSE" };
    }

    // Process Semantic Modifications
    let { divergencePressure, hallucinationRisk, memorySaturation, systemLogs } = state;
    const type = state.spirit.cognitiveType;
    let log: LogEntry | null = null;

    if (type === "DIVERGENCE_FRAGMENT") {
      divergencePressure += 15;
      log = { id: Date.now().toString(), type: "WARNING", message: "Divergence probability increasing." };
    } else if (type === "ARBITRATION_NODE") {
      divergencePressure = Math.max(0, divergencePressure - 20);
      hallucinationRisk += 10;
      log = { id: Date.now().toString(), type: "ARBITRATION", message: "Stabilization node attached." };
    } else if (type === "MEMORY_CHAIN") {
      memorySaturation += 15;
    } else if (type === "EMBEDDING_CLUSTER") {
      memorySaturation = Math.max(0, memorySaturation - 10);
      hallucinationRisk += 5;
    } else if (type === "ROUTING_PIPELINE") {
      hallucinationRisk = Math.max(0, hallucinationRisk - 15);
      log = { id: Date.now().toString(), type: "ROUTING", message: "Pipeline synchronized." };
    }

    // Check catastrophic thresholds before clearing lines
    if (divergencePressure >= 100) return { gameStatus: GameStatus.GameOver, failureReason: "SEMANTIC COLLAPSE" };
    if (hallucinationRisk >= 100) return { gameStatus: GameStatus.GameOver, failureReason: "HALLUCINATION CASCADE" };
    if (memorySaturation >= 100) return { gameStatus: GameStatus.GameOver, failureReason: "MEMORY FRACTURE" };

    if (log) {
      systemLogs = [log, ...systemLogs].slice(0, 50); // Keep last 50 logs
    }

    // Lock bricks
    const spiritBricks = getSpiritLocation(state.spirit).map(p => createBrick(p.x, p.y));
    let newBricks = [...state.bricks, ...spiritBricks];

    // Check line clears
    const rowCounts = new Map<number, Set<number>>();
    newBricks.forEach(b => {
      if (!rowCounts.has(b.location.y)) rowCounts.set(b.location.y, new Set());
      rowCounts.get(b.location.y)!.add(b.location.x);
    });

    const linesToClear: number[] = [];
    rowCounts.forEach((xSet, y) => {
      if (xSet.size >= MATRIX_WIDTH) linesToClear.push(y);
    });

    linesToClear.sort((a, b) => a - b);
    
    let finalBricks = newBricks;
    if (linesToClear.length > 0) {
      linesToClear.forEach(lineY => {
        finalBricks = finalBricks.filter(b => b.location.y !== lineY);
        finalBricks = finalBricks.map(b => (b.location.y < lineY ? offsetBrick(b, 0, 1) : b));
      });

      // HEAL FROM CLEARING LINES
      const healAmount = linesToClear.length * 30;
      divergencePressure = Math.max(0, divergencePressure - healAmount);
      hallucinationRisk = Math.max(0, hallucinationRisk - healAmount);
      memorySaturation = Math.max(0, memorySaturation - healAmount);
      systemLogs = [{ id: Date.now().toString() + "c", type: "SYSTEM", message: `Orchestration layer stabilized (${linesToClear.length}).` }, ...systemLogs].slice(0, 50);
    }

    // Pull next piece
    let reserve = state.spiritReserve;
    if (reserve.length === 0) reserve = generateSpiritReserve();
    const nextSpirit = reserve[0];
    reserve = reserve.slice(1);

    const isGameOver = !isValidInMatrix(nextSpirit, finalBricks);

    return {
      bricks: finalBricks,
      spirit: isGameOver ? EmptySpirit : nextSpirit,
      spiritReserve: reserve.length === 0 ? generateSpiritReserve() : reserve,
      score: state.score + calculateScore(linesToClear.length) + SCORE_EVERY_SPIRIT,
      line: state.line + linesToClear.length,
      divergencePressure,
      hallucinationRisk,
      memorySaturation,
      systemLogs,
      gameStatus: isGameOver ? GameStatus.GameOver : GameStatus.Running,
      failureReason: isGameOver ? "ROUTING COLLAPSE" : state.failureReason
    };
  })
}));
