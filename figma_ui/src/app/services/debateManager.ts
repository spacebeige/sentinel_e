// ============================================================
// Debate Manager — tracks multi-round debate state
// ============================================================

import type { DebateResult, OmegaMetadata } from "../api";

export interface DebatePosition {
  model: string;
  position: string;
  confidence: number;
  key_points: string[];
}

export interface DebateRound {
  round: number;
  positions: DebatePosition[];
  consensus?: string;
  timestamp: string;
}

export interface DebateState {
  rounds: DebateRound[];
  totalRounds: number;
  currentRound: number;
  isComplete: boolean;
  overallConsensus: string | null;
  disagreementMetrics: {
    averageDisagreement: number;
    maxDisagreement: number;
    convergenceTrend: number[];
  };
}

const EMPTY_STATE: DebateState = {
  rounds: [],
  totalRounds: 0,
  currentRound: 0,
  isComplete: false,
  overallConsensus: null,
  disagreementMetrics: {
    averageDisagreement: 0,
    maxDisagreement: 0,
    convergenceTrend: [],
  },
};

/**
 * Extract a DebateRound from an Omega response.
 */
export function extractDebateRound(
  metadata: OmegaMetadata | undefined,
  existingRounds: DebateRound[]
): DebateRound | null {
  if (!metadata?.debate_result) return null;
  const dr = metadata.debate_result;
  const round: DebateRound = {
    round: existingRounds.length + 1,
    positions: Array.isArray(dr.positions)
      ? dr.positions.map((p) => ({
          model: String(p.model ?? ""),
          position: String(p.position ?? ""),
          confidence: Number(p.confidence ?? 0),
          key_points: Array.isArray(p.key_points) ? p.key_points.map(String) : [],
        }))
      : [],
    consensus: typeof dr.consensus === "string" ? dr.consensus : undefined,
    timestamp: new Date().toISOString(),
  };
  return round;
}

/**
 * Compute disagreement between positions.
 */
function computeDisagreement(positions: DebatePosition[]): number {
  if (positions.length < 2) return 0;
  const confidences = positions.map((p) => p.confidence);
  const max = Math.max(...confidences);
  const min = Math.min(...confidences);
  return max - min; // simple spread metric
}

/**
 * Build full DebateState from accumulated rounds.
 */
export function buildDebateState(
  rounds: DebateRound[],
  totalRoundsRequested: number
): DebateState {
  if (rounds.length === 0) return { ...EMPTY_STATE, totalRounds: totalRoundsRequested };

  const disagreements = rounds.map((r) => computeDisagreement(r.positions));
  const lastRound = rounds[rounds.length - 1];

  return {
    rounds,
    totalRounds: totalRoundsRequested,
    currentRound: rounds.length,
    isComplete: rounds.length >= totalRoundsRequested || !!lastRound.consensus,
    overallConsensus: lastRound.consensus || null,
    disagreementMetrics: {
      averageDisagreement:
        disagreements.length > 0
          ? disagreements.reduce((a, b) => a + b, 0) / disagreements.length
          : 0,
      maxDisagreement: disagreements.length > 0 ? Math.max(...disagreements) : 0,
      convergenceTrend: disagreements,
    },
  };
}

/**
 * Merge new debate result into existing state.
 */
export function mergeDebateResult(
  current: DebateState,
  metadata: OmegaMetadata | undefined
): DebateState {
  const newRound = extractDebateRound(metadata, current.rounds);
  if (!newRound) return current;
  const rounds = [...current.rounds, newRound];
  return buildDebateState(rounds, current.totalRounds);
}

/**
 * Create a fresh debate state.
 */
export function createDebateState(totalRounds: number): DebateState {
  return { ...EMPTY_STATE, totalRounds };
}
