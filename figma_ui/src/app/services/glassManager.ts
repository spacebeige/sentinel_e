// ============================================================
// Glass Manager — tracks confidence pipeline + behavioral metrics
// ============================================================

import type { OmegaMetadata, ConfidenceEvolution, BehavioralRiskProfile, StressResult } from "../api";

export interface GlassState {
  /** Kill override disables adaptive learning and shows raw inference */
  killOverride: boolean;
  /** Confidence pipeline trace across messages */
  confidenceHistory: ConfidenceEvolution[];
  /** Fragility index history */
  fragilityHistory: number[];
  /** Boundary severity trace */
  boundaryTrace: number[];
  /** Behavioral analytics history */
  behavioralHistory: BehavioralRiskProfile[];
  /** Stress test results */
  stressHistory: StressResult[];
  /** Session intelligence summary */
  sessionMetrics: {
    averageConfidence: number;
    averageFragility: number;
    averageBoundarySeverity: number;
    peakRisk: string;
    totalMessages: number;
  };
}

const EMPTY_GLASS: GlassState = {
  killOverride: false,
  confidenceHistory: [],
  fragilityHistory: [],
  boundaryTrace: [],
  behavioralHistory: [],
  stressHistory: [],
  sessionMetrics: {
    averageConfidence: 0,
    averageFragility: 0,
    averageBoundarySeverity: 0,
    peakRisk: "LOW",
    totalMessages: 0,
  },
};

function avg(arr: number[]): number {
  return arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function peakRiskLevel(behavioral: BehavioralRiskProfile[]): string {
  const levels = behavioral.map((b) => b.risk_level);
  if (levels.includes("CRITICAL")) return "CRITICAL";
  if (levels.includes("HIGH")) return "HIGH";
  if (levels.includes("MEDIUM")) return "MEDIUM";
  return "LOW";
}

/**
 * Merge new Omega response into Glass state.
 */
export function mergeGlassState(
  current: GlassState,
  metadata: OmegaMetadata | undefined
): GlassState {
  if (!metadata) return current;

  const confidenceHistory = metadata.confidence_evolution
    ? [...current.confidenceHistory, metadata.confidence_evolution]
    : current.confidenceHistory;

  const fragilityHistory =
    metadata.fragility_index != null
      ? [...current.fragilityHistory, metadata.fragility_index]
      : current.fragilityHistory;

  const boundaryTrace =
    metadata.boundary_result?.severity_score != null
      ? [...current.boundaryTrace, metadata.boundary_result.severity_score]
      : current.boundaryTrace;

  const behavioralHistory = metadata.behavioral_risk
    ? [...current.behavioralHistory, metadata.behavioral_risk]
    : current.behavioralHistory;

  const stressHistory = metadata.stress_result
    ? [...current.stressHistory, metadata.stress_result]
    : current.stressHistory;

  const totalMessages = current.sessionMetrics.totalMessages + 1;

  return {
    killOverride: current.killOverride,
    confidenceHistory,
    fragilityHistory,
    boundaryTrace,
    behavioralHistory,
    stressHistory,
    sessionMetrics: {
      averageConfidence: avg(confidenceHistory.map((c) => c.final)),
      averageFragility: avg(fragilityHistory),
      averageBoundarySeverity: avg(boundaryTrace),
      peakRisk: peakRiskLevel(behavioralHistory),
      totalMessages,
    },
  };
}

/**
 * Toggle kill override.
 */
export function toggleKillOverride(current: GlassState): GlassState {
  return { ...current, killOverride: !current.killOverride };
}

/**
 * Create fresh Glass state.
 */
export function createGlassState(): GlassState {
  return { ...EMPTY_GLASS };
}
