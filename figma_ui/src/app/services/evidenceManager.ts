// ============================================================
// Evidence Manager — source tracking + contradiction analysis
// ============================================================

import type { OmegaMetadata, EvidenceResult, EvidenceSource } from "../api";

export interface EvidenceState {
  /** All sources collected across messages */
  allSources: EvidenceSource[];
  /** Unique domains seen */
  domains: Set<string>;
  /** Contradictions accumulated */
  contradictions: Array<Record<string, unknown>>;
  /** Confidence evolution across evidence queries */
  confidenceTrace: number[];
  /** Source agreement trace */
  agreementTrace: number[];
  /** Data lineage chain */
  lineage: Array<Record<string, string>>;
  /** Summary metrics */
  summary: {
    totalSources: number;
    uniqueDomains: number;
    totalContradictions: number;
    averageReliability: number;
    averageAgreement: number;
    searchesExecuted: number;
  };
}

const EMPTY_EVIDENCE: EvidenceState = {
  allSources: [],
  domains: new Set(),
  contradictions: [],
  confidenceTrace: [],
  agreementTrace: [],
  lineage: [],
  summary: {
    totalSources: 0,
    uniqueDomains: 0,
    totalContradictions: 0,
    averageReliability: 0,
    averageAgreement: 0,
    searchesExecuted: 0,
  },
};

function avg(arr: number[]): number {
  return arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

/**
 * Merge new Omega response into Evidence state.
 */
export function mergeEvidenceState(
  current: EvidenceState,
  metadata: OmegaMetadata | undefined
): EvidenceState {
  if (!metadata?.evidence_result) return current;
  const er = metadata.evidence_result;

  const newSources = [...current.allSources, ...er.sources];
  const newDomains = new Set(current.domains);
  er.sources.forEach((s) => {
    if (s.domain) newDomains.add(s.domain);
  });

  const newContradictions = [...current.contradictions, ...er.contradictions];
  const newConfidence = er.evidence_confidence
    ? [...current.confidenceTrace, er.evidence_confidence]
    : current.confidenceTrace;
  const newAgreement = er.source_agreement
    ? [...current.agreementTrace, er.source_agreement]
    : current.agreementTrace;
  const newLineage = [...current.lineage, ...er.lineage];
  const searchesExecuted = current.summary.searchesExecuted + (er.search_executed ? 1 : 0);

  return {
    allSources: newSources,
    domains: newDomains,
    contradictions: newContradictions,
    confidenceTrace: newConfidence,
    agreementTrace: newAgreement,
    lineage: newLineage,
    summary: {
      totalSources: newSources.length,
      uniqueDomains: newDomains.size,
      totalContradictions: newContradictions.length,
      averageReliability: avg(newSources.map((s) => s.reliability_score)),
      averageAgreement: avg(newAgreement),
      searchesExecuted,
    },
  };
}

/**
 * Create fresh Evidence state.
 */
export function createEvidenceState(): EvidenceState {
  return {
    ...EMPTY_EVIDENCE,
    domains: new Set(),
  };
}

/**
 * Get top sources by reliability.
 */
export function getTopSources(state: EvidenceState, limit = 10): EvidenceSource[] {
  return [...state.allSources]
    .sort((a, b) => b.reliability_score - a.reliability_score)
    .slice(0, limit);
}

/**
 * Check for contradictions between specific sources.
 */
export function hasContradictions(state: EvidenceState): boolean {
  return state.contradictions.length > 0;
}
