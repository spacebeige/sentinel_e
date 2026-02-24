/**
 * ============================================================
 * Analytics Visibility Controller
 * ============================================================
 *
 * Centralized gate for all analytics display decisions.
 * Nothing in the render layer should decide on its own
 * whether to show analytics — this controller decides.
 *
 * Decision factors:
 *   1. Query complexity (simple → never show)
 *   2. User profile preference (minimal/standard/forensic)
 *   3. Mode context (standard vs experimental sub-modes)
 *   4. Metric significance (suppress noise, elevate signals)
 *
 * ============================================================
 */

import memoryManager from './memoryManager';
import { detectTaskComplexity } from './responseNormalizer';

/**
 * Master visibility decision for a given message.
 *
 * @param {Object} params
 * @param {string} params.userQuery       — The user's query text
 * @param {string} params.mode            — standard | experimental
 * @param {string} params.subMode         — debate | glass | evidence | null
 * @param {Object} params.responseData    — Full response from backend
 * @returns {Object} Visibility flags for each analytics section
 */
export function getVisibility({ userQuery, mode, subMode, responseData }) {
  const prefs = memoryManager.getUserPreferences();
  const complexity = detectTaskComplexity(userQuery || '');
  const meta = responseData?.omega_metadata || {};
  const boundary = responseData?.boundary_result || meta.boundary_result || {};
  const visLevel = prefs.analyticsVisibility; // minimal | standard | forensic

  // ─── Rule 1: Simple queries → NO analytics ever ──────────
  if (complexity === 'simple') {
    return createNullVisibility();
  }

  // ─── Rule 2: Single model mode → NO analytics ────────────
  const singleModels = ['qwen', 'llama70b', 'groq'];
  if (singleModels.includes(mode)) {
    return createNullVisibility();
  }

  // ─── Rule 3: Minimal preference → only critical alerts ───
  if (visLevel === 'minimal') {
    return createMinimalVisibility(boundary, meta);
  }

  // ─── Rule 4: Mode-specific visibility ─────────────────────
  if (mode === 'standard') {
    return createStandardVisibility(complexity, boundary, meta, visLevel);
  }

  if (mode === 'experimental') {
    return createExperimentalVisibility(complexity, subMode, boundary, meta, visLevel);
  }

  // Default fallback
  return createStandardVisibility(complexity, boundary, meta, visLevel);
}

/**
 * Check if ANY analytics should render for a message.
 * Quick gate to avoid rendering empty analytics containers.
 */
export function hasAnyVisibleAnalytics(visibilityFlags) {
  if (!visibilityFlags) return false;
  return Object.values(visibilityFlags).some(v => v === true);
}

/**
 * Get visibility for the session panel / right panel.
 */
export function getSessionPanelVisibility() {
  const prefs = memoryManager.getUserPreferences();
  return {
    showTrendCharts: prefs.analyticsVisibility !== 'minimal',
    showRiskHistory: prefs.analyticsVisibility === 'forensic',
    showModelPerformance: prefs.analyticsVisibility !== 'minimal',
    showFeedbackSummary: true, // Always show feedback summary
  };
}

// ─── Visibility Generators ──────────────────────────────────

function createNullVisibility() {
  return {
    showBoundary: false,
    showConfidence: false,
    showFragility: false,
    showAgreement: false,
    showReasoningTrace: false,
    showBehavioralRisk: false,
    showEvidenceSources: false,
    showStructuredOutput: false,
    showAnalysisDetails: false,
    showInsightsSummary: false,
  };
}

function createMinimalVisibility(boundary, meta) {
  // Only show if severity is critical (> 70)
  const criticalBoundary = (boundary?.severity_score || 0) > 70;
  return {
    showBoundary: criticalBoundary,
    showConfidence: false,
    showFragility: false,
    showAgreement: false,
    showReasoningTrace: false,
    showBehavioralRisk: false,
    showEvidenceSources: false,
    showStructuredOutput: false,
    showAnalysisDetails: false,
    showInsightsSummary: false,
  };
}

function createStandardVisibility(complexity, boundary, meta, visLevel) {
  const isForensic = visLevel === 'forensic';
  const hasBoundary = (boundary?.severity_score || 0) > 30;
  const hasAggregation = !!meta.aggregation_result;
  const confidence = meta.confidence ?? meta.weighted_confidence ?? null;

  return {
    showBoundary: hasBoundary,
    showConfidence: confidence !== null,
    showFragility: false,
    showAgreement: hasAggregation,
    showReasoningTrace: isForensic,
    showBehavioralRisk: false,
    showEvidenceSources: false,
    showStructuredOutput: hasAggregation,
    showAnalysisDetails: isForensic,
    showInsightsSummary: complexity === 'complex',
  };
}

function createExperimentalVisibility(complexity, subMode, boundary, meta, visLevel) {
  const isForensic = visLevel === 'forensic';
  const hasBoundary = (boundary?.severity_score || 0) > 20;

  // Debate mode
  if (subMode === 'debate') {
    return {
      showBoundary: hasBoundary,
      showConfidence: true,
      showFragility: (meta.fragility_index || 0) > 0.1,
      showAgreement: true,
      showReasoningTrace: isForensic,
      showBehavioralRisk: isForensic,
      showEvidenceSources: false,
      showStructuredOutput: !!meta.debate_result,
      showAnalysisDetails: true,
      showInsightsSummary: true,
    };
  }

  // Evidence mode
  if (subMode === 'evidence') {
    return {
      showBoundary: hasBoundary,
      showConfidence: true,
      showFragility: false,
      showAgreement: false,
      showReasoningTrace: isForensic,
      showBehavioralRisk: false,
      showEvidenceSources: !!meta.evidence_result,
      showStructuredOutput: !!meta.forensic_result || !!meta.audit_result,
      showAnalysisDetails: true,
      showInsightsSummary: true,
    };
  }

  // Glass mode
  if (subMode === 'glass') {
    return {
      showBoundary: hasBoundary,
      showConfidence: true,
      showFragility: (meta.fragility_index || 0) > 0.05,
      showAgreement: true,
      showReasoningTrace: true, // Glass is inherently transparent
      showBehavioralRisk: isForensic,
      showEvidenceSources: false,
      showStructuredOutput: true,
      showAnalysisDetails: true,
      showInsightsSummary: true,
    };
  }

  // Default experimental
  return {
    showBoundary: hasBoundary,
    showConfidence: true,
    showFragility: (meta.fragility_index || 0) > 0.1,
    showAgreement: false,
    showReasoningTrace: isForensic,
    showBehavioralRisk: false,
    showEvidenceSources: false,
    showStructuredOutput: !!meta.aggregation_result,
    showAnalysisDetails: isForensic,
    showInsightsSummary: complexity === 'complex',
  };
}

const analyticsVisibilityController = {
  getVisibility,
  hasAnyVisibleAnalytics,
  getSessionPanelVisibility,
};

export default analyticsVisibilityController;
