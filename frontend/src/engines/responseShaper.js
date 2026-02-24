/**
 * ============================================================
 * Response Shaper — Adaptive Output Transformation Engine
 * ============================================================
 *
 * Transforms raw LLM output before rendering based on:
 *   - User profile preferences (verbosity, analytics visibility)
 *   - Intent complexity classification
 *   - Current mode context
 *   - Analytical trend signals
 *
 * This engine is what makes the system feel adaptive.
 * It does NOT modify model logic — it shapes presentation only.
 *
 * ============================================================
 */

import memoryManager from './memoryManager';
import { detectTaskComplexity, normalizeResponseText } from './responseNormalizer';

/**
 * Shape a response before rendering.
 *
 * @param {Object} params
 * @param {string} params.baseOutput      — Raw response text from backend
 * @param {string} params.mode            — Current mode (standard, experimental)
 * @param {string} params.subMode         — Sub-mode (debate, glass, evidence, null)
 * @param {string} params.userQuery       — Original user query
 * @param {Object} params.responseData    — Full backend response object
 * @returns {Object} Shaped output with rendering directives
 */
export function shapeResponse({ baseOutput, mode, subMode, userQuery, responseData }) {
  const prefs = memoryManager.getUserPreferences();
  const signals = memoryManager.getAnalyticalSignals();
  const complexity = detectTaskComplexity(userQuery);

  // Start with normalized text
  let shapedText = normalizeResponseText(baseOutput || '');

  // ─── Verbosity Adjustment ─────────────────────────────────
  if (prefs.verbosity === 'concise' && shapedText.length > 600) {
    shapedText = truncateToEssentials(shapedText);
  } else if (prefs.verbosity === 'detailed' && shapedText.length < 200 && complexity !== 'simple') {
    // Don't artificially expand — but signal that user prefers detail
    // Backend should be receiving this preference
  }

  // ─── Analytics Visibility Decision ────────────────────────
  const analyticsDecision = computeAnalyticsVisibility({
    complexity,
    mode,
    subMode,
    prefs,
    signals,
    responseData,
  });

  // ─── Structural Layout Decision ───────────────────────────
  const layout = computeLayout({
    complexity,
    mode,
    subMode,
    shapedText,
    responseData,
  });

  return {
    text: shapedText,
    analytics: analyticsDecision,
    layout,
    complexity,
    verbosity: prefs.verbosity,
  };
}

/**
 * Compute what analytics elements should be visible.
 */
function computeAnalyticsVisibility({ complexity, mode, subMode, prefs, signals, responseData }) {
  const meta = responseData?.omega_metadata || {};
  const boundary = responseData?.boundary_result || meta.boundary_result || {};

  // Base decision: what the user profile allows
  const visLevel = prefs.analyticsVisibility; // minimal | standard | forensic

  // Simple queries NEVER show analytics regardless of preference
  if (complexity === 'simple') {
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
    };
  }

  // Single-model modes (qwen, llama70b, groq) — minimal analytics
  if (mode === 'standard' && !subMode) {
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
    };
  }

  // Standard aggregation mode — show based on significance
  if (mode === 'standard') {
    const hasMeaningfulBoundary = boundary.severity_score > 40;
    return {
      showBoundary: hasMeaningfulBoundary && visLevel !== 'minimal',
      showConfidence: visLevel !== 'minimal',
      showFragility: false, // Not relevant for standard
      showAgreement: visLevel !== 'minimal',
      showReasoningTrace: false,
      showBehavioralRisk: false,
      showEvidenceSources: false,
      showStructuredOutput: !!meta.aggregation_result,
      showAnalysisDetails: visLevel === 'forensic',
    };
  }

  // Experimental modes — show based on sub-mode + preference
  if (mode === 'experimental') {
    const isForensic = visLevel === 'forensic';
    const isMinimal = visLevel === 'minimal';

    return {
      showBoundary: !isMinimal && boundary.severity_score > 30,
      showConfidence: !isMinimal,
      showFragility: !isMinimal && meta.fragility_index > 0.1,
      showAgreement: !isMinimal && subMode === 'debate',
      showReasoningTrace: isForensic,
      showBehavioralRisk: isForensic && !!meta.behavioral_risk,
      showEvidenceSources: subMode === 'evidence' && !!meta.evidence_result,
      showStructuredOutput: !!(meta.debate_result || meta.forensic_result || meta.audit_result || meta.aggregation_result),
      showAnalysisDetails: !isMinimal,
    };
  }

  // Default — moderate visibility
  return {
    showBoundary: boundary.severity_score > 40,
    showConfidence: true,
    showFragility: false,
    showAgreement: false,
    showReasoningTrace: false,
    showBehavioralRisk: false,
    showEvidenceSources: false,
    showStructuredOutput: false,
    showAnalysisDetails: false,
  };
}

/**
 * Compute the output layout structure.
 */
function computeLayout({ complexity, mode, subMode, shapedText, responseData }) {
  // Simple tasks — plain text only
  if (complexity === 'simple') {
    return { type: 'plain', sections: ['text'] };
  }

  // Code output
  if (/```[\w]*\n/.test(shapedText)) {
    return { type: 'code', sections: ['text'] };
  }

  // Debate mode
  if (subMode === 'debate') {
    return {
      type: 'structured',
      sections: ['text', 'structuredOutput', 'analytics'],
    };
  }

  // Evidence mode
  if (subMode === 'evidence') {
    return {
      type: 'structured',
      sections: ['text', 'structuredOutput', 'sources'],
    };
  }

  // Glass mode
  if (subMode === 'glass') {
    return {
      type: 'structured',
      sections: ['text', 'structuredOutput', 'analytics'],
    };
  }

  // Standard with analysis
  if (mode === 'standard' && complexity === 'complex') {
    return {
      type: 'analysis',
      sections: ['text', 'structuredOutput'],
    };
  }

  // Default — text with optional analytics
  return {
    type: 'standard',
    sections: ['text'],
  };
}

/**
 * Truncate text to essentials for concise verbosity.
 * Keeps first paragraph + bullet points + conclusion.
 */
function truncateToEssentials(text) {
  const paragraphs = text.split('\n\n').filter(p => p.trim());

  if (paragraphs.length <= 3) return text;

  // Keep first paragraph, any bullet lists, and last paragraph
  const essential = [];
  essential.push(paragraphs[0]);

  for (let i = 1; i < paragraphs.length - 1; i++) {
    if (paragraphs[i].trim().startsWith('•') || paragraphs[i].trim().startsWith('-') || /^\d+\./.test(paragraphs[i].trim())) {
      essential.push(paragraphs[i]);
    }
  }

  essential.push(paragraphs[paragraphs.length - 1]);

  return essential.join('\n\n');
}

/**
 * Get adaptive parameters for backend context injection.
 * These are sent to the backend to influence LLM behavior.
 */
export function getAdaptiveParams(userQuery, mode, subMode) {
  const prefs = memoryManager.getUserPreferences();
  const signals = memoryManager.getAnalyticalSignals();
  const complexity = detectTaskComplexity(userQuery);

  return {
    preferredVerbosity: prefs.verbosity,
    intentComplexity: complexity,
    analyticsVisibility: prefs.analyticsVisibility,
    citationBias: prefs.citationBias,
    // Adaptive signals
    reduceDebateDepth: signals.shouldReduceDebateDepth,
    increaseExplanation: signals.shouldIncreaseExplanation,
    elevateCitationStrictness: signals.shouldElevateCitationStrictness,
  };
}

const responseShaper = { shapeResponse, getAdaptiveParams };
export default responseShaper;
