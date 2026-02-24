/**
 * ============================================================
 * Context Injector — Stateful Context Builder for API Calls
 * ============================================================
 *
 * Builds the context payload that gets injected into FormData
 * before every backend API call. This is how the frontend's
 * memory state flows into the LLM's conversation history.
 *
 * Responsibilities:
 *   - Build context JSON from memoryManager state
 *   - Resolve pronouns and follow-up intent
 *   - Maintain subject continuity across turns
 *   - Get adaptive parameters for backend behavior tuning
 *
 * ============================================================
 */

import memoryManager from './memoryManager';
import { getAdaptiveParams } from './responseShaper';
import { assessEpistemicNeed } from './cognitiveGovernor';

/**
 * Build the full context payload for a backend API call.
 *
 * @param {string} userQuery — The user's raw query text
 * @param {string} mode      — Current mode (standard, experimental)
 * @param {string} subMode   — Current sub-mode (debate, glass, evidence, null)
 * @returns {Object} Context object to be JSON-serialized into FormData
 */
export function buildContextPayload(userQuery, mode, subMode) {
  const bundle = memoryManager.buildContextBundle(userQuery, mode, subMode);
  const adaptive = getAdaptiveParams(userQuery, mode, subMode);
  const cs = bundle.cognitiveState || {};

  // Epistemic need assessment — tells backend if retrieval is warranted
  const epistemic = assessEpistemicNeed(userQuery, cs, bundle.currentMode);

  return {
    // Cognitive state (Section II — active entity, operation, objective, constraints)
    cognitiveState: {
      activeEntity: cs.activeEntity,
      activeOperation: cs.activeOperation,
      activeObjective: cs.activeObjective,
      epistemicConfidence: cs.epistemicConfidence,
      intentTrajectory: (cs.intentTrajectory || []).slice(-5),
      activeConstraints: cs.activeConstraints || {},
      activeDependencies: (cs.activeDependencies || []).slice(-5),
    },

    // Short-term memory context
    shortTerm: {
      sessionId: bundle.shortTerm.sessionId,
      activeEntity: bundle.shortTerm.activeEntity,
      activeTopic: bundle.shortTerm.activeTopic,
      lastIntent: bundle.shortTerm.lastIntent,
      isFollowUp: bundle.shortTerm.isFollowUp,
      resolvedQuery: bundle.shortTerm.resolvedQuery,
      implicitEntity: bundle.shortTerm.implicitEntity || null,
      // Send last 5 messages as context (compact)
      recentMessages: (bundle.shortTerm.recentMessages || []).slice(-5).map(m => ({
        role: m.role,
        content: m.content?.substring(0, 300) || '', // Truncate for bandwidth
      })),
    },

    // Epistemic need assessment (Section V — tells backend if RAG should trigger)
    epistemicNeed: {
      needed: epistemic.needed,
      reason: epistemic.reason,
      confidence: epistemic.confidence,
    },

    // Analytical trend signals
    analytical: bundle.analytical,

    // User preference signals for backend response tuning
    preferences: {
      verbosity: bundle.userProfile.verbosity,
      analyticsVisibility: bundle.userProfile.analyticsVisibility,
      citationBias: bundle.userProfile.citationBias,
    },

    // Adaptive parameters from response shaper
    adaptive,
  };
}

/**
 * Inject context into a FormData object before sending to backend.
 *
 * @param {FormData} formData  — The FormData being built for the API call
 * @param {string} userQuery   — The user's raw query text
 * @param {string} mode        — Current mode
 * @param {string} subMode     — Current sub-mode
 * @returns {FormData} The same FormData with context appended
 */
export function injectContext(formData, userQuery, mode, subMode) {
  try {
    const context = buildContextPayload(userQuery, mode, subMode);
    formData.append('context', JSON.stringify(context));
  } catch (err) {
    console.warn('[ContextInjector] Failed to build context:', err);
    // Don't block the API call if context building fails
  }
  return formData;
}

/**
 * Get a resolved query that accounts for pronoun resolution
 * and follow-up intent. Use this for display purposes or
 * for sending a clarified query to the backend.
 *
 * @param {string} userQuery — Raw user input
 * @param {string} mode
 * @param {string} subMode
 * @returns {string} Resolved query (or original if no resolution needed)
 */
export function getResolvedQuery(userQuery, mode, subMode) {
  const bundle = memoryManager.buildContextBundle(userQuery, mode, subMode);
  return bundle.shortTerm.resolvedQuery || userQuery;
}

const contextInjector = {
  buildContextPayload,
  injectContext,
  getResolvedQuery,
};

export default contextInjector;
