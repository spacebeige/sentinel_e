/**
 * ============================================================
 * Cognitive Governor — Pre-Output Self-Governance Engine
 * ============================================================
 *
 * This engine evaluates EVERY response before it reaches the user.
 * It implements Sections IX and XII of the Cognitive Core Spec:
 *
 * Section IX — Error Correction Mechanism:
 *   If inconsistency is detected between current response and
 *   stored conversation state → reconcile before output.
 *   Never propagate contradiction.
 *
 * Section XII — Self-Governance:
 *   Before generating final output, internally evaluate:
 *   - Is my answer contextually coherent?
 *   - Does it respect prior clarifications?
 *   - Is retrieval necessary?
 *   - Am I over-explaining?
 *   - Am I ignoring user preference?
 *   - Is mode behavior aligned?
 *
 * Section V — Epistemic Need Detection:
 *   Determine if retrieval is needed based on epistemic need,
 *   not keywords. Evaluate: time-sensitivity, knowledge gap,
 *   external grounding need, consequence of inaccuracy.
 *
 * Section VII — Complexity Awareness:
 *   Estimate cognitive complexity, risk of error, dependency on
 *   prior context, knowledge uncertainty.
 *
 * ============================================================
 */

import memoryManager from './memoryManager';
import { detectTaskComplexity } from './responseNormalizer';

// ─── Pre-Output Governance Evaluation ───────────────────────

/**
 * Evaluate a response before rendering.
 * Returns a governance verdict with flags and optional fixes.
 *
 * @param {Object} params
 * @param {string} params.userQuery     — The user's input
 * @param {string} params.responseText  — The response text from backend
 * @param {Object} params.responseData  — Full response object
 * @param {string} params.mode          — Current mode
 * @param {string} params.subMode       — Current sub-mode
 * @returns {Object} Governance verdict
 */
export function evaluateResponse({ userQuery, responseText, responseData, mode, subMode }) {
  const cs = memoryManager.getCognitiveState();
  const prefs = memoryManager.getUserPreferences();
  const analyticalSignals = memoryManager.getAnalyticalSignals();
  const constraints = memoryManager.getActiveConstraints();
  const complexity = detectTaskComplexity(userQuery);

  const verdict = {
    pass: true,
    flags: [],
    suggestions: [],
    complexityAssessment: complexity,
    epistemicNeed: null,
    modeAlignment: true,
    constraintViolations: [],
  };

  // ─── 1. Contextual Coherence Check ──────────────────────
  if (cs.activeEntity && responseText) {
    // If active entity exists but response doesn't reference it or related terms
    // and user query was about it → flag potential coherence issue
    const queryReferencesEntity = userQuery &&
      (userQuery.toLowerCase().includes(cs.activeEntity.toLowerCase()) ||
       /\b(it|they|he|she|this|that)\b/i.test(userQuery));
    const responseReferencesEntity = responseText.toLowerCase().includes(cs.activeEntity.toLowerCase());

    if (queryReferencesEntity && !responseReferencesEntity && responseText.length > 100) {
      verdict.flags.push('entity_drift');
      verdict.suggestions.push(`Response may not address active entity: "${cs.activeEntity}"`);
    }
  }

  // ─── 2. Constraint Respect Check (Section VIII) ─────────
  if (Object.keys(constraints).length > 0 && responseText) {
    const rt = responseText.toLowerCase();

    // Check language constraint
    if (constraints.language && !rt.includes(constraints.language)) {
      // Only flag if this looks like a code generation task
      if (cs.activeOperation === 'create' || cs.activeOperation === 'fix') {
        verdict.constraintViolations.push(`language=${constraints.language}`);
        verdict.flags.push('constraint_violation');
      }
    }

    // Check style constraint
    if (constraints.style === 'concise' && responseText.length > 800) {
      verdict.flags.push('verbosity_violation');
      verdict.suggestions.push('User requested concise output but response is long');
    }
    if (constraints.style === 'detailed' && responseText.length < 150 && complexity !== 'simple') {
      verdict.flags.push('depth_violation');
      verdict.suggestions.push('User requested detailed output but response is brief');
    }

    // Check exclusion constraint
    if (constraints.exclusion && rt.includes(constraints.exclusion)) {
      verdict.constraintViolations.push(`exclusion=${constraints.exclusion}`);
      verdict.flags.push('constraint_violation');
    }
  }

  // ─── 3. Over-Analysis Detection (Section XII) ──────────
  if (complexity === 'simple' && responseText && responseText.length > 500) {
    verdict.flags.push('over_analysis');
    verdict.suggestions.push('Simple query received analysis-grade response');
  }

  // ─── 4. Under-Analysis Detection (Section VII) ─────────
  if (complexity === 'complex' && responseText && responseText.length < 100) {
    verdict.flags.push('under_analysis');
    verdict.suggestions.push('Complex query received minimal response');
  }

  // ─── 5. Mode Alignment Check (Section VI) ──────────────
  const meta = responseData?.omega_metadata || {};
  if (mode === 'experimental' && subMode === 'debate' && !meta.debate_result) {
    verdict.modeAlignment = false;
    verdict.flags.push('mode_misalignment');
  }
  if (mode === 'experimental' && subMode === 'evidence' && !meta.evidence_result) {
    verdict.modeAlignment = false;
    verdict.flags.push('mode_misalignment');
  }

  // ─── 6. User Preference Alignment ─────────────────────
  if (prefs.verbosity === 'concise' && responseText && responseText.length > 1000) {
    verdict.flags.push('preference_violation');
    verdict.suggestions.push('User prefers concise responses');
  }
  if (prefs.citationBias && mode === 'experimental' && subMode === 'evidence') {
    if (responseText && !(/\[\d+\]|source|citation|according to/i.test(responseText))) {
      verdict.flags.push('citation_gap');
      verdict.suggestions.push('User prefers cited responses but none found');
    }
  }

  // ─── 7. Epistemic Need Assessment (Section V) ──────────
  verdict.epistemicNeed = assessEpistemicNeed(userQuery, cs, complexity);

  // ─── 8. Contradiction Detection (Section IX) ──────────
  if (responseText) {
    const contradictions = detectContradictions(responseText, memoryManager);
    if (contradictions.length > 0) {
      verdict.flags.push('contradiction_detected');
      verdict.suggestions.push(...contradictions);
    }
  }

  // ─── 9. Analytical Instability Warning ─────────────────
  if (analyticalSignals.instabilityCount > 3) {
    verdict.flags.push('high_instability');
    verdict.suggestions.push('Session shows high opinion instability — consider anchoring response');
  }

  // Set overall pass/fail
  const criticalFlags = ['contradiction_detected', 'constraint_violation'];
  verdict.pass = !verdict.flags.some(f => criticalFlags.includes(f));

  return verdict;
}

// ─── Epistemic Need Assessment (Section V) ──────────────────

/**
 * Determine if retrieval/RAG is needed based on epistemic evaluation,
 * NOT keyword triggers.
 *
 * Evaluates:
 *   - Is this question time-sensitive?
 *   - Is internal knowledge likely insufficient?
 *   - Does the question require external grounding?
 *   - Is there high consequence for inaccuracy?
 *
 * @returns {Object} { needed: boolean, reason: string, confidence: number }
 */
export function assessEpistemicNeed(query, cognitiveState, complexity) {
  if (!query) return { needed: false, reason: 'no_query', confidence: 1.0 };

  const q = query.toLowerCase();
  let score = 0;
  let reasons = [];

  // Time-sensitivity: current events, recent data, "latest", "today"
  if (/\b(latest|recent|current|today|this\s+(?:week|month|year)|2024|2025|2026|now)\b/i.test(q)) {
    score += 0.4;
    reasons.push('time_sensitive');
  }

  // External grounding: statistics, data, specific facts
  if (/\b(statistics|data|study|research|paper|survey|report|percent|%)\b/i.test(q)) {
    score += 0.3;
    reasons.push('external_grounding');
  }

  // High-consequence domains
  if (/\b(medical|legal|financial|safety|security|regulatory|compliance)\b/i.test(q)) {
    score += 0.35;
    reasons.push('high_consequence');
  }

  // Verifiability: fact-checking, citations needed
  if (/\b(prove|verify|fact|source|citation|evidence|true|false)\b/i.test(q)) {
    score += 0.3;
    reasons.push('verification_needed');
  }

  // Complexity boost
  if (complexity === 'complex') {
    score += 0.15;
    reasons.push('high_complexity');
  }

  // Epistemic confidence reduction: if the system has low confidence
  // in understanding what's being asked, retrieval helps ground the answer
  if (cognitiveState && cognitiveState.epistemicConfidence < 0.4) {
    score += 0.2;
    reasons.push('low_epistemic_confidence');
  }

  // Internal knowledge likely sufficient for these
  if (/\b(write\s+code|implement|create\s+a|build\s+a|hello|hi|thanks)\b/i.test(q)) {
    score -= 0.3;
  }

  const needed = score >= 0.4;

  return {
    needed,
    reason: reasons.length > 0 ? reasons.join(', ') : 'none',
    confidence: Math.max(0, Math.min(1, 1 - score)),
    score: Math.max(0, Math.min(1, score)),
  };
}

// ─── Contradiction Detection (Section IX) ───────────────────

/**
 * Check if the response contradicts stored conversation state.
 * Returns array of contradiction descriptions.
 */
function detectContradictions(responseText, memory) {
  const contradictions = [];
  const constraints = memory.getActiveConstraints();
  const cogState = memory.getCognitiveState();
  const rt = responseText.toLowerCase();

  // Check if response contradicts explicit constraints
  if (constraints.exclusion && rt.includes(constraints.exclusion.toLowerCase())) {
    contradictions.push(
      `Response includes excluded content: "${constraints.exclusion}"`
    );
  }

  // Check if response contradicts the active objective
  if (cogState.activeObjective && rt.length > 50) {
    const objectiveWords = cogState.activeObjective.toLowerCase().split(/\s+/).filter(w => w.length > 3);
    const negationPattern = /\b(not|never|don't|won't|shouldn't|can't|cannot|impossible)\b/i;
    if (negationPattern.test(rt) && objectiveWords.some(w => rt.includes(w))) {
      // Response negates something related to the user's objective — worth flagging
      contradictions.push(
        `Response may contradict active objective: "${cogState.activeObjective.substring(0, 60)}"`
      );
    }
  }

  // Check if response uses overridden claims from graph
  const overriddenClaims = memory.graph.nodes.filter(n =>
    n.type === 'claim' && !n.active
  );
  for (const claim of overriddenClaims.slice(-5)) {
    if (textOverlapRatio(claim.content, responseText) > 0.5) {
      contradictions.push(
        `Response may reference overridden claim: "${claim.content.substring(0, 60)}..."`
      );
    }
  }

  return contradictions;
}

/**
 * Simple text overlap for contradiction detection.
 */
function textOverlapRatio(a, b) {
  if (!a || !b) return 0;
  const wordsA = new Set(a.toLowerCase().split(/\s+/).filter(w => w.length > 3));
  const wordsB = new Set(b.toLowerCase().split(/\s+/).filter(w => w.length > 3));
  if (wordsA.size === 0) return 0;
  let matches = 0;
  for (const w of wordsA) {
    if (wordsB.has(w)) matches++;
  }
  return matches / wordsA.size;
}

// ─── Complexity-Aware Strategy (Section VII) ─────────────────

/**
 * Determine the appropriate reasoning strategy based on complexity.
 * This influences how the frontend treats the response.
 *
 * @returns {Object} Strategy recommendation
 */
export function getReasoningStrategy(userQuery) {
  const complexity = detectTaskComplexity(userQuery);
  const cs = memoryManager.getCognitiveState();
  const epistemic = assessEpistemicNeed(userQuery, cs, complexity);

  if (complexity === 'simple') {
    return {
      strategy: 'direct',
      showAnalytics: false,
      showStructured: false,
      suggestRetrieval: false,
      suggestMultiModel: false,
    };
  }

  if (complexity === 'moderate') {
    return {
      strategy: 'structured',
      showAnalytics: false,
      showStructured: true,
      suggestRetrieval: epistemic.needed,
      suggestMultiModel: false,
    };
  }

  // Complex
  return {
    strategy: 'deep',
    showAnalytics: true,
    showStructured: true,
    suggestRetrieval: epistemic.needed,
    suggestMultiModel: cs.epistemicConfidence < 0.6,
  };
}

const cognitiveGovernor = {
  evaluateResponse,
  assessEpistemicNeed,
  getReasoningStrategy,
};

export default cognitiveGovernor;
