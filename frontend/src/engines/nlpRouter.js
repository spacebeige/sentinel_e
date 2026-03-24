/**
 * ============================================================
 * NLP Router — Intelligent Query Routing & Classification
 * ============================================================
 *
 * Handles:
 *   - Linguistic analysis of user queries
 *   - Grievance classification (GR-01 to GR-08)
 *   - Intent detection (complaint, question, feedback)
 *   - Mode/SubMode recommendation
 *   - Language preservation
 *   - Domain-specific routing
 */

export const GRIEVANCE_CATEGORIES = {
  'GR-01': {
    name: 'Water Supply',
    keywords: ['water', 'tap', 'jal', 'नल', 'తాగుచు'],
    modes: ['standard'],
  },
  'GR-02': {
    name: 'Sanitation & Sewerage',
    keywords: ['sewerage', 'sanitation', 'waste', 'svachhata', 'स्वच्छता'],
    modes: ['standard'],
  },
  'GR-03': {
    name: 'Road Infrastructure',
    keywords: ['road', 'pothole', 'street', 'सड़क', 'రోడ్డు'],
    modes: ['standard'],
  },
  'GR-04': {
    name: 'Public Health',
    keywords: ['health', 'hospital', 'medicines', 'स्वास्थ्य', 'ఆరోగ్యం'],
    modes: ['standard'],
  },
  'GR-05': {
    name: 'Electricity',
    keywords: ['electricity', 'power', 'light', 'बिजली', 'విద్యుత్'],
    modes: ['standard'],
  },
  'GR-06': {
    name: 'Education',
    keywords: ['education', 'school', 'college', 'शिक्षा', 'విద్య'],
    modes: ['standard'],
  },
  'GR-07': {
    name: 'Documents & Permits',
    keywords: ['document', 'certificate', 'permit', 'नागरिकता', 'లిఖితం'],
    modes: ['standard'],
  },
  'GR-08': {
    name: 'Other',
    keywords: ['other', 'miscellaneous'],
    modes: ['standard'],
  },
};

export const INTENT_TYPES = {
  COMPLAINT: 'complaint',
  QUESTION: 'question',
  FEEDBACK: 'feedback',
  SUGGESTION: 'suggestion',
  GENERAL: 'general',
};

/**
 * Analyze query and route to appropriate model
 *
 * @param {string} text - Transcribed user query
 * @param {string} language - Detected language code (e.g., 'hi', 'ta', 'en')
 * @returns {Object} - Routing decision with recommended mode, grievance category, intent, etc.
 */
export function analyzeAndRoute(text, language = 'en') {
  if (!text || !text.trim()) {
    return {
      error: 'Empty query',
      recommended_mode: 'standard',
      subMode: null,
      confidence: 0,
    };
  }

  const normalized = text.toLowerCase().trim();

  // ─────────────────────────────────────────────────────────
  // 1. DETECT INTENT
  // ─────────────────────────────────────────────────────────
  const intent = detectIntent(normalized);

  // ─────────────────────────────────────────────────────────
  // 2. CLASSIFY GRIEVANCE
  // ─────────────────────────────────────────────────────────
  const grievance = classifyGrievance(normalized, language);

  // ─────────────────────────────────────────────────────────
  // 3. DETERMINE BEST MODE & SUB-MODE
  // ─────────────────────────────────────────────────────────
  const routing = determineMode(
    intent,
    grievance.category,
    language,
    normalized
  );

  return {
    query: text,
    language,
    detected_intent: intent,
    detected_grievance: grievance,
    recommended_mode: routing.mode,
    recommended_subMode: routing.subMode,
    confidence: routing.confidence,
    routing_reason: routing.reason,
    should_preserve_language: language !== 'en',
  };
}

/**
 * Detect user intent from query
 */
function detectIntent(query) {
  const complaintPatterns = [
    'problem',
    'issue',
    'broken',
    'not working',
    'poor',
    'bad',
    'complaint',
    'शिकायत',
    'समस्या',
    'సమస్య',
    'പ്രശ്നം',
    'ಸಮಸ್ಯೆ',
    'ගැටලුව',
    'সমস্যা',
  ];

  const questionPatterns = [
    'how',
    'what',
    'why',
    'when',
    'where',
    'कैसे',
    'क्या',
    'क्यों',
    'कब',
    'कहाँ',
    'గా',
    'ఎందుకు',
    'എങ്ങനെ',
    'ಹೇಗೆ',
    'නම්',
  ];

  const feedbackPatterns = [
    'suggest',
    'feedback',
    'improvement',
    'better',
    'सुझाव',
    'राय',
    'సూచన',
    'നിർദ്ദേശം',
    'ಸಲಹೆ',
  ];

  for (const pattern of complaintPatterns) {
    if (query.includes(pattern)) {
      return INTENT_TYPES.COMPLAINT;
    }
  }

  for (const pattern of questionPatterns) {
    if (query.includes(pattern)) {
      return INTENT_TYPES.QUESTION;
    }
  }

  for (const pattern of feedbackPatterns) {
    if (query.includes(pattern)) {
      return INTENT_TYPES.FEEDBACK;
    }
  }

  return INTENT_TYPES.GENERAL;
}

/**
 * Classify grievance category from query
 */
function classifyGrievance(query, language) {
  let maxMatches = 0;
  let bestCategory = 'GR-08'; // Default: Other

  for (const [code, category] of Object.entries(GRIEVANCE_CATEGORIES)) {
    let matches = 0;

    for (const keyword of category.keywords) {
      if (query.includes(keyword.toLowerCase())) {
        matches += 1;
      }
    }

    if (matches > maxMatches) {
      maxMatches = matches;
      bestCategory = code;
    }
  }

  return {
    category_code: bestCategory,
    category_name: GRIEVANCE_CATEGORIES[bestCategory].name,
    matches: maxMatches,
    confidence: Math.min(maxMatches * 0.3, 1),
  };
}

/**
 * Determine recommended mode and sub-mode based on analysis
 */
function determineMode(intent, grievanceCode, language, query) {
  // For multi-language queries, use standard mode
  if (language && language !== 'en') {
    return {
      mode: 'standard',
      subMode: null,
      confidence: 0.8,
      reason: `Native language (${language}) - routing to standard multilingual mode`,
    };
  }

  // Complaints → standard mode for direct routing
  if (intent === INTENT_TYPES.COMPLAINT) {
    return {
      mode: 'standard',
      subMode: null,
      confidence: 0.85,
      reason: 'Complaint detected - standard grievance processing mode',
    };
  }

  // Questions → could use debate/evidence
  if (intent === INTENT_TYPES.QUESTION) {
    // Check if question is about understanding/reasoning
    if (
      query.includes('why') ||
      query.includes('why') ||
      query.includes('कैसे')
    ) {
      return {
        mode: 'experimental',
        subMode: 'debate',
        confidence: 0.75,
        reason: 'Why/How question - enabling debate mode for reasoning',
      };
    }

    return {
      mode: 'standard',
      subMode: null,
      confidence: 0.7,
      reason: 'Question detected - standard processing mode',
    };
  }

  // Feedback → use evidence mode for systematic review
  if (intent === INTENT_TYPES.FEEDBACK) {
    return {
      mode: 'experimental',
      subMode: 'evidence',
      confidence: 0.8,
      reason: 'Feedback detected - enabling evidence mode for analysis',
    };
  }

  // Default
  return {
    mode: 'standard',
    subMode: null,
    confidence: 0.5,
    reason: 'Query type not specifically matched - using standard mode',
  };
}

/**
 * Get grievance category details
 */
export function getGrievanceCategoryInfo(categoryCode) {
  return GRIEVANCE_CATEGORIES[categoryCode] || GRIEVANCE_CATEGORIES['GR-08'];
}

/**
 * Format routing decision for display
 */
export function formatRoutingDecision(routingDecision) {
  return {
    userQuery: routingDecision.query,
    language: routingDecision.language,
    detectedIntent: routingDecision.detected_intent,
    grievanceCategory: routingDecision.detected_grievance.category_name,
    modelMode: routingDecision.recommended_mode,
    processingSubMode: routingDecision.recommended_subMode,
    routingConfidence: `${Math.round(
      routingDecision.confidence * 100
    )}%`,
    reason: routingDecision.routing_reason,
    preserveLanguage: routingDecision.should_preserve_language,
  };
}

const nlpRouterExports = {
  analyzeAndRoute,
  detectIntent,
  classifyGrievance,
  determineMode,
  getGrievanceCategoryInfo,
  formatRoutingDecision,
  GRIEVANCE_CATEGORIES,
  INTENT_TYPES,
};

export default nlpRouterExports;
