/**
 * ============================================================
 * Memory Manager — Cognitive State & Graph Memory Architecture
 * ============================================================
 *
 * Implements the Sentinel-E Cognitive Memory System:
 *
 * Layer 1: Short-Term Conversational Memory
 *   - Entity continuity, topic threading, pronoun resolution
 *   - Cognitive context model (active operation, objective, constraints)
 *   - Intent trajectory tracking
 *
 * Layer 2: Analytical Memory
 *   - Disagreement trends, boundary instability, evidence reliability
 *
 * Layer 3: User Preference & Feedback Memory (UserProfile)
 *   - Preferred verbosity, analytics visibility, behavioral patterns
 *   - Gradual adaptation from session-level feedback patterns
 *
 * Layer 4: Cognitive Graph Memory
 *   - Entities, operations, claims, constraints, corrections
 *   - Edges: depends_on, clarifies, overrides, contradicts, extends
 *   - Traversed before every response generation
 *
 * All layers persist to localStorage for cross-session continuity.
 * All layers are injected into every mode execution.
 *
 * ============================================================
 */

const STORAGE_KEY_PREFIX = 'sentinel_e_memory_';

// ─── Layer 1: Short-Term Conversational Memory ──────────────

/**
 * @typedef {Object} CognitiveState
 * @property {string|null} activeEntity       — last referenced entity
 * @property {string|null} activeOperation    — current operation (create, analyze, compare, etc.)
 * @property {string|null} activeObjective    — user's overarching goal
 * @property {Object} clarifiedConstraints    — constraints the user has explicitly set
 * @property {string[]} logicalDependencies   — what current reasoning depends on
 * @property {number} epistemicConfidence     — system's confidence in understanding user intent (0–1)
 * @property {string[]} intentTrajectory      — ordered history of classified intents
 */

/**
 * @typedef {Object} ShortTermMemory
 * @property {string} sessionId
 * @property {CognitiveState} cognitiveState  — active cognitive context model
 * @property {string|null} activeTopic        — current conversation topic
 * @property {string|null} lastIntent         — classified intent of last query
 * @property {string|null} lastMode           — last mode used
 * @property {string|null} lastSubMode        — last sub-mode used
 * @property {Array} recentMessages           — sliding window of recent messages (max 20)
 * @property {Object} entityMap               — entity → last mentioned context
 */
function createShortTermMemory(sessionId) {
  return {
    sessionId: sessionId || generateSessionId(),
    cognitiveState: {
      activeEntity: null,
      activeOperation: null,
      activeObjective: null,
      clarifiedConstraints: {},
      logicalDependencies: [],
      epistemicConfidence: 0.5,
      intentTrajectory: [],
    },
    activeTopic: null,
    lastIntent: null,
    lastMode: null,
    lastSubMode: null,
    recentMessages: [],
    entityMap: {},
  };
}

// ─── Layer 2: Analytical Memory ─────────────────────────────

/**
 * @typedef {Object} AnalyticalMemory
 * @property {number[]} disagreementTrend        — rolling window of disagreement scores
 * @property {number[]} boundaryTrend            — rolling window of boundary severity
 * @property {number[]} instabilityIndex         — rolling window of fragility indices
 * @property {number[]} evidenceReliabilityTrend — rolling window of evidence reliability
 * @property {number[]} confidenceTrend          — rolling window of confidence scores
 * @property {number} averageDisagreement        — computed average
 * @property {number} averageBoundary            — computed average
 * @property {number} averageInstability         — computed average
 */
function createAnalyticalMemory() {
  return {
    disagreementTrend: [],
    boundaryTrend: [],
    instabilityIndex: [],
    evidenceReliabilityTrend: [],
    confidenceTrend: [],
    averageDisagreement: 0,
    averageBoundary: 0,
    averageInstability: 0,
  };
}

// ─── Layer 3: User Profile & Feedback Memory ────────────────

/**
 * @typedef {Object} UserProfile
 * @property {"concise"|"balanced"|"detailed"} preferredVerbosity
 * @property {"minimal"|"standard"|"forensic"} analyticsVisibility
 * @property {boolean} citationBias              — prefers citations
 * @property {string|null} preferredMode
 * @property {string[]} negativePatterns          — patterns from thumbs-down
 * @property {string[]} positivePatterns          — patterns from thumbs-up
 * @property {number} totalFeedbackCount
 * @property {number} thumbsUpCount
 * @property {number} thumbsDownCount
 * @property {number} longResponseDownvotes       — consecutive long response negatives
 * @property {number} shortResponseDownvotes      — consecutive short response negatives
 * @property {number} analyticsUpvotes            — times user liked analytics-heavy output
 * @property {number} analyticsDownvotes          — times user disliked analytics-heavy output
 * @property {Object} modeUsage                   — { mode: count }
 */
function createUserProfile() {
  return {
    preferredVerbosity: 'balanced',
    analyticsVisibility: 'standard',
    citationBias: false,
    preferredMode: null,
    negativePatterns: [],
    positivePatterns: [],
    totalFeedbackCount: 0,
    thumbsUpCount: 0,
    thumbsDownCount: 0,
    longResponseDownvotes: 0,
    shortResponseDownvotes: 0,
    analyticsUpvotes: 0,
    analyticsDownvotes: 0,
    modeUsage: {},
  };
}

// ─── Persistence ────────────────────────────────────────────

function persist(key, data) {
  try {
    localStorage.setItem(STORAGE_KEY_PREFIX + key, JSON.stringify(data));
  } catch (e) {
    // localStorage full or unavailable — degrade silently
  }
}

function restore(key, defaultFactory) {
  try {
    const stored = localStorage.getItem(STORAGE_KEY_PREFIX + key);
    if (stored) {
      const parsed = JSON.parse(stored);
      // Merge with defaults to handle schema evolution
      const defaults = defaultFactory();
      return { ...defaults, ...parsed };
    }
  } catch (e) {
    // Corrupt data — reset
  }
  return defaultFactory();
}

function generateSessionId() {
  return 'session_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
}

// ─── Layer 4: Cognitive Graph Memory ────────────────────────

/**
 * Graph-structured memory where nodes are semantic units
 * and edges represent relationships between them.
 *
 * Node types: entity, operation, claim, constraint, correction, preference
 * Edge types: depends_on, clarifies, overrides, contradicts, extends
 *
 * The graph is traversed before output to:
 *   - Prevent losing clarifications
 *   - Prevent reverting to overridden interpretations
 *   - Maintain logical dependency chains
 */
function createCognitiveGraph() {
  return {
    nodes: [],   // { id, type, content, timestamp, active }
    edges: [],   // { from, to, relation, timestamp }
    nextId: 1,
  };
}

// ─── Memory Manager Class ───────────────────────────────────

class MemoryManager {
  constructor() {
    this.shortTerm = restore('short_term', () => createShortTermMemory());
    this.analytical = restore('analytical', createAnalyticalMemory);
    this.userProfile = restore('user_profile', createUserProfile);
    this.graph = restore('cognitive_graph', createCognitiveGraph);
    // Ensure cognitiveState exists (schema migration)
    if (!this.shortTerm.cognitiveState) {
      this.shortTerm.cognitiveState = {
        activeEntity: this.shortTerm.activeEntity || null,
        activeOperation: null,
        activeObjective: null,
        clarifiedConstraints: {},
        logicalDependencies: [],
        epistemicConfidence: 0.5,
        intentTrajectory: [],
      };
    }
  }

  // ─── Cognitive State Operations ───────────────────────────

  /**
   * Get the current cognitive state snapshot.
   */
  getCognitiveState() {
    return { ...this.shortTerm.cognitiveState };
  }

  /**
   * Update cognitive state from a new user message.
   * This is the core of Section II — Cognitive Context Model.
   *
   * When new input arrives:
   *   - Update active entity/operation/objective
   *   - If user clarifies intent → update constraint bindings
   *   - If user modifies task semantics → re-anchor objective
   *   - Track intent trajectory for pattern detection
   */
  _updateCognitiveState(text, extracted) {
    const cs = this.shortTerm.cognitiveState;

    // 1. Update entity (semantic continuity, not string matching)
    if (extracted.entity) {
      cs.activeEntity = extracted.entity;
    }

    // 2. Detect and track operation
    const operation = detectOperation(text);
    if (operation) {
      const previousOp = cs.activeOperation;
      cs.activeOperation = operation;

      // If operation changed, add dependency edge in graph
      if (previousOp && previousOp !== operation) {
        this._addGraphEdge(
          this._findOrCreateNode('operation', operation),
          this._findOrCreateNode('operation', previousOp),
          'extends'
        );
      }
    }

    // 3. Detect objective shifts
    const objective = detectObjective(text);
    if (objective) {
      if (cs.activeObjective && cs.activeObjective !== objective) {
        // User re-anchored objective — record override
        this._addGraphEdge(
          this._findOrCreateNode('claim', objective),
          this._findOrCreateNode('claim', cs.activeObjective),
          'overrides'
        );
      }
      cs.activeObjective = objective;
    }

    // 4. Detect constraint clarifications
    const constraints = detectConstraints(text);
    for (const [key, value] of Object.entries(constraints)) {
      if (cs.clarifiedConstraints[key] && cs.clarifiedConstraints[key] !== value) {
        // User overrode a previous constraint
        this._addGraphEdge(
          this._findOrCreateNode('constraint', `${key}=${value}`),
          this._findOrCreateNode('constraint', `${key}=${cs.clarifiedConstraints[key]}`),
          'overrides'
        );
      }
      cs.clarifiedConstraints[key] = value;
    }

    // 5. Detect user corrections (instruction binding — Section VIII)
    if (detectCorrection(text)) {
      const correctionNode = this._findOrCreateNode('correction', text.substring(0, 120));
      // Mark previous related claims as overridden
      const relatedClaims = this.graph.nodes.filter(n =>
        n.type === 'claim' && n.active &&
        textOverlap(n.content, text) > 0.3
      );
      for (const claim of relatedClaims) {
        this._addGraphEdge(correctionNode, claim.id, 'overrides');
        claim.active = false;
      }
      // Update epistemic confidence — correction means we were wrong
      cs.epistemicConfidence = Math.max(0.2, cs.epistemicConfidence - 0.15);
    }

    // 6. Track intent trajectory
    if (extracted.intent) {
      cs.intentTrajectory.push(extracted.intent);
      if (cs.intentTrajectory.length > 20) {
        cs.intentTrajectory = cs.intentTrajectory.slice(-20);
      }
    }

    // 7. Update logical dependencies
    cs.logicalDependencies = this._computeActiveDependencies();

    // 8. Adjust epistemic confidence
    if (extracted.intent === 'greeting' || extracted.intent === 'closing') {
      cs.epistemicConfidence = 0.9; // High confidence on simple intent
    } else if (cs.intentTrajectory.length > 2) {
      // If intent is consistent, confidence rises
      const last3 = cs.intentTrajectory.slice(-3);
      if (last3.every(i => i === last3[0])) {
        cs.epistemicConfidence = Math.min(1.0, cs.epistemicConfidence + 0.1);
      }
    }

    this.shortTerm.cognitiveState = cs;
  }

  // ─── Graph Memory Operations ──────────────────────────────

  /**
   * Find or create a node in the cognitive graph.
   */
  _findOrCreateNode(type, content) {
    const existing = this.graph.nodes.find(n =>
      n.type === type && n.content === content && n.active
    );
    if (existing) return existing.id;

    const id = this.graph.nextId++;
    this.graph.nodes.push({
      id, type, content,
      timestamp: Date.now(),
      active: true,
    });

    // Prune old nodes (keep graph manageable)
    if (this.graph.nodes.length > 200) {
      this.graph.nodes = this.graph.nodes.slice(-150);
    }

    return id;
  }

  /**
   * Add an edge between two nodes.
   */
  _addGraphEdge(fromId, toId, relation) {
    if (fromId === toId) return;
    // Avoid duplicate edges
    const exists = this.graph.edges.some(e =>
      e.from === fromId && e.to === toId && e.relation === relation
    );
    if (exists) return;

    this.graph.edges.push({
      from: fromId, to: toId, relation,
      timestamp: Date.now(),
    });

    // Prune old edges
    if (this.graph.edges.length > 500) {
      this.graph.edges = this.graph.edges.slice(-350);
    }

    this._persistGraph();
  }

  /**
   * Compute active logical dependencies by traversing the graph.
   * Returns a list of active constraints and claims that current
   * reasoning depends on.
   */
  _computeActiveDependencies() {
    const activeNodes = this.graph.nodes.filter(n => n.active);
    const overriddenIds = new Set(
      this.graph.edges
        .filter(e => e.relation === 'overrides')
        .map(e => e.to)
    );

    return activeNodes
      .filter(n => !overriddenIds.has(n.id))
      .filter(n => ['constraint', 'correction', 'claim'].includes(n.type))
      .slice(-10)
      .map(n => n.content);
  }

  /**
   * Get the effective active constraints (after overrides).
   * Traverses override edges to find the latest value for each key.
   */
  getActiveConstraints() {
    const cs = this.shortTerm.cognitiveState;
    // Start with explicit constraints
    const constraints = { ...cs.clarifiedConstraints };

    // Check graph for overrides
    const overrideEdges = this.graph.edges.filter(e => e.relation === 'overrides');
    const overriddenIds = new Set(overrideEdges.map(e => e.to));

    // Remove overridden constraint nodes
    for (const node of this.graph.nodes) {
      if (node.type === 'constraint' && overriddenIds.has(node.id)) {
        const [key] = node.content.split('=');
        if (key && constraints[key]) {
          // Already overridden by newer constraint — keep newer
        }
      }
    }

    return constraints;
  }

  /**
   * Check if a claim has been contradicted or overridden.
   */
  isClaimValid(claimContent) {
    const claimNode = this.graph.nodes.find(n =>
      n.type === 'claim' && n.content === claimContent
    );
    if (!claimNode) return true; // Unknown claim, assume valid

    const overridden = this.graph.edges.some(e =>
      e.to === claimNode.id && (e.relation === 'overrides' || e.relation === 'contradicts')
    );
    return !overridden;
  }

  // ─── Short-Term Memory Operations ─────────────────────────

  /**
   * Record a new message into short-term memory.
   * Extracts entities, updates cognitive state, builds graph.
   */
  recordMessage(message, mode, subMode) {
    // Add to sliding window (keep last 20)
    this.shortTerm.recentMessages.push({
      role: message.role,
      content: message.content,
      timestamp: message.timestamp || new Date().toISOString(),
      mode,
      subMode,
    });
    if (this.shortTerm.recentMessages.length > 20) {
      this.shortTerm.recentMessages = this.shortTerm.recentMessages.slice(-20);
    }

    // Track mode usage
    this.shortTerm.lastMode = mode;
    this.shortTerm.lastSubMode = subMode;

    // Update mode usage in user profile
    if (mode) {
      const key = subMode ? `${mode}:${subMode}` : mode;
      this.userProfile.modeUsage[key] = (this.userProfile.modeUsage[key] || 0) + 1;
    }

    // Extract entities and intent from user messages
    if (message.role === 'user') {
      const extracted = extractEntitiesAndIntent(message.content);

      // Update cognitive state model (Section II)
      this._updateCognitiveState(message.content, extracted);

      // Legacy entity/topic tracking (backward compat)
      if (extracted.entity) {
        this.shortTerm.entityMap[extracted.entity.toLowerCase()] = {
          lastMentioned: new Date().toISOString(),
          context: message.content.substring(0, 100),
        };
      }
      if (extracted.topic) {
        this.shortTerm.activeTopic = extracted.topic;
      }
      this.shortTerm.lastIntent = extracted.intent;

      // Add entity/claim nodes to cognitive graph
      if (extracted.entity) {
        this._findOrCreateNode('entity', extracted.entity);
      }
    }

    // Record assistant claims in graph for future contradiction checking
    if (message.role === 'assistant' && message.content) {
      const claims = extractClaims(message.content);
      for (const claim of claims) {
        this._findOrCreateNode('claim', claim);
      }
    }

    this._persistShortTerm();
    this._persistGraph();
  }

  /**
   * Resolve context for the next query.
   * Uses semantic continuity (not keyword matching) for pronoun resolution.
   * Evaluates whether the message depends on prior state.
   */
  resolveContext(userQuery) {
    const cs = this.shortTerm.cognitiveState;
    const context = {
      originalQuery: userQuery,
      resolvedQuery: userQuery,
      activeEntity: cs.activeEntity,
      activeTopic: this.shortTerm.activeTopic,
      activeOperation: cs.activeOperation,
      activeObjective: cs.activeObjective,
      epistemicConfidence: cs.epistemicConfidence,
      isFollowUp: false,
      previousMessages: this.shortTerm.recentMessages.slice(-6),
      lastIntent: this.shortTerm.lastIntent,
      activeDependencies: cs.logicalDependencies,
      activeConstraints: this.getActiveConstraints(),
    };

    // Detect follow-up by semantic continuity, not just keyword patterns
    const isShortQuery = userQuery.trim().split(/\s+/).length < 8;
    const hasPronounsOrDeictic = /\b(he|she|it|they|him|her|them|his|its|their|that|this|those|these)\b/i.test(userQuery);
    const hasFollowUpSignal = /^(what about|how about|and what|tell me more|elaborate|explain|why|can you|also|but|however|instead|rather)\b/i.test(userQuery.trim());
    const lacksExplicitSubject = isShortQuery && !extractEntitiesAndIntent(userQuery).entity;

    // Semantic follow-up detection: short query + no explicit entity + prior conversation exists
    context.isFollowUp = (
      (hasPronounsOrDeictic || hasFollowUpSignal || lacksExplicitSubject) &&
      this.shortTerm.recentMessages.length > 0
    );

    // Pronoun resolution via semantic continuity (not hardcoded rules)
    if (context.isFollowUp && cs.activeEntity) {
      const pronouns = /\b(he|she|it|they|him|her|them|his|its|their)\b/gi;
      if (pronouns.test(userQuery)) {
        context.resolvedQuery = userQuery.replace(pronouns, cs.activeEntity);
      } else if (lacksExplicitSubject && cs.activeEntity) {
        // No pronouns but no subject either — infer subject from context
        context.resolvedQuery = userQuery;
        // Don't rewrite, but signal that entity context should be injected
        context.implicitEntity = cs.activeEntity;
      }
    }

    return context;
  }

  // ─── Analytical Memory Operations ─────────────────────────

  /**
   * Record analytical metrics from a response.
   * Computes rolling averages and trend direction.
   */
  recordAnalytics(responseData) {
    if (!responseData) return;

    const MAX_WINDOW = 20;
    const push = (arr, val) => {
      if (val == null || isNaN(val)) return arr;
      arr.push(val);
      if (arr.length > MAX_WINDOW) arr.shift();
      return arr;
    };

    // Extract metrics from response
    const meta = responseData.omega_metadata || {};
    const boundary = responseData.boundary_result || meta.boundary_result || {};
    const session = meta.session_state || {};

    push(this.analytical.confidenceTrend, responseData.confidence);
    push(this.analytical.boundaryTrend, boundary.severity_score);
    push(this.analytical.instabilityIndex, meta.fragility_index);

    if (session.disagreement_score != null) {
      push(this.analytical.disagreementTrend, session.disagreement_score);
    }

    if (meta.evidence_result?.aggregate_reliability != null) {
      push(this.analytical.evidenceReliabilityTrend, meta.evidence_result.aggregate_reliability);
    }

    // Compute rolling averages
    const avg = (arr) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    this.analytical.averageDisagreement = avg(this.analytical.disagreementTrend);
    this.analytical.averageBoundary = avg(this.analytical.boundaryTrend);
    this.analytical.averageInstability = avg(this.analytical.instabilityIndex);

    this._persistAnalytical();
  }

  /**
   * Get analytical trend signals for adaptive behavior.
   */
  getAnalyticalSignals() {
    const trend = (arr) => {
      if (arr.length < 3) return 'stable';
      const recent = arr.slice(-3);
      const older = arr.slice(-6, -3);
      if (older.length === 0) return 'stable';
      const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
      const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;
      const delta = recentAvg - olderAvg;
      if (delta > 0.1) return 'increasing';
      if (delta < -0.1) return 'decreasing';
      return 'stable';
    };

    return {
      disagreementTrend: trend(this.analytical.disagreementTrend),
      boundaryTrend: trend(this.analytical.boundaryTrend),
      instabilityTrend: trend(this.analytical.instabilityIndex),
      evidenceReliabilityTrend: trend(this.analytical.evidenceReliabilityTrend),
      averageDisagreement: this.analytical.averageDisagreement,
      averageBoundary: this.analytical.averageBoundary,
      averageInstability: this.analytical.averageInstability,
      // Adaptive signals
      shouldReduceDebateDepth: this.analytical.averageDisagreement < 0.2 && this.analytical.disagreementTrend.length >= 5,
      shouldIncreaseExplanation: this.analytical.averageInstability > 0.6,
      shouldElevateCitationStrictness: this.analytical.evidenceReliabilityTrend.length >= 3 &&
        (this.analytical.evidenceReliabilityTrend.slice(-3).reduce((a, b) => a + b, 0) / 3) < 0.5,
    };
  }

  // ─── User Profile Operations ──────────────────────────────

  /**
   * Process feedback event and update UserProfile adaptively.
   * This is the behavioral shaping engine.
   */
  recordFeedback(vote, messageContext) {
    this.userProfile.totalFeedbackCount++;

    if (vote === 'up') {
      this.userProfile.thumbsUpCount++;
      this.userProfile.longResponseDownvotes = 0; // Reset streak
      this.userProfile.shortResponseDownvotes = 0;

      // Learn positive patterns
      if (messageContext) {
        if (messageContext.hadAnalytics) {
          this.userProfile.analyticsUpvotes++;
        }
        if (messageContext.responseLength > 500) {
          // User likes detailed responses
          this._nudgeVerbosity('detailed');
        }
        if (messageContext.hadCitations) {
          this.userProfile.citationBias = true;
        }
        if (messageContext.mode) {
          this.userProfile.positivePatterns.push(messageContext.mode);
          if (this.userProfile.positivePatterns.length > 50) {
            this.userProfile.positivePatterns = this.userProfile.positivePatterns.slice(-50);
          }
        }
      }
    } else if (vote === 'down') {
      this.userProfile.thumbsDownCount++;

      if (messageContext) {
        // Track response length dissatisfaction
        if (messageContext.responseLength > 800) {
          this.userProfile.longResponseDownvotes++;
          // If >3 consecutive long-response downvotes → reduce verbosity
          if (this.userProfile.longResponseDownvotes >= 3) {
            this._nudgeVerbosity('concise');
            this.userProfile.longResponseDownvotes = 0;
          }
        } else if (messageContext.responseLength < 100) {
          this.userProfile.shortResponseDownvotes++;
          if (this.userProfile.shortResponseDownvotes >= 3) {
            this._nudgeVerbosity('detailed');
            this.userProfile.shortResponseDownvotes = 0;
          }
        }

        // Track analytics dissatisfaction
        if (messageContext.hadAnalytics) {
          this.userProfile.analyticsDownvotes++;
          if (this.userProfile.analyticsDownvotes >= 3 && this.userProfile.analyticsUpvotes < 2) {
            this.userProfile.analyticsVisibility = 'minimal';
          }
        }

        if (messageContext.mode) {
          this.userProfile.negativePatterns.push(messageContext.mode);
          if (this.userProfile.negativePatterns.length > 50) {
            this.userProfile.negativePatterns = this.userProfile.negativePatterns.slice(-50);
          }
        }
      }
    }

    // Determine preferred mode from usage
    this._updatePreferredMode();

    this._persistUserProfile();
  }

  /**
   * Get the effective user preferences for response shaping.
   */
  getUserPreferences() {
    return {
      verbosity: this.userProfile.preferredVerbosity,
      analyticsVisibility: this.userProfile.analyticsVisibility,
      citationBias: this.userProfile.citationBias,
      preferredMode: this.userProfile.preferredMode,
      satisfactionRatio: this.userProfile.totalFeedbackCount > 0
        ? this.userProfile.thumbsUpCount / this.userProfile.totalFeedbackCount
        : 1.0,
    };
  }

  // ─── Context Bundle (for mode execution) ──────────────────

  /**
   * Build the complete context bundle for mode execution.
   * This is injected into EVERY mode before execution.
   *
   * The bundle includes:
   *   - Cognitive state (entity, operation, objective, constraints, dependencies)
   *   - Short-term context (recent messages, follow-up resolution)
   *   - Analytical signals (trends, adaptive flags)
   *   - User preferences (verbosity, analytics, citation bias)
   *   - Graph-derived dependencies (active constraints after overrides)
   */
  buildContextBundle(userQuery, mode, subMode) {
    const resolved = this.resolveContext(userQuery);
    const signals = this.getAnalyticalSignals();
    const prefs = this.getUserPreferences();
    const cs = this.shortTerm.cognitiveState;

    return {
      // Cognitive state (Section II)
      cognitiveState: {
        activeEntity: cs.activeEntity,
        activeOperation: cs.activeOperation,
        activeObjective: cs.activeObjective,
        epistemicConfidence: cs.epistemicConfidence,
        intentTrajectory: cs.intentTrajectory.slice(-5),
        activeDependencies: cs.logicalDependencies,
        activeConstraints: this.getActiveConstraints(),
      },
      // Short-term context
      shortTerm: {
        sessionId: this.shortTerm.sessionId,
        activeEntity: cs.activeEntity,
        activeTopic: this.shortTerm.activeTopic,
        lastIntent: this.shortTerm.lastIntent,
        lastMode: this.shortTerm.lastMode,
        recentMessages: this.shortTerm.recentMessages.slice(-6),
        isFollowUp: resolved.isFollowUp,
        resolvedQuery: resolved.resolvedQuery,
        implicitEntity: resolved.implicitEntity || null,
      },
      // Analytical signals
      analytical: signals,
      // User preferences
      userProfile: prefs,
      // Mode context
      currentMode: mode,
      currentSubMode: subMode,
    };
  }

  // ─── Session Management ───────────────────────────────────

  /**
   * Start a new session (clears short-term + graph, preserves profile + analytics)
   */
  newSession() {
    this.shortTerm = createShortTermMemory();
    this.graph = createCognitiveGraph();
    this._persistShortTerm();
    this._persistGraph();
  }

  /**
   * Full reset (clears everything)
   */
  reset() {
    this.shortTerm = createShortTermMemory();
    this.analytical = createAnalyticalMemory();
    this.userProfile = createUserProfile();
    this.graph = createCognitiveGraph();
    this._persistShortTerm();
    this._persistAnalytical();
    this._persistUserProfile();
    this._persistGraph();
  }

  // ─── Private Helpers ──────────────────────────────────────

  _nudgeVerbosity(direction) {
    const levels = ['concise', 'balanced', 'detailed'];
    const current = levels.indexOf(this.userProfile.preferredVerbosity);
    const target = levels.indexOf(direction);
    // Move one step toward target
    if (target > current && current < 2) {
      this.userProfile.preferredVerbosity = levels[current + 1];
    } else if (target < current && current > 0) {
      this.userProfile.preferredVerbosity = levels[current - 1];
    }
  }

  _updatePreferredMode() {
    const usage = this.userProfile.modeUsage;
    if (Object.keys(usage).length === 0) return;
    const sorted = Object.entries(usage).sort((a, b) => b[1] - a[1]);
    if (sorted[0][1] >= 5) {
      this.userProfile.preferredMode = sorted[0][0];
    }
  }

  _persistShortTerm() { persist('short_term', this.shortTerm); }
  _persistAnalytical() { persist('analytical', this.analytical); }
  _persistUserProfile() { persist('user_profile', this.userProfile); }
  _persistGraph() { persist('cognitive_graph', this.graph); }
}

// ─── Entity + Intent Extraction (Lightweight NER) ───────────

function extractEntitiesAndIntent(text) {
  const result = { entity: null, topic: null, intent: null };
  if (!text) return result;

  // Intent classification
  const intentMap = [
    { pattern: /^(write|create|make|generate|code|implement|build)\b/i, intent: 'creation' },
    { pattern: /^(explain|what\s+is|define|describe)\b/i, intent: 'explanation' },
    { pattern: /^(compare|versus|vs|difference|between)\b/i, intent: 'comparison' },
    { pattern: /^(analyze|evaluate|assess|review|critique)\b/i, intent: 'analysis' },
    { pattern: /^(debate|argue|discuss|pros?\s+and\s+cons?)\b/i, intent: 'debate' },
    { pattern: /^(find|search|look\s+up|cite|evidence|source)\b/i, intent: 'research' },
    { pattern: /^(fix|debug|error|bug|issue|problem|solve)\b/i, intent: 'debugging' },
    { pattern: /^(hello|hi|hey|good\s+(morning|afternoon|evening))\b/i, intent: 'greeting' },
    { pattern: /^(thanks?|thank\s+you|bye|goodbye)\b/i, intent: 'closing' },
    { pattern: /^(summarize|summary|tldr|brief)\b/i, intent: 'summarization' },
    { pattern: /^(translate|convert)\b/i, intent: 'translation' },
  ];

  for (const { pattern, intent } of intentMap) {
    if (pattern.test(text.trim())) {
      result.intent = intent;
      break;
    }
  }
  if (!result.intent) result.intent = 'general';

  // Entity extraction — proper nouns (capitalized words not at sentence start)
  const words = text.split(/\s+/);
  const properNouns = [];
  for (let i = 1; i < words.length; i++) {
    const w = words[i].replace(/[^a-zA-Z]/g, '');
    if (w.length > 1 && w[0] === w[0].toUpperCase() && w[0] !== w[0].toLowerCase()) {
      properNouns.push(w);
    }
  }
  // Group consecutive proper nouns
  if (properNouns.length > 0) {
    result.entity = properNouns.slice(0, 3).join(' ');
  }

  // Topic extraction — key phrases
  const topicPatterns = [
    /about\s+(.{3,40})(?:\.|$)/i,
    /regarding\s+(.{3,40})(?:\.|$)/i,
    /on\s+the\s+topic\s+of\s+(.{3,40})(?:\.|$)/i,
  ];
  for (const tp of topicPatterns) {
    const match = text.match(tp);
    if (match) {
      result.topic = match[1].trim();
      break;
    }
  }

  return result;
}

// ─── Cognitive Extraction Utilities ─────────────────────────

/**
 * Detect the operation the user is requesting.
 * Maps to cognitive context "activeOperation".
 */
function detectOperation(text) {
  if (!text) return null;
  const opMap = [
    { pattern: /\b(create|write|generate|make|build|implement|code)\b/i, op: 'create' },
    { pattern: /\b(analyze|evaluate|assess|review|audit|examine)\b/i, op: 'analyze' },
    { pattern: /\b(compare|contrast|versus|difference|similarities)\b/i, op: 'compare' },
    { pattern: /\b(explain|describe|define|clarify|what\s+is)\b/i, op: 'explain' },
    { pattern: /\b(fix|debug|repair|solve|resolve|troubleshoot)\b/i, op: 'fix' },
    { pattern: /\b(summarize|summarise|recap|condense|tldr|brief)\b/i, op: 'summarize' },
    { pattern: /\b(translate|convert|transform|migrate|port)\b/i, op: 'transform' },
    { pattern: /\b(find|search|look\s+up|research|investigate)\b/i, op: 'research' },
    { pattern: /\b(optimize|improve|refactor|enhance|upgrade)\b/i, op: 'optimize' },
    { pattern: /\b(debate|argue|discuss|pros?\s+and\s+cons?)\b/i, op: 'debate' },
    { pattern: /\b(plan|design|architect|structure|outline|strategy)\b/i, op: 'plan' },
    { pattern: /\b(test|verify|validate|check|confirm)\b/i, op: 'verify' },
  ];
  for (const { pattern, op } of opMap) {
    if (pattern.test(text)) return op;
  }
  return null;
}

/**
 * Detect the user's overarching objective.
 * Looks for goal-oriented phrases.
 */
function detectObjective(text) {
  if (!text) return null;
  const objPatterns = [
    /(?:i\s+(?:want|need|am\s+trying)\s+to)\s+(.{5,60}?)(?:\.|$)/i,
    /(?:my\s+goal\s+is\s+to)\s+(.{5,60}?)(?:\.|$)/i,
    /(?:help\s+me)\s+(.{5,60}?)(?:\.|$)/i,
    /(?:i'm\s+(?:working\s+on|building|creating))\s+(.{5,60}?)(?:\.|$)/i,
  ];
  for (const p of objPatterns) {
    const match = text.match(p);
    if (match) return match[1].trim();
  }
  return null;
}

/**
 * Detect explicit constraints set by the user.
 * Returns key-value pairs.
 */
function detectConstraints(text) {
  if (!text) return {};
  const constraints = {};
  const patterns = [
    { pattern: /\b(?:use|using|in)\s+(python|javascript|typescript|java|go|rust|c\+\+|ruby|swift|kotlin)\b/i, key: 'language' },
    { pattern: /\b(?:keep\s+it|make\s+it|should\s+be)\s+(short|brief|concise|detailed|thorough|simple)\b/i, key: 'style' },
    { pattern: /\b(?:format\s+(?:it\s+)?as|output\s+(?:in|as))\s+(json|csv|table|list|markdown|yaml|xml)\b/i, key: 'format' },
    { pattern: /\b(?:no\s+more\s+than|max(?:imum)?|limit\s+to)\s+(\d+)\s*(words?|lines?|paragraphs?|chars?|characters?)/i, key: 'length_limit' },
    { pattern: /\b(?:don'?t|do\s+not|avoid|skip|without)\s+(.{3,30}?)(?:\.|,|$)/i, key: 'exclusion' },
  ];
  for (const { pattern, key } of patterns) {
    const match = text.match(pattern);
    if (match) {
      constraints[key] = match[1].trim().toLowerCase();
    }
  }
  return constraints;
}

/**
 * Detect if the user is correcting a previous response.
 * This triggers instruction rebinding (Section VIII).
 */
function detectCorrection(text) {
  if (!text) return false;
  const correctionPatterns = [
    /^(no|wrong|incorrect|that'?s?\s+not|actually|i\s+meant|not\s+what\s+i)/i,
    /\b(i\s+said|i\s+asked\s+for|what\s+i\s+meant|let\s+me\s+clarify|to\s+clarify)\b/i,
    /\b(instead|rather\s+than|don'?t\s+do\s+that|not\s+like\s+that)\b/i,
    /\b(when\s+i\s+say|by\s+['"]?\w+['"]?\s+i\s+mean)\b/i,
  ];
  return correctionPatterns.some(p => p.test(text.trim()));
}

/**
 * Extract notable claims from assistant output for graph tracking.
 * Keeps assertions that can be validated or contradicted later.
 */
function extractClaims(text) {
  if (!text || text.length < 50) return [];
  const claims = [];
  // Extract sentences that make definitive assertions
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 20);
  for (const s of sentences.slice(0, 5)) {
    const t = s.trim();
    // Assertive patterns
    if (/\b(is|are|was|were|has|have|will|can|must|should)\b/i.test(t) && t.length < 120) {
      claims.push(t);
    }
  }
  return claims.slice(0, 3);
}

/**
 * Compute text overlap ratio between two strings (Jaccard-like).
 * Used for graph traversal — finding related claims.
 */
function textOverlap(a, b) {
  if (!a || !b) return 0;
  const wordsA = new Set(a.toLowerCase().split(/\s+/).filter(w => w.length > 2));
  const wordsB = new Set(b.toLowerCase().split(/\s+/).filter(w => w.length > 2));
  if (wordsA.size === 0 || wordsB.size === 0) return 0;
  let intersection = 0;
  for (const w of wordsA) {
    if (wordsB.has(w)) intersection++;
  }
  return intersection / Math.max(wordsA.size, wordsB.size);
}

// ─── Singleton Export ───────────────────────────────────────

const memoryManager = new MemoryManager();
export default memoryManager;

export {
  MemoryManager,
  createShortTermMemory,
  createAnalyticalMemory,
  createUserProfile,
};
