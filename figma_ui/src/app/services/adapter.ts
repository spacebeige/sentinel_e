// ============================================================
// Response Adapter — transforms backend → safe UI models
// Ensures the UI never crashes on missing/unexpected data
// ============================================================

import type {
  SentinelRunResponse,
  OmegaMetadata,
  OmegaBoundaryResult,
  OmegaReasoningTrace,
  ConfidenceEvolution,
  BehavioralRiskProfile,
  EvidenceResult,
  EvidenceSource,
  StressResult,
  DebateResult,
  OmegaSessionState,
  ChatHistoryItem,
  ChatMessage,
  HealthStatus,
  KernelStatus,
  SessionDescriptive,
  CrossAnalysisResult,
  CrossAnalysisStep,
  CrossAnalysisModelProfile,
  LearningSummary,
  SessionStats,
} from "../api";

// ============================================================
// SAFE DEFAULTS
// ============================================================

const DEFAULT_BOUNDARY: OmegaBoundaryResult = {
  risk_level: "LOW",
  severity_score: 0,
  explanation: "",
  risk_dimensions: {},
  human_review_required: false,
};

const DEFAULT_REASONING: OmegaReasoningTrace = {
  passes_executed: 0,
  initial_confidence: 0,
  final_confidence: 0,
  assumptions_extracted: 0,
  logical_gaps_detected: 0,
  boundary_severity: 0,
  self_critique_applied: false,
  refinement_applied: false,
};

const DEFAULT_EVOLUTION: ConfidenceEvolution = {
  initial: 0.5,
  final: 0.5,
};

const DEFAULT_BEHAVIORAL: BehavioralRiskProfile = {
  self_preservation_score: 0,
  manipulation_probability: 0,
  evasion_index: 0,
  confidence_inflation: 0,
  overall_risk: 0,
  risk_level: "LOW",
  signals_detected: 0,
  signal_breakdown: {},
  explanation: "",
};

const DEFAULT_EVIDENCE: EvidenceResult = {
  query: "",
  sources: [],
  source_count: 0,
  contradictions: [],
  contradiction_count: 0,
  evidence_confidence: 0,
  source_agreement: 0,
  lineage: [],
  search_executed: false,
};

const DEFAULT_STRESS: StressResult = {
  stability_after_stress: 1,
  contradictions_found: 0,
  revised_confidence: 0.5,
  overall_stability: 1,
  vector_results: {},
  breakdown_points: [],
};

// ============================================================
// ADAPTERS
// ============================================================

/** Clamp a number between 0 and 1 (for scores / percentages). */
function clamp01(v: unknown): number {
  const n = Number(v);
  if (isNaN(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

/** Safely parse a number with fallback. */
function safeNum(v: unknown, fallback = 0): number {
  const n = Number(v);
  return isNaN(n) ? fallback : n;
}

/** Sanitize text to prevent XSS when rendering via innerHTML (if ever needed). */
export function sanitizeText(text: unknown): string {
  if (typeof text !== "string") return "";
  return text
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Sanitize plain text for rendering in React (safe as-is via JSX). */
export function safeString(v: unknown, fallback = ""): string {
  return typeof v === "string" ? v : fallback;
}

// --- Run Response ---

export function adaptRunResponse(raw: Record<string, unknown>): SentinelRunResponse {
  const meta = (raw.omega_metadata || {}) as Partial<OmegaMetadata>;
  return {
    chat_id: safeString(raw.chat_id),
    chat_name: safeString(raw.chat_name),
    mode: safeString(raw.mode, "standard"),
    sub_mode: safeString(raw.sub_mode),
    original_mode: safeString(raw.original_mode),
    formatted_output: safeString(raw.formatted_output),
    data: {
      priority_answer: safeString(
        (raw.data as Record<string, unknown>)?.priority_answer ?? raw.formatted_output
      ),
    },
    confidence: clamp01(raw.confidence ?? 0.5),
    session_state: adaptSessionState(raw.session_state),
    reasoning_trace: adaptReasoningTrace(raw.reasoning_trace),
    boundary_result: adaptBoundary(raw.boundary_result),
    omega_metadata: adaptMetadata(meta),
  };
}

export function adaptMetadata(raw: Partial<OmegaMetadata> | null | undefined): OmegaMetadata {
  if (!raw) {
    return {
      omega_version: "4.5.0",
      mode: "standard",
      confidence: 0.5,
    };
  }
  return {
    omega_version: safeString(raw.omega_version, "4.5.0"),
    mode: safeString(raw.mode, "standard"),
    sub_mode: safeString(raw.sub_mode),
    original_mode: safeString(raw.original_mode),
    confidence: clamp01(raw.confidence ?? 0.5),
    session_state: raw.session_state ? adaptSessionState(raw.session_state) : undefined,
    reasoning_trace: raw.reasoning_trace ? adaptReasoningTrace(raw.reasoning_trace) : undefined,
    boundary_result: raw.boundary_result ? adaptBoundary(raw.boundary_result) : undefined,
    confidence_evolution: raw.confidence_evolution ? adaptEvolution(raw.confidence_evolution) : undefined,
    fragility_index: raw.fragility_index != null ? clamp01(raw.fragility_index) : undefined,
    behavioral_risk: raw.behavioral_risk ? adaptBehavioral(raw.behavioral_risk) : undefined,
    evidence_result: raw.evidence_result ? adaptEvidence(raw.evidence_result) : undefined,
    stress_result: raw.stress_result ? adaptStress(raw.stress_result) : undefined,
    confidence_components: raw.confidence_components as Record<string, unknown> | undefined,
    debate_result: raw.debate_result ? adaptDebate(raw.debate_result) : undefined,
    kill_active: raw.kill_active ?? undefined,
  };
}

export function adaptBoundary(raw: unknown): OmegaBoundaryResult {
  if (!raw || typeof raw !== "object") return { ...DEFAULT_BOUNDARY };
  const r = raw as Record<string, unknown>;
  return {
    risk_level: safeString(r.risk_level, "LOW"),
    severity_score: safeNum(r.severity_score),
    explanation: safeString(r.explanation),
    risk_dimensions: (r.risk_dimensions as Record<string, number>) ?? {},
    human_review_required: Boolean(r.human_review_required),
  };
}

export function adaptReasoningTrace(raw: unknown): OmegaReasoningTrace {
  if (!raw || typeof raw !== "object") return { ...DEFAULT_REASONING };
  const r = raw as Record<string, unknown>;
  return {
    passes_executed: safeNum(r.passes_executed),
    initial_confidence: clamp01(r.initial_confidence),
    final_confidence: clamp01(r.final_confidence),
    assumptions_extracted: safeNum(r.assumptions_extracted),
    logical_gaps_detected: safeNum(r.logical_gaps_detected),
    boundary_severity: safeNum(r.boundary_severity),
    self_critique_applied: Boolean(r.self_critique_applied),
    refinement_applied: Boolean(r.refinement_applied),
  };
}

export function adaptEvolution(raw: unknown): ConfidenceEvolution {
  if (!raw || typeof raw !== "object") return { ...DEFAULT_EVOLUTION };
  const r = raw as Record<string, unknown>;
  return {
    initial: clamp01(r.initial ?? 0.5),
    post_debate: r.post_debate != null ? clamp01(r.post_debate) : undefined,
    post_boundary: r.post_boundary != null ? clamp01(r.post_boundary) : undefined,
    post_evidence: r.post_evidence != null ? clamp01(r.post_evidence) : undefined,
    post_stress: r.post_stress != null ? clamp01(r.post_stress) : undefined,
    final: clamp01(r.final ?? 0.5),
  };
}

export function adaptBehavioral(raw: unknown): BehavioralRiskProfile {
  if (!raw || typeof raw !== "object") return { ...DEFAULT_BEHAVIORAL };
  const r = raw as Record<string, unknown>;
  return {
    self_preservation_score: clamp01(r.self_preservation_score),
    manipulation_probability: clamp01(r.manipulation_probability),
    evasion_index: clamp01(r.evasion_index),
    confidence_inflation: clamp01(r.confidence_inflation),
    overall_risk: clamp01(r.overall_risk),
    risk_level: safeString(r.risk_level, "LOW"),
    signals_detected: safeNum(r.signals_detected),
    signal_breakdown: (r.signal_breakdown as Record<string, number>) ?? {},
    explanation: safeString(r.explanation),
  };
}

export function adaptEvidence(raw: unknown): EvidenceResult {
  if (!raw || typeof raw !== "object") return { ...DEFAULT_EVIDENCE };
  const r = raw as Record<string, unknown>;
  const sources = Array.isArray(r.sources) ? r.sources.map(adaptEvidenceSource) : [];
  return {
    query: safeString(r.query),
    sources,
    source_count: safeNum(r.source_count, sources.length),
    contradictions: Array.isArray(r.contradictions) ? r.contradictions : [],
    contradiction_count: safeNum(r.contradiction_count),
    evidence_confidence: clamp01(r.evidence_confidence),
    source_agreement: clamp01(r.source_agreement),
    lineage: Array.isArray(r.lineage) ? r.lineage : [],
    search_executed: Boolean(r.search_executed),
  };
}

function adaptEvidenceSource(raw: unknown): EvidenceSource {
  if (!raw || typeof raw !== "object") {
    return { url: "", title: "", content_snippet: "", reliability_score: 0, domain: "" };
  }
  const r = raw as Record<string, unknown>;
  return {
    url: safeString(r.url),
    title: safeString(r.title),
    content_snippet: safeString(r.content_snippet),
    reliability_score: clamp01(r.reliability_score),
    domain: safeString(r.domain),
  };
}

export function adaptStress(raw: unknown): StressResult {
  if (!raw || typeof raw !== "object") return { ...DEFAULT_STRESS };
  const r = raw as Record<string, unknown>;
  return {
    stability_after_stress: clamp01(r.stability_after_stress ?? 1),
    contradictions_found: safeNum(r.contradictions_found),
    revised_confidence: clamp01(r.revised_confidence ?? 0.5),
    overall_stability: clamp01(r.overall_stability ?? 1),
    vector_results: (r.vector_results as Record<string, unknown>) ?? {},
    breakdown_points: Array.isArray(r.breakdown_points) ? r.breakdown_points.map(String) : [],
  };
}

export function adaptDebate(raw: unknown): DebateResult {
  if (!raw || typeof raw !== "object") return {};
  const r = raw as Record<string, unknown>;
  return {
    positions: Array.isArray(r.positions)
      ? r.positions.map((p: Record<string, unknown>) => ({
          model: safeString(p.model),
          position: safeString(p.position),
          confidence: clamp01(p.confidence),
          key_points: Array.isArray(p.key_points) ? p.key_points.map(String) : [],
        }))
      : undefined,
    rounds: safeNum(r.rounds) || undefined,
    consensus: safeString(r.consensus) || undefined,
  };
}

export function adaptSessionState(raw: unknown): OmegaSessionState | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const r = raw as Record<string, unknown>;
  return {
    session_id: safeString(r.session_id),
    chat_name: safeString(r.chat_name),
    primary_goal: safeString(r.primary_goal),
    inferred_domain: safeString(r.inferred_domain, "general"),
    user_expertise_score: clamp01(r.user_expertise_score),
    message_count: safeNum(r.message_count),
    error_patterns: Array.isArray(r.error_patterns) ? r.error_patterns : [],
    boundary_history_count: safeNum(r.boundary_history_count),
    latest_boundary_severity: safeNum(r.latest_boundary_severity),
    boundary_trend: safeString(r.boundary_trend, "stable"),
    disagreement_score: clamp01(r.disagreement_score),
    fragility_index: clamp01(r.fragility_index),
    session_confidence: clamp01(r.session_confidence ?? 0.5),
    reasoning_depth: safeString(r.reasoning_depth, "standard"),
  };
}

// --- Chat History ---

export function adaptChatHistoryItem(raw: Record<string, unknown>): ChatHistoryItem {
  return {
    id: safeString(raw.id),
    name: safeString(raw.name || raw.chat_name, "Untitled Chat"),
    mode: safeString(raw.mode, "standard"),
    created_at: safeString(raw.created_at),
    updated_at: safeString(raw.updated_at || raw.created_at),
    priority_answer: safeString(raw.priority_answer),
    machine_metadata: raw.machine_metadata ? adaptMetadata(raw.machine_metadata as Partial<OmegaMetadata>) : undefined,
    rounds: safeNum(raw.rounds),
  };
}

export function adaptChatMessage(raw: Record<string, unknown>): ChatMessage {
  return {
    role: (raw.role === "user" ? "user" : "assistant") as "user" | "assistant",
    content: safeString(raw.content),
    timestamp: typeof raw.timestamp === "string" ? raw.timestamp : null,
  };
}

// --- Health / Status ---

export function adaptHealth(raw: Record<string, unknown>): HealthStatus {
  return {
    status: raw.status === "degraded" ? "degraded" : "healthy",
    version: safeString(raw.version, "unknown"),
    omega_kernel: Boolean(raw.omega_kernel),
    knowledge_learner: Boolean(raw.knowledge_learner),
    orchestrator: Boolean(raw.orchestrator),
    redis: safeString(raw.redis),
    database: safeString(raw.database),
  };
}

export function adaptKernel(raw: Record<string, unknown>): KernelStatus {
  return {
    status: raw.status === "online" ? "online" : "offline",
    version: safeString(raw.version, "unknown"),
    active_sessions: safeNum(raw.active_sessions),
    session_ids: Array.isArray(raw.session_ids) ? raw.session_ids.map(String) : [],
    sub_modes: Array.isArray(raw.sub_modes) ? raw.sub_modes.map(String) : ["debate", "glass", "evidence"],
    behavioral_analyzer: Boolean(raw.behavioral_analyzer),
    evidence_engine: Boolean(raw.evidence_engine),
    message: safeString(raw.message),
  };
}

export function adaptSessionDescriptive(raw: Record<string, unknown>): SessionDescriptive | null {
  if (raw.error) return null;
  return {
    chat_name: safeString(raw.chat_name),
    goal: safeString(raw.goal),
    domain: safeString(raw.domain),
    domain_key: safeString(raw.domain_key),
    expertise: {
      label: safeString((raw.expertise as Record<string, unknown>)?.label, "Unknown"),
      score: clamp01((raw.expertise as Record<string, unknown>)?.score),
      description: safeString((raw.expertise as Record<string, unknown>)?.description),
    },
    confidence: {
      label: safeString((raw.confidence as Record<string, unknown>)?.label, "Unknown"),
      score: clamp01((raw.confidence as Record<string, unknown>)?.score ?? 0.5),
    },
    fragility: {
      label: safeString((raw.fragility as Record<string, unknown>)?.label, "Stable"),
      score: clamp01((raw.fragility as Record<string, unknown>)?.score),
    },
    disagreement: {
      score: clamp01((raw.disagreement as Record<string, unknown>)?.score),
      label: safeString((raw.disagreement as Record<string, unknown>)?.label, "None"),
    },
    message_count: safeNum(raw.message_count),
    reasoning_depth: safeString(raw.reasoning_depth, "standard"),
    error_count: safeNum(raw.error_count),
    boundary_count: safeNum(raw.boundary_count),
    last_boundary_severity: safeNum(raw.last_boundary_severity),
  };
}

export function adaptSessionStats(raw: Record<string, unknown>): SessionStats {
  return {
    total_sessions: safeNum(raw.total_sessions),
    mode_distribution: (raw.mode_distribution as Record<string, number>) ?? {},
    active_omega_sessions: safeNum(raw.active_omega_sessions),
  };
}

export function adaptLearningSummary(raw: Record<string, unknown>): LearningSummary {
  return {
    status: safeString(raw.status, "disabled"),
    summary: (raw.summary as Record<string, unknown>) ?? undefined,
    threshold_suggestions: (raw.threshold_suggestions as Record<string, unknown>) ?? undefined,
    risk_profiles: (raw.risk_profiles as Record<string, unknown>) ?? undefined,
    claim_type_risks: (raw.claim_type_risks as Record<string, unknown>) ?? undefined,
    message: safeString(raw.message),
  };
}

// --- Cross Analysis ---

export function adaptCrossAnalysis(raw: Record<string, unknown>): CrossAnalysisResult {
  const overall = (raw.overall_risk || {}) as Record<string, unknown>;
  return {
    pipeline_version: safeString(raw.pipeline_version, "1.0"),
    timestamp: safeString(raw.timestamp, new Date().toISOString()),
    elapsed_seconds: safeNum(raw.elapsed_seconds),
    steps_completed: safeNum(raw.steps_completed),
    steps_total: safeNum(raw.steps_total),
    steps: Array.isArray(raw.steps) ? raw.steps.map(adaptCrossStep) : [],
    model_profiles: Object.fromEntries(
      Object.entries((raw.model_profiles || {}) as Record<string, unknown>).map(([k, v]) => [
        k,
        adaptCrossProfile(v as Record<string, unknown>),
      ])
    ),
    analyzed_models: Array.isArray(raw.analyzed_models)
      ? raw.analyzed_models.map((m: Record<string, unknown>) => ({
          id: safeString(m.id),
          name: safeString(m.name),
          color: safeString(m.color, "#888"),
          analyzed_in_steps: Array.isArray(m.analyzed_in_steps) ? m.analyzed_in_steps.map(Number) : [],
        }))
      : [],
    overall_risk: {
      level: safeString(overall.level, "UNKNOWN"),
      average_threat: clamp01(overall.average_threat),
      average_manipulation: clamp01(overall.average_manipulation),
      average_risk: clamp01(overall.average_risk),
      max_threat: clamp01(overall.max_threat),
      models_analyzed: safeNum(overall.models_analyzed),
    },
  };
}

function adaptCrossStep(raw: unknown): CrossAnalysisStep {
  if (!raw || typeof raw !== "object") {
    return {
      step: 0,
      type: "individual",
      subject: "",
      subject_id: "",
      description: "",
      status: "error",
      scores: {
        manipulation_level: 0,
        risk_level: 0,
        self_preservation: 0,
        evasion_index: 0,
        confidence_inflation: 0,
        threat_level: 0,
        overall_risk: "UNKNOWN",
        key_signals: [],
      },
    };
  }
  const r = raw as Record<string, unknown>;
  const s = (r.scores || {}) as Record<string, unknown>;
  return {
    step: safeNum(r.step),
    type: r.type === "consensus" ? "consensus" : "individual",
    analyzer: safeString(r.analyzer) || undefined,
    analyzers: Array.isArray(r.analyzers) ? r.analyzers.map(String) : undefined,
    subject: safeString(r.subject),
    subject_id: safeString(r.subject_id),
    description: safeString(r.description),
    status: r.status === "success" ? "success" : "error",
    error: safeString(r.error) || undefined,
    scores: {
      manipulation_level: clamp01(s.manipulation_level),
      risk_level: clamp01(s.risk_level),
      self_preservation: clamp01(s.self_preservation),
      evasion_index: clamp01(s.evasion_index),
      confidence_inflation: clamp01(s.confidence_inflation),
      threat_level: clamp01(s.threat_level),
      overall_risk: safeString(s.overall_risk, "UNKNOWN"),
      reasoning: safeString(s.reasoning) || undefined,
      consensus_reasoning: safeString(s.consensus_reasoning) || undefined,
      agreement_score: s.agreement_score != null ? clamp01(s.agreement_score) : undefined,
      key_signals: Array.isArray(s.key_signals) ? s.key_signals.map(String) : [],
    },
  };
}

function adaptCrossProfile(raw: Record<string, unknown>): CrossAnalysisModelProfile {
  return {
    name: safeString(raw.name),
    color: safeString(raw.color, "#888"),
    status: raw.status === "analyzed" ? "analyzed" : "no_data",
    scores: (raw.scores as Record<string, number>) ?? {},
    overall_risk: safeString(raw.overall_risk) || undefined,
    step_count: safeNum(raw.step_count),
    key_signals: Array.isArray(raw.key_signals) ? raw.key_signals.map(String) : undefined,
    individual_steps: Array.isArray(raw.individual_steps) ? raw.individual_steps.map(adaptCrossStep) : undefined,
    consensus_steps: Array.isArray(raw.consensus_steps) ? raw.consensus_steps.map(adaptCrossStep) : undefined,
  };
}
