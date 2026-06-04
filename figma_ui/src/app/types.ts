export interface OmegaBoundaryResult {
  risk_level: string;
  severity_score: number;
  explanation: string;
  risk_dimensions?: Record<string, number>;
  human_review_required: boolean;
}

export interface OmegaReasoningTrace {
  passes_executed: number;
  initial_confidence: number;
  final_confidence: number;
  assumptions_extracted: number;
  logical_gaps_detected: number;
  boundary_severity: number;
  self_critique_applied: boolean;
  refinement_applied: boolean;
}

export interface ConfidenceEvolution {
  initial: number;
  post_debate?: number;
  post_boundary?: number;
  post_evidence?: number;
  post_stress?: number;
  final: number;
}

export interface BehavioralRiskProfile {
  self_preservation_score: number;
  manipulation_probability: number;
  evasion_index: number;
  confidence_inflation: number;
  overall_risk: number;
  risk_level: string;
  signals_detected: number;
  signal_breakdown: Record<string, number>;
  explanation: string;
}

export interface EvidenceSource {
  url: string;
  title: string;
  content_snippet: string;
  reliability_score: number;
  domain: string;
}

export interface EvidenceResult {
  query: string;
  sources: EvidenceSource[];
  source_count: number;
  contradictions: Record<string, unknown>[];
  contradiction_count: number;
  evidence_confidence: number;
  source_agreement: number;
  lineage: Record<string, string>[];
  search_executed: boolean;
}

export interface StressResult {
  stability_after_stress: number;
  contradictions_found: number;
  revised_confidence: number;
  overall_stability: number;
  vector_results: Record<string, unknown>;
  breakdown_points: string[];
}

export interface DebateResult {
  positions?: Array<{
    model: string;
    position: string;
    confidence: number;
    key_points: string[];
  }>;
  rounds?: number;
  consensus?: string;
  [key: string]: unknown;
}

export interface OmegaSessionState {
  session_id: string;
  chat_name?: string;
  primary_goal?: string;
  inferred_domain: string;
  user_expertise_score: number;
  message_count: number;
  error_patterns: Record<string, unknown>[];
  boundary_history_count: number;
  latest_boundary_severity: number;
  boundary_trend: string;
  disagreement_score: number;
  fragility_index: number;
  session_confidence: number;
  reasoning_depth: string;
}

export interface OmegaMetadata {
  omega_version: string;
  mode: string;
  sub_mode?: string;
  original_mode?: string;
  confidence: number;
  session_state?: OmegaSessionState;
  reasoning_trace?: OmegaReasoningTrace;
  boundary_result?: OmegaBoundaryResult;
  confidence_evolution?: ConfidenceEvolution;
  fragility_index?: number;
  behavioral_risk?: BehavioralRiskProfile;
  evidence_result?: EvidenceResult;
  stress_result?: StressResult;
  confidence_components?: Record<string, unknown>;
  debate_result?: DebateResult;
}

export interface SentinelRunResponse {
  chat_id: string;
  chat_name: string;
  mode: string;
  sub_mode?: string;
  original_mode?: string;
  formatted_output: string;
  data: {
    priority_answer: string;
  };
  confidence: number;
  session_state?: OmegaSessionState;
  reasoning_trace?: OmegaReasoningTrace;
  boundary_result?: OmegaBoundaryResult;
  omega_metadata?: OmegaMetadata;
}

export interface ChatHistoryItem {
  id: string;
  name: string;
  mode: string;
  created_at: string;
  updated_at: string;
  priority_answer?: string;
  machine_metadata?: OmegaMetadata;
  rounds?: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: string | null;
}

export interface HealthStatus {
  status: "healthy" | "degraded";
  version: string;
  omega_kernel: boolean;
  knowledge_learner: boolean;
  orchestrator: boolean;
  redis?: string;
  database?: string;
}

export interface KernelStatus {
  status: "online" | "offline";
  version: string;
  active_sessions: number;
  session_ids?: string[];
  sub_modes: string[];
  behavioral_analyzer?: boolean;
  evidence_engine?: boolean;
  message?: string;
}

export interface RootStatus {
  status: string;
  service: string;
  version: string;
  modes: string[];
  sub_modes: string[];
  omega_active: boolean;
  learning_active: boolean;
}

export interface SessionStats {
  total_sessions: number;
  mode_distribution: Record<string, number>;
  active_omega_sessions: number;
}

export interface LearningSummary {
  status: string;
  summary?: Record<string, unknown>;
  threshold_suggestions?: Record<string, unknown>;
  risk_profiles?: Record<string, unknown>;
  claim_type_risks?: Record<string, unknown>;
  message?: string;
}

export interface CrossAnalysisStep {
  step: number;
  type: "individual" | "consensus";
  analyzer?: string;
  analyzers?: string[];
  subject: string;
  subject_id: string;
  description: string;
  status: "success" | "error";
  error?: string;
  scores: {
    manipulation_level: number;
    risk_level: number;
    self_preservation: number;
    evasion_index: number;
    confidence_inflation: number;
  }
}

export interface CrossAnalysisModelProfile {
  name: string;
  color: string;
  status: "analyzed" | "no_data";
  scores: Record<string, number>;
  overall_risk?: string;
  step_count: number;
  key_signals?: string[];
  individual_steps?: CrossAnalysisStep[];
  consensus_steps?: CrossAnalysisStep[];
}

export interface CrossAnalysisResult {
  pipeline_version: string;
  timestamp: string;
  elapsed_seconds: number;
  steps_completed: number;
  steps_total: number;
  steps: CrossAnalysisStep[];
  model_profiles: Record<string, CrossAnalysisModelProfile>;
  analyzed_models: { id: string; name: string; color: string; analyzed_in_steps: number[] }[];
  overall_risk: {
    level: string;
    average_threat: number;
    average_manipulation: number;
    average_risk: number;
    max_threat: number;
    models_analyzed: number;
  }
}

export interface OmegaSessionResponse {
  chat_id: string;
  session_state: OmegaSessionState | null;
  boundary_trend?: string;
  initialized: boolean;
}

export interface SessionDescriptive {
  chat_name: string;
  goal: string;
  domain: string;
  domain_key: string;
  expertise: {
    label: string;
    score: number;
    description: string;
  };
  confidence: {
    label: string;
    score: number;
  };
  fragility: {
    label: string;
    score: number;
  }
}

export interface AdminRequestData {
  name: string;
  email: string;
  organization: string;
  reason: string;
}
