// ============================================================
// Sentinel-E FastAPI Backend Integration
// All calls routed through apiClient (retry, timeout, errors)
// All responses normalized through adapter (safe defaults)
// ============================================================

import { apiRequest, postForm, postJson, getQuick, ApiError } from "./services/apiClient";
import {
  adaptRunResponse,
  adaptChatHistoryItem,
  adaptChatMessage,
  adaptHealth,
  adaptKernel,
  adaptSessionDescriptive,
  adaptSessionStats,
  adaptLearningSummary,
  adaptCrossAnalysis,
} from "./services/adapter";
import ENV from "./services/config";
import { supabase } from "./lib/supabase";
import { MODEL_RUNTIME_MAP, ORCHESTRATION_MODE_MAP } from "./config/runtime";

export { ApiError };

// ============================================================
// TYPES — Matched to backend/sentinel/schemas.py
// ============================================================

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
  forensic_result?: EvidenceResult;
  audit_result?: Record<string, unknown>;
  glass_result?: Record<string, unknown>;
  synthesis_result?: Record<string, unknown>;
  kill_active?: boolean;
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
  messages?: ChatMessage[];
}

export interface ChatMessage {
  id: string;
  chat_id?: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string | null;
  reasoning_json?: OmegaMetadata;
}

export interface ChatHistoryBundle {
  chats: ChatHistoryItem[];
  messages: ChatMessage[];
}

export interface RuntimeModelStatus {
  id: string;
  name: string;
  provider?: string;
  enabled: boolean;
  active?: boolean;
  disable_reason?: string | null;
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
  };
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
    threat_level: number;
    overall_risk: string;
    reasoning?: string;
    consensus_reasoning?: string;
    agreement_score?: number;
    key_signals: string[];
  };
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
  };
  disagreement: {
    score: number;
    label: string;
  };
  message_count: number;
  reasoning_depth: string;
  error_count: number;
  boundary_count: number;
  last_boundary_severity: number;
  error?: string;
}

// ============================================================
// API CALLS — routed through apiClient with adapter normalization
// ============================================================

/**
 * Root status — GET /
 */
export async function getRootStatus(): Promise<RootStatus | null> {
  try {
    return await getQuick<RootStatus>("/", 2000);
  } catch {
    return null;
  }
}

/**
 * Health check — GET /health
 */
export async function checkHealth(): Promise<HealthStatus | null> {
  try {
    const raw = await getQuick<Record<string, unknown>>("/health", 3000);
    return adaptHealth(raw);
  } catch {
    return null;
  }
}

/**
 * Kernel status — GET /kernel-status
 */
export async function getKernelStatus(): Promise<KernelStatus | null> {
  try {
    const raw = await apiRequest<Record<string, unknown>>("/kernel-status", { retries: 0 });
    return adaptKernel(raw);
  } catch {
    return null;
  }
}

/**
 * Session stats — GET /session-stats
 */
export async function getSessionStats(): Promise<SessionStats | null> {
  try {
    const raw = await apiRequest<Record<string, unknown>>("/session-stats", { retries: 0 });
    return adaptSessionStats(raw);
  } catch {
    return null;
  }
}

// ============================================================
// RUN ENDPOINTS
// ============================================================

async function fileToBase64(file: File): Promise<{ b64: string; mime: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      const [header, b64] = dataUrl.split(",");
      const mimeMatch = header.match(/:(.*?);/);
      resolve({
        b64,
        mime: mimeMatch ? mimeMatch[1] : file.type || "application/octet-stream",
      });
    };
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(file);
  });
}

/**
 * Run Standard mode — POST /api/mco/run (JSON)
 */
export async function runStandard(
  text: string,
  modelId: string,
  chatId?: string,
  file?: File,
  signal?: AbortSignal,
  options?: { responseStyle?: string; preferences?: Record<string, unknown> }
): Promise<SentinelRunResponse> {
  const mappedModel = MODEL_RUNTIME_MAP[modelId]?.model || modelId;
  const payload: Record<string, any> = {
    runtime: "single-model",
    selected_model: mappedModel,
    mode: "standard",
    query: text,
  };
  if (options?.responseStyle) payload.response_style = options.responseStyle;
  if (options?.preferences) payload.preferences = options.preferences;
  
  if (chatId) payload.chat_id = chatId;
  
  if (file) {
    const { b64, mime } = await fileToBase64(file);
    payload.image_b64 = b64;
    payload.image_mime = mime;
  }

  const raw = await postJson<Record<string, unknown>>("/api/mco/run", payload, { signal });
  return adaptRunResponse(raw);
}

/**
 * Run Experimental mode — POST /api/mco/run (JSON)
 * Sub-modes: debate, glass, evidence
 */
export async function runExperimental(
  text: string,
  subMode: string = "debate",
  rounds: number = 6,
  chatId?: string,
  killSwitch: boolean = false,
  file?: File,
  signal?: AbortSignal,
  options?: { responseStyle?: string; preferences?: Record<string, unknown> }
): Promise<SentinelRunResponse> {
  const activeSubMode = killSwitch ? "glass" : subMode;
  const mappedMode = ORCHESTRATION_MODE_MAP[activeSubMode]?.mode || activeSubMode;
  const payload: Record<string, any> = {
    runtime: "mco",
    mode: "experimental",
    sub_mode: mappedMode,
    orchestration: true,
    rounds,
    query: killSwitch ? "kill" : text,
  };
  if (options?.responseStyle) payload.response_style = options.responseStyle;
  if (options?.preferences) {
    payload.preferences = options.preferences;
  }
  
  if (chatId) payload.chat_id = chatId;
  
  if (file) {
    const { b64, mime } = await fileToBase64(file);
    payload.image_b64 = b64;
    payload.image_mime = mime;
  }

  const raw = await postJson<Record<string, unknown>>("/api/mco/run", payload, { signal });
  return adaptRunResponse(raw);
}

/**
 * Run Omega Standard — POST /run/omega/standard (FormData)
 */
export async function runOmegaStandard(
  text: string,
  chatId?: string,
  file?: File,
  signal?: AbortSignal
): Promise<SentinelRunResponse> {
  const formData = new FormData();
  formData.append("text", text);
  if (chatId) formData.append("chat_id", chatId);
  if (file) formData.append("file", file);

  const raw = await postForm<Record<string, unknown>>("/run/omega/standard", formData, { signal });
  return adaptRunResponse(raw);
}

/**
 * Run Omega Experimental — POST /run/omega/experimental (FormData)
 */
export async function runOmegaExperimental(
  text: string,
  subMode: string = "debate",
  rounds: number = 3,
  chatId?: string,
  file?: File,
  signal?: AbortSignal
): Promise<SentinelRunResponse> {
  const formData = new FormData();
  formData.append("text", text);
  formData.append("sub_mode", subMode);
  formData.append("rounds", rounds.toString());
  if (chatId) formData.append("chat_id", chatId);
  if (file) formData.append("file", file);

  const raw = await postForm<Record<string, unknown>>("/run/omega/experimental", formData, { signal });
  return adaptRunResponse(raw);
}

/**
 * Run Omega Kill — POST /run/omega/kill (FormData)
 * Diagnostic mode: returns session cognitive state without new reasoning
 */
export async function runOmegaKill(
  chatId?: string,
  text: string = "",
  signal?: AbortSignal
): Promise<SentinelRunResponse> {
  const formData = new FormData();
  formData.append("text", text || "kill");
  if (chatId) formData.append("chat_id", chatId);

  const raw = await postForm<Record<string, unknown>>("/run/omega/kill", formData, { signal });
  return adaptRunResponse(raw);
}

/**
 * Unified JSON endpoint — POST /api/run
 */
export async function runSentinel(params: {
  text: string;
  mode?: string;
  sub_mode?: string;
  rounds?: number;
  chat_id?: string;
  enable_shadow?: boolean;
  kill?: boolean;
  role_map?: Record<string, string>;
}, signal?: AbortSignal): Promise<SentinelRunResponse> {
  const raw = await postJson<Record<string, unknown>>("/api/run", {
    text: params.text,
    mode: params.mode || "standard",
    sub_mode: params.sub_mode || null,
    rounds: params.rounds || 1,
    chat_id: params.chat_id || null,
    enable_shadow: params.enable_shadow || false,
    kill: params.kill || false,
    role_map: params.role_map || null,
  }, { signal });
  return adaptRunResponse(raw);
}

// ============================================================
// CHAT HISTORY & MESSAGES
// ============================================================

/**
 * List chats — GET /api/chats
 */
export async function getChatHistory(
  limit: number = 50,
  offset: number = 0
): Promise<ChatHistoryBundle> {
  try {
    const res = await apiRequest<{ success: boolean; data: any }>(
      `/api/history?limit=${limit}&offset=${offset}`,
      { retries: 1 }
    );
    if (!res || !res.success || !res.data?.chats) return { chats: [], messages: [] };

    const rawMessages = Array.isArray(res.data.messages)
      ? res.data.messages
      : [];
    const messages = rawMessages.map(adaptChatMessage);

    const messagesByChat = new Map<string, ChatMessage[]>();
    for (const msg of messages) {
      const chatId = (msg as any).chat_id || null;
      if (!chatId) continue;
      const list = messagesByChat.get(chatId) || [];
      list.push(msg);
      messagesByChat.set(chatId, list);
    }

    const chats = res.data.chats.map((raw: Record<string, unknown>) => {
      const chat = adaptChatHistoryItem(raw);
      const embedded = Array.isArray((raw as any).messages)
        ? (raw as any).messages.map(adaptChatMessage)
        : [];
      const byChat = messagesByChat.get(chat.id) || [];
      const mergedMessages = embedded.length > 0 ? embedded : byChat;
      return {
        ...chat,
        messages: mergedMessages,
      };
    });

    return { chats, messages };
  } catch (err) {
    console.error('getChatHistory fallback error:', err);
    return { chats: [], messages: [] };
  }
}

/**
 * Get full chat details
 */
export async function getChatDetails(
  chatId: string
): Promise<{ chat: ChatHistoryItem; messages: ChatMessage[] }> {
  try {
    const chatRes = await apiRequest<{ success: boolean; data: any }>(`/api/v2/chat/${chatId}`);
    if (!chatRes || !chatRes.success || !chatRes.data) throw new Error('Chat not found');

    const chatData = chatRes.data.chat || {};
    const chat: ChatHistoryItem = {
      id: chatData.id || chatId,
      name: chatData.title || chatData.chat_name || chatData.name || 'New Chat',
      mode: chatData.mode || 'standard',
      created_at: chatData.created_at,
      updated_at: chatData.updated_at || chatData.created_at,
      priority_answer: chatData.priority_answer,
      machine_metadata: chatData.machine_metadata ? adaptMetadata(chatData.machine_metadata) : undefined,
      rounds: chatData.rounds,
    };

    const messages: ChatMessage[] = (chatRes.data.messages || []).map(adaptChatMessage);

    return { chat, messages };
  } catch (err) {
    console.error('getChatDetails error:', err);
    return {
      chat: { id: chatId, name: 'Unknown', mode: 'standard', created_at: '', updated_at: '' },
      messages: []
    };
  }
}

/**
 * Get messages for a chat
 */
export async function getChatMessages(
  chatId: string
): Promise<ChatMessage[]> {
  try {
    const res = await apiRequest<{ success: boolean; data: any }>(`/api/v2/chat/${chatId}/messages`);
    if (!res || !res.success || !res.data) return [];
    
    const messages = Array.isArray(res.data) ? res.data : (res.data.messages || []);
    return messages.map(adaptChatMessage);
  } catch (err) {
    console.error('getChatMessages error:', err);
    return [];
  }
}

/**
 * Sync a new or updated conversation.
 * The live backend runtime persists conversations during /api/mco/run.
 */
export async function syncConversation(chatId: string, title: string, mode: string): Promise<void> {
  void chatId;
  void title;
  void mode;
}

/**
 * Sync a message to Supabase (Now via v2 API)
 */
export async function syncMessage(chatId: string, role: string, content: string): Promise<void> {
  // NO-OP: Phase 1 architecture dictates that the backend API routes 
  // (e.g. /api/chat/...) automatically persist both user and assistant messages.
  // Frontend no longer needs to explicitly sync individual messages to the DB.
}

/**
 * Share a chat
 */
export async function shareChat(
  chatId: string
): Promise<{ share_token: string }> {
  return postJson<{ share_token: string }>("/api/v2/chat/share", { chat_id: chatId });
}

/**
 * Update chat title / metadata
 */
export async function updateChatMetadata(
  chatId: string,
  payload: { title?: string; mode?: string; machine_metadata?: Record<string, unknown> }
): Promise<void> {
  await apiRequest(`/api/v2/chat/${chatId}`, {
    method: "PUT",
    body: payload,
    json: true,
  });
}

/**
 * Update user settings
 */
export async function updateUserSettings(
  settings: Record<string, any>
): Promise<void> {
  await apiRequest("/api/v2/user/settings", {
    method: "PUT",
    body: settings,
    json: true,
  });
}

/**
 * Get backend model registry
 */
export async function getModelRegistry(): Promise<RuntimeModelStatus[]> {
  try {
    const res = await apiRequest<{ success: boolean; data: any }>("/api/models/status", { retries: 1 });
    if (!res?.success || !Array.isArray(res.data?.models)) return [];
    return res.data.models.map((m: any) => ({
      id: String(m.id || ""),
      name: String(m.name || m.id || "Unknown"),
      provider: m.provider ? String(m.provider) : undefined,
      enabled: Boolean(m.enabled),
      active: Boolean(m.active),
      disable_reason: m.disable_reason ? String(m.disable_reason) : null,
    }));
  } catch (err) {
    console.error("getModelRegistry error:", err);
    return [];
  }
}

// ============================================================
// FEEDBACK
// ============================================================

/**
 * Submit feedback to the backend runtime so it is stored on the conversation.
 */
export async function submitFeedback(params: {
  run_id: string;
  feedback: string;
  rating?: number;
  reason?: string;
  mode?: string;
  sub_mode?: string;
  boundary_severity?: number;
  fragility_index?: number;
  disagreement_score?: number;
  confidence?: number;
}): Promise<{ status: string; feedback_id?: string; storage?: string; learning?: boolean }> {
  try {
    const formData = new FormData();
    formData.append("run_id", params.run_id);
    formData.append("feedback", params.feedback);
    if (params.rating != null) formData.append("rating", String(params.rating));
    if (params.reason) formData.append("reason", params.reason);
    if (params.mode) formData.append("mode", params.mode);
    if (params.sub_mode) formData.append("sub_mode", params.sub_mode);
    if (params.boundary_severity != null) formData.append("boundary_severity", String(params.boundary_severity));
    if (params.fragility_index != null) formData.append("fragility_index", String(params.fragility_index));
    if (params.disagreement_score != null) formData.append("disagreement_score", String(params.disagreement_score));
    if (params.confidence != null) formData.append("confidence", String(params.confidence));

    const result = await postForm<{ status: string; feedback_id?: string }>("/feedback", formData);
    return { ...result, storage: "backend" };
  } catch (err) {
    console.error('submitFeedback fallback error:', err);
    return { status: "error" };
  }
}

// ============================================================
// OMEGA SESSION INTELLIGENCE
// ============================================================

/**
 * Get Omega session state — GET /api/omega/session/{chat_id}
 */
export async function getOmegaSession(
  chatId: string
): Promise<OmegaSessionResponse | null> {
  try {
    return await apiRequest<OmegaSessionResponse>(`/api/v2/omega/session/${chatId}`, { retries: 0 });
  } catch {
    return null;
  }
}

/**
 * Session descriptive summary — GET /api/session/{chat_id}/descriptive
 */
export async function getSessionDescriptive(
  chatId: string
): Promise<SessionDescriptive | null> {
  try {
    const raw = await apiRequest<Record<string, unknown>>(`/api/v2/session/${chatId}/descriptive`, { retries: 0 });
    return adaptSessionDescriptive(raw);
  } catch {
    return null;
  }
}

// ============================================================
// CROSS-MODEL ANALYSIS
// ============================================================

/**
 * Run cross-model analysis — POST /api/cross-analysis
 */
export async function runCrossAnalysis(params: {
  chat_id?: string;
  query?: string;
  llm_response?: string;
}): Promise<CrossAnalysisResult> {
  const raw = await postJson<Record<string, unknown>>("/api/cross-analysis", params as Record<string, unknown>);
  return adaptCrossAnalysis(raw);
}

/**
 * Get analyzed models — GET /api/cross-analysis/models
 */
export async function getCrossAnalysisModels(): Promise<{
  analyzed_models: string[];
  analysis_steps: string[];
  total_steps: number;
}> {
  return apiRequest<{ analyzed_models: string[]; analysis_steps: string[]; total_steps: number }>(
    "/api/cross-analysis/models"
  );
}

// ============================================================
// KNOWLEDGE LEARNER
// ============================================================

/**
 * Learning summary — GET /api/learning
 */
export async function getLearningSummary(): Promise<LearningSummary | null> {
  try {
    const raw = await apiRequest<Record<string, unknown>>("/api/learning", { retries: 0 });
    return adaptLearningSummary(raw);
  } catch {
    return null;
  }
}

/**
 * Risk profiles — GET /api/learning/risk-profiles
 */
export async function getLearningRiskProfiles(): Promise<Record<string, unknown> | null> {
  try {
    return await apiRequest<Record<string, unknown>>("/api/learning/risk-profiles", { retries: 0 });
  } catch {
    return null;
  }
}

// ============================================================
// ADMIN WORKFLOWS
// ============================================================

export interface AdminRequestData {
  name: string;
  email: string;
  organization: string;
  reason: string;
}

/**
 * Get current user profile and stats from backend
 */
export async function getCurrentUser(): Promise<any> {
  try {
    const res = await apiRequest<{ success: boolean; data: any }>("/api/v2/user", { retries: 1 });
    if (res && res.success) {
      return res.data;
    }
    return null;
  } catch (err) {
    console.error('getCurrentUser error:', err);
    return null;
  }
}

/**
 * Submit a request for admin access.
 */
export async function submitAdminRequest(data: AdminRequestData): Promise<{ status: "success" | "error", message?: string }> {
  try {
    const res = await apiRequest<{ status: string, message?: string }>('/admin/requests', {
      method: 'POST',
      body: {
        name: data.name,
        email: data.email,
        organization: data.organization,
        reason: data.reason
      },
      json: true
    });

    if (res && res.status === "success") {
      return { status: "success" };
    } else {
      return { status: "error", message: res?.message || "Failed to submit request" };
    }
  } catch (err: any) {
    console.error("submitAdminRequest exception:", err);
    return { status: "error", message: err.message || "Failed to submit request" };
  }
}

/**
 * Check the status of a user's admin request.
 */
export async function getAdminRequestStatus(email: string): Promise<'pending' | 'approved' | 'rejected' | null> {
  try {
    const res = await apiRequest<{ status: string, data: any[] }>('/admin/requests', { method: 'GET' });
    if (res && res.status === "success" && res.data) {
      const myRequest = res.data.find((r: any) => r.email === email);
      if (myRequest) {
        return myRequest.status as 'pending' | 'approved' | 'rejected';
      }
    }
    return null;
  } catch (err) {
    console.error("getAdminRequestStatus exception:", err);
    return null;
  }
}
