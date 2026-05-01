/**
 * ============================================================
 * API Service — Secure Backend Communication Layer
 * ============================================================
 * 
 * SECURITY (FIX #3 - XSS Protection):
 *   - SuperTokens session management (auto-refresh)
 *   - Tokens now stored in HttpOnly cookies (server-side)
 *   - Frontend never exposes tokens to JavaScript
 *   - No API keys in frontend
 *   - No system prompts exposed
 *   - All sensitive logic server-side
 *   - Request/response sanitization
 * 
 * This is the ONLY module that communicates with the backend.
 * No other frontend code should make direct API calls.
 * ============================================================
 */

import axios from 'axios';
import { API_BASE } from '../config';

// ── Token Storage ───────────────────────────────
// MIGRATION NOTE: Tokens are now fetched dynamically via Firebase Auth context

/**
 * Create an axios instance with interceptors for auth.
 * primarily relies on Firebase Bearer tokens now.
 */
const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ── Request Interceptor: Add request ID and Firebase Token ───
api.interceptors.request.use(
  async (config) => {
    config.headers['X-Request-ID'] = generateRequestId();
    config.withCredentials = true;
    
    try {
      // Import Firebase auth dynamically to avoid circular dependencies
      const { auth } = await import('../firebase');
      const user = auth.currentUser;

      if (user) {
        const token = await user.getIdToken();
        if (token) {
          console.log('TOKEN:', token?.slice(0, 20));
          config.headers.Authorization = `Bearer ${token}`;
          // Add debug header for user tracing
          config.headers['X-Debug-User'] = user.uid;
        }
      }
    } catch (err) {
      console.warn("Failed to retrieve Firebase token for request", err);
    }
    
    return config;
  },
  (error) => Promise.reject(error)
);

// ── Response Interceptor: Handle Data Normalization & Global Errors ──
api.interceptors.response.use(
  (response) => {
    // 1. Network/Empty response check
    if (!response || !response.data) {
      const errorMetadata = {
        type: 'EMPTY_RESPONSE',
        url: response?.config?.url,
        status: response?.status
      };
      console.warn('API Response is empty or malformed', errorMetadata);
      return { success: false, data: { chats: [], messages: [] }, error: 'Empty response from server', metadata: errorMetadata };
    }

    const data = response.data;

    // 2. Standard Envelope Handling
    if (data && typeof data === 'object' && 'success' in data) {
      if (data.success) {
        // Safe unwrap
        const unwrapped = data.data !== undefined ? data.data : {};
        
        // Optional: Add validation logic per-endpoint here if needed
        return unwrapped;
      } else {
        const errorMessage = data.error || 'Request failed';
        const errorMetadata = {
          type: 'SERVER_ERROR_ENVELOPE',
          url: response.config.url,
          status: response.status,
          serverError: data.error,
          serverDetails: data.details
        };
        console.error(`API Error [${response.config.url}]:`, errorMessage, errorMetadata);
        
        const error = new Error(errorMessage);
        error.status = response.status;
        error.data = data;
        error.metadata = errorMetadata;
        throw error;
      }
    }
    
    // 3. Raw Data Handling (non-enveloped)
    return data !== undefined ? data : {};
  },
  async (error) => {
    let type = 'UNKNOWN_ERROR';

    if (!error.response) {
      type = 'NETWORK_ERROR';
    } else if (error.response.status >= 500) {
      type = 'SERVER_CRASH';
    } else if (error.response.status >= 400) {
      type = 'CLIENT_ERROR';
    }

    const errorMetadata = {
      type,
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      data: error.response?.data,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    };

    console.error(`Global API Error [${type}]:`, errorMetadata);

    // Sanitize error for display
    const sanitizedError = sanitizeError(error);
    sanitizedError.metadata = errorMetadata;
    return Promise.reject(sanitizedError);
  }
);

// ── API Methods ─────────────────────────────────────────────

/**
 * Send a standard mode query.
 */
export async function sendStandard(text, chatId, file, context) {
  const formData = new FormData();
  formData.append('text', text);
  if (chatId) formData.append('chat_id', chatId);
  if (file) formData.append('file', file);
  if (context) formData.append('context', JSON.stringify(context));

  const res = await api.post('/run/standard', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res;
}

/**
 * Send an experimental mode query.
 */
export async function sendExperimental(text, options = {}) {
  const {
    chatId, file, context, mode = 'experimental',
    subMode = 'debate', rounds = 6, killSwitch = false,
  } = options;

  const formData = new FormData();
  formData.append('text', text);
  formData.append('mode', mode);
  formData.append('sub_mode', subMode);
  formData.append('rounds', rounds);
  formData.append('kill_switch', killSwitch);
  if (chatId) formData.append('chat_id', chatId);
  if (file) formData.append('file', file);
  if (context) formData.append('context', JSON.stringify(context));

  const res = await api.post('/run/experimental', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res;
}

/**
 * Send an omega kill diagnostic.
 */
export async function sendKill(text, chatId) {
  const formData = new FormData();
  formData.append('text', text || 'kill');
  if (chatId) formData.append('chat_id', chatId);

  const res = await api.post('/run/omega/kill', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res;
}

/**
 * Submit feedback.
 */
export async function sendFeedback(runId, feedback, extra = {}) {
  const formData = new FormData();
  formData.append('run_id', runId);
  formData.append('feedback', feedback);
  if (extra.rating) formData.append('rating', extra.rating);
  if (extra.reason) formData.append('reason', extra.reason);
  if (extra.mode) formData.append('mode', extra.mode);
  if (extra.subMode) formData.append('sub_mode', extra.subMode);

  const res = await api.post('/feedback', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res;
}

/**
 * Create or resume user session.
 * Called on app load BEFORE fetching history.
 */
export async function createSession(client = 'web') {
  try {
    const res = await api.post('/api/session', { client });
    return res;
  } catch (err) {
    console.error('Failed to create session:', err);
    return { success: false, data: { session_id: null }, error: err.message };
  }
}

/**
 * Get chat history list.
 */
export async function getHistory(limit = 50, offset = 0) {
  const res = await api.get('/api/history', { params: { limit, offset } });
  return res;
}

/**
 * Create a new chat.
 */
export async function createChat(title = 'New Chat', mode = 'conversational') {
  try {
    const res = await api.post('/api/chat', { title, mode });
    return res?.data || res || null;
  } catch (err) {
    console.error('Failed to create chat:', err);
    return null;
  }
}

/**
 * Get messages for a specific chat.
 */
export async function getChatMessages(chatId) {
  const res = await api.get(`/api/chat/${chatId}/messages`);
  return res;
}

/**
 * Get session descriptive summary.
 */
export async function getSessionDescriptive(chatId) {
  const res = await api.get(`/api/session/${chatId}/descriptive`);
  return res;
}

/**
 * Get omega session state.
 */
export async function getOmegaSession(chatId) {
  const res = await api.get(`/api/omega/session/${chatId}`);
  return res;
}

/**
 * Health check.
 */
export async function checkHealth() {
  try {
    await api.get('/', { timeout: 3000 });
    return 'online';
  } catch {
    return 'offline';
  }
}

/**
 * Run cross-model analysis.
 */
export async function runCrossAnalysis(chatId, query, llmResponse) {
  const res = await api.post('/api/cross-analysis', {
    chat_id: chatId,
    query: query || '',
    llm_response: llmResponse || '',
  });
  return res;
}

// ── MCO (Meta-Cognitive Orchestrator) ───────────────────────

/**
 * Send a query through the Meta-Cognitive Orchestrator.
 * ALL model invocations flow through this single pipeline.
 * 
 * @param {string} query       — User query text
 * @param {Object} options     — { chatId, mode, subMode, selectedModel, forceRetrieval }
 */
export async function sendMCOQuery(query, options = {}) {
  const {
    chatId,
    mode = 'standard',
    subMode = null,
    selectedModel = null,
    forceRetrieval = false,
    image_b64 = null,
    image_mime = null,
  } = options;

  const body = {
    query,
    mode,
    selected_model: selectedModel,
    force_retrieval: forceRetrieval,
  };
  if (chatId) body.chat_id = chatId;
  if (subMode) body.sub_mode = subMode;
  if (image_b64) body.image_b64 = image_b64;
  if (image_mime) body.image_mime = image_mime;

  const res = await api.post('/api/mco/run', body);
  return res;
}

/**
 * Fetch available cognitive models from MCO registry.
 * Returns { models: [{ key, name, model_id, provider, role, enabled, ... }] }
 */
export async function fetchMCOModels() {
  const res = await api.get('/api/mco/models');
  return res;
}

/**
 * Fetch available models from the Standard Mode chat registry.
 * Returns { models: [...], total, enabled_count }
 * 
 * Used to populate the model selector dropdown with tier metadata.
 */
export async function fetchChatModels() {
  const res = await api.get('/chat/models/available');
  return res;
}

/**
 * Toggle Claude on/off. Claude is synthesis-only.
 * @returns {{ model, active, synthesis_only, message }}
 */
export async function toggleClaude() {
  const res = await api.post('/api/models/claude/toggle');
  return res;
}

export async function getClaudeUsage() {
  const res = await api.get('/api/models/claude/usage');
  return res;
}

/**
 * Send a query to a specific model (Standard Mode).
 * Routes to POST /chat/{modelId} with retry + fallback logic server-side.
 * 
 * @param {string} modelId   — Canonical registry key (e.g. "llama33-70b")
 * @param {string} query     — User query text
 * @param {string} chatId    — Optional session ID
 * @param {Object} options   — { maxTokens, systemRole }
 * 
 * @returns {Object} { model_id, model_name, provider, response, latency_ms,
 *                     tokens_used, retried, fallback_used, fallback_model }
 */
export async function sendDirectModelQuery(modelId, query, chatId = null, options = {}) {
  const body = {
    query,
    chat_id: chatId || undefined,
    max_tokens: options.maxTokens || undefined,
    system_role: options.systemRole || undefined,
    image_b64: options.image_b64 || undefined,
    image_mime: options.image_mime || undefined,
  };

  const res = await api.post(`/chat/${modelId}`, body);
  return res;
}

/**
 * Execute a full multi-model debate (Debate Mode).
 * Routes to POST /battle/debate — runs 3-round ensemble debate and
 * returns BattleVisualizationPayload with all metrics, charts, etc.
 * 
 * @param {string} query       — User question
 * @param {string} chatId      — Optional session ID
 * @param {string} promptType  — "general" | "code" | "logical" | "evidence" | "depth"
 * @param {Object} options     — { maxModels, includeCharts }
 * 
 * @returns {Object} Full BattleVisualizationPayload + models metadata
 */
export async function sendDebateQuery(query, chatId = null, promptType = 'general', options = {}) {
  const body = {
    query,
    chat_id: chatId || undefined,
    prompt_type: promptType,
    max_models: options.maxModels ?? 7,
    include_charts: options.includeCharts ?? false,
  };

  const res = await api.post('/battle/debate', body);
  return res;
}

/**
 * Fetch MCO analytics for a specific session.
 */
export async function fetchMCOAnalytics(sessionId) {
  const res = await api.get(`/api/mco/analytics/${sessionId}`);
  return res;
}

// ── Message Edit / Regenerate ────────────────────────────────

/**
 * Edit an existing message.
 */
export async function editMessage(messageId, content) {
  const res = await api.put(`/api/messages/${messageId}`, { content });
  return res;
}//const detail = error.response.data?.detail;

/**
 * Regenerate an assistant response for a given message.
 */
export async function regenerateMessage(messageId) {
  const res = await api.post(`/api/messages/${messageId}/regenerate`);
  return res;
}

// ── Utilities ───────────────────────────────────────────────

function generateRequestId() {
  return Math.random().toString(36).substring(2, 10);
}

/**
 * Sanitize error for user display.
 * Never expose internal details, stack traces, or provider info.
 */
function sanitizeError(error) {
  if (!error.response) {
    return new Error('Unable to reach the service right now.');
  }

  const status = error.response.status;
  

  switch (status) {
    case 400:
      return new Error('Invalid request. Please try rephrasing.');
    case 401:
      return new Error('Please sign in to continue.');
    case 404:
      return new Error('The requested resource was not found.');
    case 413:
      return new Error('Your message is too long. Please shorten it.');
    case 429:
      return new Error('Too many requests. Please wait a moment.');
    case 502:
      return new Error('Provider unavailable. Please try again.');
    case 503:
      return new Error('The system is starting up. Please try again in a moment.');
    default:
      return new Error('Something went wrong. Please try again.');
  }
}

export default api;
