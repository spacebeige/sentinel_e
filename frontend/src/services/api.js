/**
 * ============================================================
 * API Service — Secure Backend Communication Layer
 * ============================================================
 * 
 * SECURITY:
 *   - JWT token management (auto-refresh)
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

const API_BASE = process.env.REACT_APP_API_URL || '';

// ── Token Storage (memory-only, not localStorage) ───────────
let _accessToken = null;
let _refreshToken = null;
let _sessionId = null;
let _tokenRefreshPromise = null;

/**
 * Create an axios instance with interceptors for auth.
 */
const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ── Request Interceptor: Attach JWT ─────────────────────────
api.interceptors.request.use(
  (config) => {
    if (_accessToken) {
      config.headers.Authorization = `Bearer ${_accessToken}`;
    }
    // Add request ID for tracing
    config.headers['X-Request-ID'] = generateRequestId();
    return config;
  },
  (error) => Promise.reject(error)
);

// ── Response Interceptor: Handle 401, retry with refresh ────
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        await refreshSession();
        originalRequest.headers.Authorization = `Bearer ${_accessToken}`;
        return api(originalRequest);
      } catch (refreshError) {
        // Session expired, bootstrap new one
        await initSession();
        originalRequest.headers.Authorization = `Bearer ${_accessToken}`;
        return api(originalRequest);
      }
    }

    // Sanitize error for display
    const sanitizedError = sanitizeError(error);
    return Promise.reject(sanitizedError);
  }
);

// ── Session Management ──────────────────────────────────────

/**
 * Initialize a new anonymous session.
 * Called once on app startup.
 */
export async function initSession() {
  try {
    const res = await axios.post(`${API_BASE}/api/auth/session`);
    _accessToken = res.data.access_token;
    _refreshToken = res.data.refresh_token;
    _sessionId = res.data.session_id;
    return res.data;
  } catch (error) {
    console.warn('Session init failed, running without auth');
    return null;
  }
}

/**
 * Refresh the access token using the refresh token.
 */
async function refreshSession() {
  if (_tokenRefreshPromise) return _tokenRefreshPromise;

  _tokenRefreshPromise = axios.post(`${API_BASE}/api/auth/refresh`, {
    refresh_token: _refreshToken,
  }).then((res) => {
    _accessToken = res.data.access_token;
    _tokenRefreshPromise = null;
    return res.data;
  }).catch((err) => {
    _tokenRefreshPromise = null;
    throw err;
  });

  return _tokenRefreshPromise;
}

export function getSessionId() {
  return _sessionId;
}

export function isAuthenticated() {
  return !!_accessToken;
}

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
  return res.data;
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
  return res.data;
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
  return res.data;
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
  return res.data;
}

/**
 * Get chat history list.
 */
export async function getHistory(limit = 50, offset = 0) {
  const res = await api.get('/api/history', { params: { limit, offset } });
  return res.data;
}

/**
 * Get messages for a specific chat.
 */
export async function getChatMessages(chatId) {
  const res = await api.get(`/api/chat/${chatId}/messages`);
  return res.data;
}

/**
 * Get session descriptive summary.
 */
export async function getSessionDescriptive(chatId) {
  const res = await api.get(`/api/session/${chatId}/descriptive`);
  return res.data;
}

/**
 * Get omega session state.
 */
export async function getOmegaSession(chatId) {
  const res = await api.get(`/api/omega/session/${chatId}`);
  return res.data;
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
  return res.data;
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
    return new Error('Unable to reach the server. Please check your connection.');
  }

  const status = error.response.status;
  const detail = error.response.data?.detail;

  switch (status) {
    case 400:
      return new Error(detail || 'Invalid request. Please try rephrasing.');
    case 401:
      return new Error('Session expired. Please refresh the page.');
    case 404:
      return new Error('The requested resource was not found.');
    case 413:
      return new Error('Your message is too long. Please shorten it.');
    case 429:
      return new Error('Too many requests. Please wait a moment.');
    case 503:
      return new Error('The system is starting up. Please try again in a moment.');
    default:
      return new Error('Something went wrong. Please try again.');
  }
}

export default api;
