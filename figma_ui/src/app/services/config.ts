// ============================================================
// Environment Configuration
// Supports staging, production, and local development
// ============================================================

const ENV = {
  /** API base URL — override via VITE_API_BASE env var */
  API_BASE: import.meta.env.VITE_API_BASE || "http://localhost:8000",

  /** Request timeout in milliseconds */
  REQUEST_TIMEOUT: Number(import.meta.env.VITE_REQUEST_TIMEOUT) || 30000,

  /** Max retry attempts for transient failures */
  MAX_RETRIES: Number(import.meta.env.VITE_MAX_RETRIES) || 2,

  /** Retry base delay in ms (uses exponential backoff) */
  RETRY_DELAY: Number(import.meta.env.VITE_RETRY_DELAY) || 1000,

  /** Health check polling interval in ms */
  HEALTH_POLL_INTERVAL: 15000,

  /** Session analytics polling interval in ms */
  SESSION_POLL_INTERVAL: 8000,

  /** Enable debug logging */
  DEBUG: import.meta.env.VITE_DEBUG === "true",
} as const;

export default ENV;
