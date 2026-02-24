// ============================================================
// Centralized API Client
// Retry, timeout, abort, error normalization
// ============================================================

import ENV from "./config";

export class ApiError extends Error {
  status: number;
  detail: string;
  retryable: boolean;

  constructor(status: number, detail: string, retryable = false) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.retryable = retryable;
  }
}

interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE" | "PATCH";
  body?: BodyInit | Record<string, unknown> | null;
  headers?: Record<string, string>;
  timeout?: number;
  retries?: number;
  signal?: AbortSignal;
  /** If true, body is sent as JSON; otherwise as-is (FormData, etc.) */
  json?: boolean;
}

const RETRYABLE_STATUSES = new Set([408, 429, 500, 502, 503, 504]);

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Core fetch wrapper with timeout, retry, and structured error handling.
 */
export async function apiRequest<T = unknown>(
  path: string,
  options: RequestOptions = {}
): Promise<T> {
  const {
    method = "GET",
    body = null,
    headers = {},
    timeout = ENV.REQUEST_TIMEOUT,
    retries = ENV.MAX_RETRIES,
    signal,
    json = false,
  } = options;

  const url = path.startsWith("http") ? path : `${ENV.API_BASE}${path}`;

  const finalHeaders: Record<string, string> = { ...headers };
  let finalBody: BodyInit | null = null;

  if (body && json) {
    finalHeaders["Content-Type"] = "application/json";
    finalBody = JSON.stringify(body);
  } else if (body) {
    finalBody = body as BodyInit;
  }

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timeoutId = timeout > 0 ? setTimeout(() => controller.abort(), timeout) : null;

    // Merge external signal
    if (signal) {
      signal.addEventListener("abort", () => controller.abort(), { once: true });
    }

    try {
      if (ENV.DEBUG) {
        console.debug(`[API] ${method} ${url} attempt=${attempt + 1}/${retries + 1}`);
      }

      const res = await fetch(url, {
        method,
        headers: finalHeaders,
        body: finalBody,
        signal: controller.signal,
      });

      if (timeoutId) clearTimeout(timeoutId);

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({ detail: res.statusText }));
        const retryable = RETRYABLE_STATUSES.has(res.status);
        const err = new ApiError(
          res.status,
          errBody.detail || errBody.message || `HTTP ${res.status}`,
          retryable
        );

        if (retryable && attempt < retries) {
          lastError = err;
          const delay = ENV.RETRY_DELAY * Math.pow(2, attempt);
          if (ENV.DEBUG) console.debug(`[API] Retrying in ${delay}ms...`);
          await sleep(delay);
          continue;
        }

        throw err;
      }

      return (await res.json()) as T;
    } catch (err) {
      if (timeoutId) clearTimeout(timeoutId);

      if (err instanceof ApiError) throw err;

      const isAbort = err instanceof DOMException && err.name === "AbortError";
      if (isAbort && signal?.aborted) {
        // User-initiated abort — don't retry
        throw new ApiError(0, "Request cancelled", false);
      }
      if (isAbort) {
        throw new ApiError(408, "Request timed out", true);
      }

      // Network error — retryable
      if (attempt < retries) {
        lastError = err instanceof Error ? err : new Error(String(err));
        const delay = ENV.RETRY_DELAY * Math.pow(2, attempt);
        await sleep(delay);
        continue;
      }

      throw new ApiError(0, (err as Error).message || "Network error", true);
    }
  }

  throw lastError || new ApiError(0, "Request failed after retries", false);
}

/**
 * Convenience: POST JSON payload.
 */
export function postJson<T = unknown>(
  path: string,
  data: Record<string, unknown>,
  opts?: Omit<RequestOptions, "method" | "body" | "json">
): Promise<T> {
  return apiRequest<T>(path, { ...opts, method: "POST", body: data, json: true });
}

/**
 * Convenience: POST FormData.
 */
export function postForm<T = unknown>(
  path: string,
  formData: FormData,
  opts?: Omit<RequestOptions, "method" | "body" | "json">
): Promise<T> {
  return apiRequest<T>(path, { ...opts, method: "POST", body: formData });
}

/**
 * Convenience: GET with short timeout (health checks, etc.).
 */
export function getQuick<T = unknown>(
  path: string,
  timeoutMs = 3000
): Promise<T> {
  return apiRequest<T>(path, { timeout: timeoutMs, retries: 0 });
}
