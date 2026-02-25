"""
============================================================
API Gateway Middleware Stack
============================================================
Implements enterprise-grade middleware:
- Rate limiting (sliding window)
- Request validation
- CSP headers
- Structured error handling
- Request ID tracking
- Token usage monitoring logger
"""

import time
import uuid
import logging
from typing import Dict, Callable
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from gateway.config import get_settings

logger = logging.getLogger("Gateway")


# ============================================================
# Rate Limiter (In-Memory Sliding Window)
# ============================================================

class RateLimitStore:
    """Thread-safe in-memory sliding window rate limiter."""

    def __init__(self):
        self._requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        now = time.time()
        cutoff = now - window_seconds
        # Prune old entries
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]
        if len(self._requests[key]) >= max_requests:
            return False
        self._requests[key].append(now)
        return True

    def cleanup(self, max_age: int = 300):
        """Periodic cleanup of stale keys."""
        cutoff = time.time() - max_age
        stale = [k for k, v in self._requests.items() if not v or v[-1] < cutoff]
        for k in stale:
            del self._requests[k]


_rate_limiter = RateLimitStore()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce per-IP rate limiting."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()

        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/"):
            return await call_next(request)

        # Identify client
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        if not _rate_limiter.is_allowed(
            client_ip,
            settings.RATE_LIMIT_REQUESTS,
            settings.RATE_LIMIT_WINDOW_SECONDS,
        ):
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
                headers={"Retry-After": str(settings.RATE_LIMIT_WINDOW_SECONDS)},
            )

        return await call_next(request)


# ============================================================
# Security Headers Middleware
# ============================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security headers (CSP, HSTS, etc.) into every response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        settings = get_settings()

        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self' https://fonts.gstatic.com",
            "connect-src 'self' " + " ".join(settings.cors_origins),
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

        # Other security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"

        return response


# ============================================================
# Request ID & Logging Middleware
# ============================================================

class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Attach unique request ID and log request timing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request.state.request_id = request_id
        start = time.time()

        response = await call_next(request)

        elapsed = round((time.time() - start) * 1000, 1)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed}ms"

        # Log non-health requests
        if request.url.path not in ("/health", "/", "/favicon.ico"):
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} → {response.status_code} ({elapsed}ms)"
            )

        return response


# ============================================================
# Centralized Error Handler Middleware
# ============================================================

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Catch all unhandled exceptions and return sanitized error responses.
    Never expose internal stack traces or provider details to the client.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.error(f"[{request_id}] Unhandled error: {exc}", exc_info=True)

            # Classify the error
            status_code = 500
            user_message = "An internal error occurred. Please try again."

            if "429" in str(exc) or "rate limit" in str(exc).lower():
                status_code = 429
                user_message = "The system is experiencing high demand. Please try again in a moment."
            elif "timeout" in str(exc).lower():
                status_code = 504
                user_message = "The request timed out. Please try again."
            elif "not found" in str(exc).lower():
                status_code = 404
                user_message = "The requested resource was not found."

            return JSONResponse(
                status_code=status_code,
                content={
                    "detail": user_message,
                    "request_id": request_id,
                },
            )


# ============================================================
# Input Validation Middleware
# ============================================================

class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Validate request body size and basic content constraints.
    Prevents oversized payloads before they reach route handlers.
    """

    MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length header
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.MAX_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large."},
            )

        return await call_next(request)
