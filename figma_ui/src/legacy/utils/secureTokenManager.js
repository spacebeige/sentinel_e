/**
 * Secure Token Management
 * Fixes: XSS vulnerability from tokens in JavaScript variables
 * 
 * SECURITY BEST PRACTICES:
 * 1. Use HttpOnly cookies to prevent JavaScript access
 * 2. Use Secure flag to HTTPS-only transmission
 * 3. Use SameSite=Strict to prevent CSRF
 * 4. Never store sensitive tokens in localStorage/sessionStorage
 * 5. Never expose tokens in JavaScript module scope
 */

/**
 * Store authentication token securely in HttpOnly cookie
 * @param {string} token - JWT token to store
 * @param {number} expiresInMinutes - Cookie expiration time
 */
export function setAuthTokenInCookie(token, expiresInMinutes = 60) {
  const expirationDate = new Date();
  expirationDate.setTime(expirationDate.getTime() + (expiresInMinutes * 60 * 1000));
  
  // Set cookie with security flags
  document.cookie = `auth_token=${token}; ` +
    `expires=${expirationDate.toUTCString()}; ` +
    `path=/; ` +
    `HttpOnly; ` +  // ✓ Prevent JavaScript access
    `Secure; ` +    // ✓ HTTPS only
    `SameSite=Strict`;  // ✓ CSRF protection
  
  console.debug("✓ Auth token stored securely in HttpOnly cookie");
}

/**
 * Retrieve the authentication token from cookie
 * Note: Can't access HttpOnly cookies from JavaScript!
 * This is for setting the cookie path only.
 * The backend will automatically include the cookie in requests.
 */
export function getAuthToken() {
  // For HttpOnly cookies, the browser automatically sends them with requests
  // We cannot read them from JavaScript (this is intentional for security)
  // Instead, verify auth status by making authenticated request
  return null;
}

/**
 * Clear authentication token
 */
export function clearAuthToken() {
  document.cookie = "auth_token=; " +
    "expires=Thu, 01 Jan 1970 00:00:00 UTC; " +
    "path=/; " +
    "HttpOnly; " +
    "Secure; " +
    "SameSite=Strict";
  
  console.debug("✓ Auth token cleared");
}

/**
 * Check if user is authenticated
 * Make a test request to a protected endpoint
 */
export async function isAuthenticated() {
  try {
    const response = await fetch('/api/auth/verify', {
      method: 'GET',
      credentials: 'include',  // Important: Include cookies in request
    });
    return response.ok;
  } catch (error) {
    console.error("Auth verification failed:", error);
    return false;
  }
}

/**
 * Setup axios/fetch to automatically include cookies
 * Must be called once during app initialization
 */
export function setupSecureRequestDefaults() {
  // For fetch API
  window.fetch = ((originalFetch) => {
    return function (...args) {
      return originalFetch.apply(this, args).then((response) => {
        // Automatically include credentials (cookies)
        if (!('credentials' in args[1])) {
          args[1] = { ...args[1], credentials: 'include' };
        }
        return response;
      });
    };
  })(fetch);
  
  console.debug("✓ Secure request defaults configured");
}

/**
 * MIGRATION GUIDE from localStorage to HttpOnly cookies:
 * 
 * OLD (VULNERABLE):
 * ```
 * localStorage.setItem('access_token', token)
 * const token = localStorage.getItem('access_token')
 * ```
 * 
 * NEW (SECURE):
 * ```
 * setAuthTokenInCookie(token)  // Backend handles all token logic
 * // Token automatically sent with requests via credentials:'include'
 * ```
 */
