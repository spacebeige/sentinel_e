/**
 * ============================================================
 * authGuards.js — Defensive Auth Priority Utilities
 * ============================================================
 *
 * These utilities enforce the production auth architecture rule:
 *   Authenticated Supabase users ALWAYS take priority.
 *   No guest access. Authentication is mandatory.
 *
 * Usage:
 *   import { isAuthenticatedUser, blockGuestHydration } from '../utils/authGuards';
 */



/**
 * Returns true if the given user object is a real authenticated user.
 *
 * @param {object|null} user
 * @returns {boolean}
 */
export function isAuthenticatedUser(user) {
  return !!user?.id;
}

// ── Persistence Guards ────────────────────────────────────────

/**
 * resolveProductionUserId — Returns the authenticated user ID, or null.
 *
 * @param {string|null} supabaseUserId - From Supabase session
 * @returns {string|null}
 */
export function resolveProductionUserId(supabaseUserId) {
  return supabaseUserId || null;
}

/**
 * buildUserScopedKey — Returns a user-scoped storage key.
 *
 * Production storage keys always follow: {prefix}:${userId}
 *
 * @param {string} prefix - Key prefix (e.g. 'conversation', 'session')
 * @param {string} userId - Authenticated Supabase user ID
 * @returns {string|null}
 */
export function buildUserScopedKey(prefix, userId) {
  if (!userId) {
    console.warn('[Sentinel Auth Guard] Attempted to build storage key with null userId.');
    return null;
  }
  return `${prefix}:${userId}`;
}

/**
 * assertAuthenticatedUser — Throws if the user is not authenticated.
 *
 * Use this at the start of any function that should only run for authenticated users.
 *
 * @param {object|null} user - User object from auth context
 * @param {string} context - Caller context for error message
 */
export function assertAuthenticatedUser(user, context = 'unknown') {
  if (!isAuthenticatedUser(user)) {
    throw new Error(
      `[Sentinel Auth Guard] Unauthenticated access attempt in: ${context}. ` +
      'This operation requires a valid Supabase session.'
    );
  }
}
