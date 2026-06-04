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

// ── Constants ─────────────────────────────────────────────────
const GUEST_USER_PREFIX = 'guest';
const GUEST_FALLBACK_ID = 'guest-session';
const GUEST_DEV_ID = 'guest-dev-user';

// ── Type Guards ───────────────────────────────────────────────

/**
 * Returns true if the given userId belongs to a stale guest session.
 * Guest IDs are NEVER valid for authenticated user persistence.
 *
 * @param {string|null} userId
 * @returns {boolean}
 */
export function isGuestUserId(userId) {
  if (!userId) return false;
  const normalized = String(userId).toLowerCase().trim();
  return (
    normalized === GUEST_FALLBACK_ID ||
    normalized === GUEST_DEV_ID ||
    normalized.startsWith(GUEST_USER_PREFIX)
  );
}

/**
 * Returns true if the given user object is a real authenticated user.
 * Requires a non-guest ID from a Supabase session.
 *
 * @param {object|null} user
 * @returns {boolean}
 */
export function isAuthenticatedUser(user) {
  if (!user?.id) return false;
  return !isGuestUserId(user.id);
}

// ── Persistence Guards ────────────────────────────────────────

/**
 * blockGuestHydration — Prevents stale guest-state contamination on rehydrate.
 *
 * Call this when restoring state from localStorage.
 * If the stored userId is a legacy guest ID, clears that state to prevent
 * it from bleeding into an authenticated user's session.
 *
 * @param {object} restoredState - The state object from localStorage
 * @returns {object} - Sanitized state
 */
export function blockGuestHydration(restoredState) {
  if (!restoredState) return restoredState;

  const storedUserId = restoredState?.userId;

  if (isGuestUserId(storedUserId)) {
    console.warn(
      '[Sentinel Auth Guard] Blocked stale guest-state hydration.',
      'Guest ID detected:', storedUserId,
      '— Clearing.'
    );
    return {
      ...restoredState,
      userId: null,
      chats: [],
      messages: [],
      memory: [],
      isLoaded: false,
    };
  }

  return restoredState;
}

/**
 * resolveProductionUserId — Returns the authenticated user ID, or null.
 *
 * Only accepts non-guest Supabase UUIDs.
 *
 * @param {string|null} supabaseUserId - From Supabase session
 * @returns {string|null}
 */
export function resolveProductionUserId(supabaseUserId) {
  if (supabaseUserId && !isGuestUserId(supabaseUserId)) {
    return supabaseUserId;
  }
  return null;
}

/**
 * buildUserScopedKey — Returns a user-scoped storage key.
 *
 * Production storage keys always follow: {prefix}:${userId}
 * NEVER accepts a guest userId.
 *
 * @param {string} prefix - Key prefix (e.g. 'conversation', 'session')
 * @param {string} userId - Authenticated Supabase user ID
 * @returns {string|null}
 */
export function buildUserScopedKey(prefix, userId) {
  if (!userId || isGuestUserId(userId)) {
    console.warn('[Sentinel Auth Guard] Attempted to build storage key with guest/null userId.');
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
