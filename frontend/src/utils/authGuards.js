/**
 * ============================================================
 * authGuards.js — Defensive Auth Priority Utilities
 * ============================================================
 *
 * These utilities enforce the production auth architecture rule:
 *   Authenticated Supabase users ALWAYS take priority.
 *   Guest state NEVER overwrites real user state.
 *   Guest session keys NEVER appear in authenticated user persistence.
 *
 * Usage:
 *   import { isAuthenticatedUser, blockGuestHydration } from '../utils/authGuards';
 *
 * TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
 */

// ── Constants ─────────────────────────────────────────────────
const GUEST_USER_PREFIX = 'guest';
const GUEST_FALLBACK_ID = 'guest-session';
const GUEST_DEV_ID = 'guest-dev-user';

// ── Environment Guard ──────────────────────────────────────────
// In production, REACT_APP_GUEST_MODE is always 'false'.
// TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
export const GUEST_MODE_ACTIVE =
  String(process.env.REACT_APP_GUEST_MODE ?? 'false').trim().toLowerCase() === 'true' &&
  process.env.NODE_ENV !== 'production';

// ── Type Guards ───────────────────────────────────────────────

/**
 * Returns true if the given userId belongs to a guest session.
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
 * blockGuestHydration — Prevents guest-state contamination on rehydrate.
 *
 * Call this when restoring state from localStorage.
 * If the stored userId is a guest ID and guest mode is not enabled,
 * returns a cleared state to prevent guest data from persisting.
 *
 * @param {object} restoredState - The state object from localStorage
 * @returns {object} - Sanitized state (clears guest data if detected in production)
 */
export function blockGuestHydration(restoredState) {
  if (!restoredState) return restoredState;

  const storedUserId = restoredState?.userId;

  if (!GUEST_MODE_ACTIVE && isGuestUserId(storedUserId)) {
    console.warn(
      '[Sentinel Auth Guard] Blocked guest-state hydration into production store.',
      'Guest ID detected:', storedUserId,
      '— Clearing stale guest session.'
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
 * Authenticated users always win. Guest IDs are rejected in production.
 *
 * @param {string|null} supabaseUserId - From Supabase session
 * @param {string|null} guestUserId - From hidden guest fallback
 * @returns {string|null}
 */
export function resolveProductionUserId(supabaseUserId, guestUserId = null) {
  // Authenticated user always takes priority
  if (supabaseUserId && !isGuestUserId(supabaseUserId)) {
    return supabaseUserId;
  }

  // TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
  // Guest fallback: only if explicitly enabled in dev environment
  if (GUEST_MODE_ACTIVE && guestUserId && isGuestUserId(guestUserId)) {
    return guestUserId;
  }

  return null;
}

/**
 * buildUserScopedKey — Returns a user-scoped storage key.
 *
 * Production storage keys always follow: conversation:${user.id}
 * NEVER use 'guest-session' as a key for authenticated users.
 *
 * @param {string} prefix - Key prefix (e.g. 'conversation', 'session')
 * @param {string} userId - Authenticated user ID
 * @returns {string|null}
 */
export function buildUserScopedKey(prefix, userId) {
  if (!userId || isGuestUserId(userId)) {
    // TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
    if (!GUEST_MODE_ACTIVE) {
      console.warn('[Sentinel Auth Guard] Attempted to build storage key with guest/null userId.');
      return null;
    }
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
