/**
 * ============================================================
 * guestSession.js — Hidden Dev/Emergency Fallback Utilities
 * ============================================================
 *
 * PRODUCTION: This module has NO effect.
 *   - HIDDEN_GUEST_FALLBACK_ENABLED = false in production
 *   - All exports are no-ops or return null/empty values
 *   - This file is NOT removed to preserve emergency fallback capability
 *
 * DEV/EMERGENCY ONLY: Activates when REACT_APP_GUEST_MODE=true
 *   - Used for offline debugging, auth-system-down scenarios
 *   - Local testing without Supabase credentials
 *   - Emergency degraded mode
 *
 * TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
 */

export {
  createGuestSession,
  restoreGuestSession,
  getGuestSessionId,
  saveConversationState,
  loadConversationState,
  saveConversationHistory,
  loadConversationHistory,
  switchConversation,
  createNewConversation,
  persistSessionState,
  listConversationHistory,
  isGuestHydrating,
} from './sessionPersistence';
