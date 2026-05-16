/**
 * ============================================================
 * firebase.js — Hidden Guest Fallback Architecture
 * ============================================================
 *
 * PRODUCTION BEHAVIOR:
 *   - HIDDEN_GUEST_FALLBACK_ENABLED = false (always)
 *   - Guest identity is NEVER created or injected
 *   - Authenticated Supabase users own ALL persistence
 *
 * DEV/EMERGENCY FALLBACK (hidden, internal only):
 *   - Requires REACT_APP_GUEST_MODE=true in .env.local
 *   - Requires NODE_ENV !== 'production'
 *   - Used ONLY for offline debugging / auth-system-down scenarios
 *   - Must NOT appear in UI, routing, or session selection
 *
 * TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
 */

import { getGuestSessionId } from './services/guestSession';

// ── Environment guards (production = always false) ──────────
export const GUEST_MODE_ENV_KEY = 'REACT_APP_GUEST_MODE';
const guestModeRaw = String(process.env.REACT_APP_GUEST_MODE ?? 'false').trim().toLowerCase();

// NEVER true in production. Requires explicit dev flag + non-production env.
// TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
export const TEMP_AUTH_DISABLED = guestModeRaw === 'true' && process.env.NODE_ENV !== 'production';
export const HIDDEN_GUEST_FALLBACK_ENABLED = TEMP_AUTH_DISABLED;

// ── Hidden guest identity (dev/emergency only) ───────────────
// This object is NEVER exposed to production users.
// It only activates when HIDDEN_GUEST_FALLBACK_ENABLED === true.
export const GUEST_USER = Object.freeze({
  uid: 'guest-session',
  email: 'guest@sentinel.local',
  displayName: 'Guest',
  isGuest: true,
  emailVerified: false,
  providerId: 'guest',
  getIdToken: async () => null,
});

/**
 * createGuestIdentity — DEV/EMERGENCY ONLY
 * Returns a temporary identity for offline debugging or auth-system-down fallback.
 * MUST NOT be called when a real Supabase user session is active.
 * TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
 */
export function createGuestIdentity(guestSessionId = null) {
  if (!HIDDEN_GUEST_FALLBACK_ENABLED) {
    // Defensive: never create a guest identity in production
    return null;
  }
  const resolvedSessionId = guestSessionId || getGuestSessionId() || 'guest-user';
  return {
    ...GUEST_USER,
    id: resolvedSessionId,
    user_id: resolvedSessionId,
    guestSessionId: resolvedSessionId,
    provider: 'guest',
    role: 'admin',
  };
}

export const auth = {
  get currentUser() {
    if (!HIDDEN_GUEST_FALLBACK_ENABLED) return null;
    return createGuestIdentity();
  },
};

export const app = null;

// TODO: Re-enable live Firebase authentication after auth configuration fixes
// Original Firebase initialization preserved below.
//
// import { initializeApp } from 'firebase/app';
// import { getAuth, setPersistence, browserLocalPersistence } from 'firebase/auth';
//
// // Firebase configuration from environment variables
// const firebaseConfig = {
//   apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
//   authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
//   projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
//   storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
//   messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
//   appId: process.env.REACT_APP_FIREBASE_APP_ID,
//   measurementId: process.env.REACT_APP_FIREBASE_MEASUREMENT_ID,
// };
//
// // Validate that Firebase config is available
// const isConfigValid = Object.values(firebaseConfig).every(value => value);
// if (!isConfigValid) {
//   console.warn('⚠️  Firebase config incomplete. Check .env.local:', {
//     hasApiKey: !!firebaseConfig.apiKey,
//     hasAuthDomain: !!firebaseConfig.authDomain,
//     hasProjectId: !!firebaseConfig.projectId,
//     hasStorageBucket: !!firebaseConfig.storageBucket,
//     hasMessagingSenderId: !!firebaseConfig.messagingSenderId,
//     hasAppId: !!firebaseConfig.appId,
//   });
// }
//
// // Initialize Firebase app
// const app = initializeApp(firebaseConfig);
//
// // Get Firebase Auth instance
// export const auth = getAuth(app);
//
// // Set persistence to LOCAL so user stays logged in across page refreshes
// setPersistence(auth, browserLocalPersistence).catch((error) => {
//   console.error('Failed to set auth persistence:', error);
// });
//
// console.log('✓ Firebase initialized successfully');
//
// export default app;
export default app;
