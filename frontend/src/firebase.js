/**
 * ============================================================
 * Firebase Initialization — Temporarily Disabled
 * ============================================================
 *
 * Guest mode keeps the app running without Firebase Auth while
 * preserving the original initialization logic for restoration.
 */

import { getGuestSessionId } from './services/guestSession';

export const GUEST_MODE_ENV_KEY = 'REACT_APP_GUEST_MODE';
const guestModeRaw = String(process.env.REACT_APP_GUEST_MODE ?? 'true').trim().toLowerCase();
export const TEMP_AUTH_DISABLED = guestModeRaw === 'true';

// TODO: Re-enable live Firebase authentication after auth configuration fixes
// TODO: Replace guest-session persistence with Firebase-auth session persistence later
export const GUEST_USER = Object.freeze({
  uid: 'guest-session',
  email: 'guest@sentinel.local',
  displayName: 'Guest',
  isGuest: true,
  emailVerified: true,
  providerId: 'guest',
  getIdToken: async () => null,
});

export function createGuestIdentity(guestSessionId = null) {
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
    if (!TEMP_AUTH_DISABLED) return null;
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
