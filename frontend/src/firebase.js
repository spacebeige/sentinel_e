/**
 * ============================================================
 * Firebase Initialization — Auth & Identity
 * ============================================================
 * 
 * Purpose:
 *   • Initialize Firebase app with configuration from environment
 *   • Export auth instance for use across frontend
 *   • Ensure single initialization (prevent duplicates)
 * 
 * Usage:
 *   import { auth } from './firebase';
 *   
 *   // Get current user
 *   if (auth.currentUser) {
 *     console.log(auth.currentUser.uid);
 *   }
 *   
 *   // Get ID token for backend requests
 *   const token = await auth.currentUser.getIdToken();
 */

import { initializeApp } from 'firebase/app';
import { getAuth, setPersistence, browserLocalPersistence } from 'firebase/auth';

// Firebase configuration from environment variables
const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.REACT_APP_FIREBASE_APP_ID,
  measurementId: process.env.REACT_APP_FIREBASE_MEASUREMENT_ID,
};

// Validate that Firebase config is available
const isConfigValid = Object.values(firebaseConfig).every(value => value);
if (!isConfigValid) {
  console.warn('⚠️  Firebase config incomplete. Check .env.local:', {
    hasApiKey: !!firebaseConfig.apiKey,
    hasAuthDomain: !!firebaseConfig.authDomain,
    hasProjectId: !!firebaseConfig.projectId,
    hasStorageBucket: !!firebaseConfig.storageBucket,
    hasMessagingSenderId: !!firebaseConfig.messagingSenderId,
    hasAppId: !!firebaseConfig.appId,
  });
}

// Initialize Firebase app
const app = initializeApp(firebaseConfig);

// Get Firebase Auth instance
export const auth = getAuth(app);

// Set persistence to LOCAL so user stays logged in across page refreshes
setPersistence(auth, browserLocalPersistence).catch((error) => {
  console.error('Failed to set auth persistence:', error);
});

console.log('✓ Firebase initialized successfully');

export default app;
