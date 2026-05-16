/**
 * ============================================================
 * Auth Context Hook — Supabase Production Auth
 * ============================================================
 *
 * PRODUCTION BEHAVIOR:
 *   - Authentication runs exclusively through Supabase.
 *   - Authenticated users always own persistence and session state.
 *   - Guest fallback is completely hidden and NEVER exposed in UI.
 *   - isGuestMode is always false in production paths.
 *
 * HIDDEN GUEST FALLBACK (dev/emergency only):
 *   - Only activates when HIDDEN_GUEST_FALLBACK_ENABLED === true
 *     AND Supabase is fully unavailable (not configured or errored).
 *   - Must not override authenticated user state.
 *   - Must not appear in routing, session selection, or conversation history.
 *
 * TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { createGuestIdentity, HIDDEN_GUEST_FALLBACK_ENABLED } from '../firebase';
import useSupabaseAuth from './useSupabaseAuth';
import {
  clearPersistenceUser,
  restoreGuestSession,
  setPersistenceUser,
} from '../services/sessionPersistence';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const {
    user: supabaseUser,
    session,
    loading,
    error: supabaseError,
    setError: setSupabaseError,
    signInWithGitHub,
    signInWithEmail,
    signUpWithEmail,
    signOut: signOutSupabase,
    isSupabaseConfigured,
  } = useSupabaseAuth();

  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authIntent, setAuthIntent] = useState('/chat');
  const [authError, setAuthError] = useState('');
  // ── Hidden guest fallback (dev/emergency only) ───────────────
  // restoreGuestSession is only called when HIDDEN_GUEST_FALLBACK_ENABLED is true.
  // This value is always null in production (REACT_APP_GUEST_MODE=false).
  // TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
  const guestSession = useMemo(
    () => (HIDDEN_GUEST_FALLBACK_ENABLED ? restoreGuestSession() : null),
    []
  );
  const guestSessionId = guestSession?.guestSessionId || null;
  // createGuestIdentity returns null in production (guarded in firebase.js)
  const guestUser = useMemo(
    () => (guestSessionId ? createGuestIdentity(guestSessionId) : null),
    [guestSessionId]
  );

  // TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
  // Guest fallback ONLY activates when:
  //   1. HIDDEN_GUEST_FALLBACK_ENABLED is explicitly true (dev env flag + non-production)
  //   2. Supabase is fully unavailable (not configured OR errored)
  //   3. No authenticated Supabase user exists
  // Authenticated users ALWAYS take priority — guest NEVER overwrites real user state.
  const shouldUseGuestFallback = Boolean(
    HIDDEN_GUEST_FALLBACK_ENABLED
      && guestUser?.id           // guest identity must exist (null in production)
      && !supabaseUser?.id       // no authenticated user
      && (!isSupabaseConfigured || Boolean(supabaseError))  // auth system down
  );

  // ── Persistence ownership ────────────────────────────────
  // Authenticated Supabase users ALWAYS own persistence.
  // Guest state NEVER overwrites a real user's persistence key.
  useEffect(() => {
    // Only set persistence for authenticated users.
    // Guest fallback persistence is only written when shouldUseGuestFallback is true,
    // which requires Supabase to be fully unavailable.
    if (supabaseUser?.id) {
      setPersistenceUser(supabaseUser.id);
    } else if (shouldUseGuestFallback && guestUser?.id) {
      // TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
      setPersistenceUser(guestUser.id);
    }
  }, [guestUser?.id, shouldUseGuestFallback, supabaseUser?.id]);

  useEffect(() => {
    if (supabaseError) {
      setAuthError(supabaseError);
    }
  }, [supabaseError]);

  // ── Resolved user (production: always Supabase user or null) ──────
  const syncedUser = supabaseUser?.id ? supabaseUser : (shouldUseGuestFallback ? guestUser : null);
  const isAuthenticated = Boolean(
    (session?.user?.id && supabaseUser?.id)       // production: Supabase session
      || (shouldUseGuestFallback && guestUser?.id) // dev fallback: guest (gated)
  );
  const isAdmin = Boolean(supabaseUser?.id && syncedUser?.role === 'admin');
  // isGuestMode is always false — guest mode is hidden and never exposed to UI
  const isGuestMode = false;

  const openAuthModal = useCallback((options = {}) => {
    setAuthIntent(options.returnTo || options.redirectTo || location.pathname || '/chat');
    setAuthError(options.error || '');
    setAuthModalOpen(true);
  }, [location.pathname]);

  const closeAuthModal = useCallback(() => {
    setAuthModalOpen(false);
    setAuthError('');
    setSupabaseError('');
  }, [setSupabaseError]);

  const buildOAuthRedirect = useCallback((destination) => {
    if (typeof window === 'undefined') {
      return destination || '/chat';
    }
    const safePath = destination && destination.startsWith('/') ? destination : '/chat';
    return `${window.location.origin}${safePath}`;
  }, []);

  const handleSignIn = useCallback(async (options = {}) => {
    if (!isSupabaseConfigured) {
      if (shouldUseGuestFallback) {
        return guestUser;
      }
      throw new Error('Supabase auth environment variables are missing.');
    }

    const returnTo = options.returnTo || authIntent || location.pathname || '/chat';
    const redirectTo = buildOAuthRedirect(returnTo);
    await signInWithGitHub({ redirectTo });
    return null;
  }, [authIntent, buildOAuthRedirect, guestUser, isSupabaseConfigured, location.pathname, shouldUseGuestFallback, signInWithGitHub]);

  const handleEmailSignIn = useCallback(async ({ email, password }) => {
    if (!isSupabaseConfigured) {
      throw new Error('Supabase auth environment variables are missing.');
    }
    try {
      const data = await signInWithEmail({ email, password });
      return data;
    } catch (error) {
      setAuthError(error.message || 'Login failed');
      throw error;
    }
  }, [isSupabaseConfigured, signInWithEmail]);

  const handleEmailSignUp = useCallback(async ({ email, password, options }) => {
    if (!isSupabaseConfigured) {
      throw new Error('Supabase auth environment variables are missing.');
    }
    try {
      const data = await signUpWithEmail({ email, password, options });
      return data;
    } catch (error) {
      setAuthError(error.message || 'Sign up failed');
      throw error;
    }
  }, [isSupabaseConfigured, signUpWithEmail]);

  const handleLoginSuccess = useCallback(() => {
    const destination = authIntent || '/chat';
    navigate(destination, { replace: true });
  }, [authIntent, navigate]);

  const signOut = useCallback(async () => {
    try {
      await signOutSupabase();
    } finally {
      clearPersistenceUser();
      closeAuthModal();
      navigate('/', { replace: true });
    }
  }, [closeAuthModal, navigate, signOutSupabase]);

  const requireAuth = useCallback((options = {}) => {
    if (isAuthenticated) {
      return true;
    }
    openAuthModal(options);
    return false;
  }, [isAuthenticated, openAuthModal]);

  // ── Defensive hydration guards ───────────────────────────
  // authResolved: auth check is complete (not still loading). Use this to gate
  //   any UI that should only render after we know the auth state.
  // sessionReady: user is authenticated AND auth is resolved. Gate history/chat loads here.
  // isHydrated: true once localStorage rehydration finishes. Use in useStore checks.
  const authResolved = !loading;
  const sessionReady = authResolved && isAuthenticated && Boolean(syncedUser?.id);

  const value = useMemo(() => ({
    user: syncedUser,
    role: syncedUser?.role || 'user',
    loading,
    // Hydration guards — use these to prevent blank UI / race-condition resets
    authResolved,   // auth check complete (loading=false)
    sessionReady,   // authenticated + auth resolved
    isHydrated: authResolved, // alias for consumers that prefer this name
    authModalOpen,
    authIntent,
    authError,
    isAuthenticated,
    isAdmin,
    isUser: true,
    isGuestMode,
    isSupabaseConfigured,
    openAuthModal,
    closeAuthModal,
    handleSignIn,
    handleEmailSignIn,
    handleEmailSignUp,
    requireAuth,
    setAuthError,
    signOut,
    onLoginSuccess: handleLoginSuccess,
  }), [
    syncedUser,
    loading,
    authResolved,
    sessionReady,
    authModalOpen,
    authIntent,
    authError,
    isAuthenticated,
    isAdmin,
    isGuestMode,
    isSupabaseConfigured,
    openAuthModal,
    closeAuthModal,
    handleSignIn,
    handleEmailSignIn,
    handleEmailSignUp,
    requireAuth,
    handleLoginSuccess,
    signOut,
  ]);

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuthContext = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuthContext must be used within AuthProvider');
  }
  return context;
};

// TODO: Restore Firebase Auth after configuration fixes
// TODO: Remove or fully restore Firebase auth after Supabase migration stabilizes
// Original Firebase auth context preserved below.
//
// import {
//   signInWithEmailAndPassword,
//   createUserWithEmailAndPassword,
//   signOut as firebaseSignOut,
//   onAuthStateChanged,
//   setPersistence,
//   browserLocalPersistence,
// } from 'firebase/auth';
// import { auth } from '../firebase';
//
// export const AuthProvider = ({ children }) => {
//   const location = useLocation();
//   const navigate = useNavigate();
//
//   const [firebaseUser, setFirebaseUser] = useState(null);
//   const [loading, setLoading] = useState(true);
//   const [authModalOpen, setAuthModalOpen] = useState(false);
//   const [authIntent, setAuthIntent] = useState('/chat');
//   const [authError, setAuthError] = useState('');
//   const [syncedUser, setSyncedUser] = useState(null);
//
//   // Listen to Firebase auth state changes
//   useEffect(() => {
//     const unsubscribe = onAuthStateChanged(auth, (user) => {
//       setFirebaseUser(user);
//       setLoading(false);
//
//       if (user) {
//         console.log('✓ Firebase user authenticated:', {
//           uid: user.uid,
//           email: user.email,
//           displayName: user.displayName,
//           emailVerified: user.emailVerified,
//         });
//
//         setSyncedUser({
//           user_id: user.uid,
//           email: user.email || '',
//           name: user.displayName || user.email?.split('@')[0] || 'User',
//           provider: 'firebase',
//           role: 'user'
//         });
//       } else {
//         console.log('User signed out');
//         setSyncedUser(null);
//       }
//     });
//
//     return unsubscribe;
//   }, []);
//
//   const handleSignIn = useCallback(async (email, password) => {
//     try {
//       await setPersistence(auth, browserLocalPersistence);
//       await signInWithEmailAndPassword(auth, email, password);
//     } catch (error) {
//       console.error('Sign in failed:', error);
//       setAuthError(error.message || 'Failed to sign in');
//       throw error;
//     }
//   }, []);
//
//   const handleSignUp = useCallback(async (email, password, displayName) => {
//     try {
//       await setPersistence(auth, browserLocalPersistence);
//       const result = await createUserWithEmailAndPassword(auth, email, password);
//       if (displayName && result.user) {
//         // Note: updateProfile is async but we'll let it happen in background
//       }
//     } catch (error) {
//       console.error('Sign up failed:', error);
//       setAuthError(error.message || 'Failed to sign up');
//       throw error;
//     }
//   }, []);
//
//   const openAuthModal = useCallback((options = {}) => {
//     const nextPath = options.returnTo || options.redirectTo || location.pathname || '/chat';
//     setAuthIntent(nextPath);
//     setAuthError(options.error || '');
//     setAuthModalOpen(true);
//   }, [location.pathname]);
//
//   const closeAuthModal = useCallback(() => {
//     setAuthModalOpen(false);
//     setAuthError('');
//   }, []);
//
//   const handleLoginSuccess = useCallback((userData) => {
//     setSyncedUser(userData);
//     setAuthModalOpen(false);
//     setAuthError('');
//
//     const destination = authIntent || '/chat';
//     if (location.pathname !== destination) {
//       navigate(destination);
//     }
//   }, [authIntent, location.pathname, navigate]);
//
//   const signOut = useCallback(async () => {
//     try {
//       await firebaseSignOut(auth);
//       setSyncedUser(null);
//       setAuthModalOpen(false);
//       setAuthError('');
//       navigate('/');
//     } catch (error) {
//       console.error('Sign out failed:', error);
//     }
//   }, [navigate]);
//
//   const requireAuth = useCallback((options = {}) => {
//     if (firebaseUser) {
//       return true;
//     }
//     openAuthModal(options);
//     return false;
//   }, [openAuthModal, firebaseUser]);
// };

export default AuthContext;
