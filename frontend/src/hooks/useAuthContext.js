/**
 * ============================================================
 * Auth Context Hook — Supabase Production Auth
 * ============================================================
 *
 * PRODUCTION BEHAVIOR:
 *   - Authentication runs exclusively through Supabase.
 *   - Authenticated users always own persistence and session state.
 *   - Guest fallback is disabled across all environments.
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
import useSupabaseAuth from './useSupabaseAuth';
import {
  clearPersistenceUser,
  setPersistenceUser,
} from '../services/sessionPersistence';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  console.log("ACTIVE_RUNTIME:AuthContext");
  const location = useLocation();
  const navigate = useNavigate();
  const {
    user: supabaseUser,
    session,
    loading,
    error: supabaseError,
    setError: setSupabaseError,
    signInWithGoogle,
    signOut: signOutSupabase,
    isSupabaseConfigured,
  } = useSupabaseAuth();

  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authIntent, setAuthIntent] = useState('/chat');
  const [authError, setAuthError] = useState('');
  // ── Persistence ownership ────────────────────────────────
  // Authenticated Supabase users ALWAYS own persistence.
  useEffect(() => {
    console.log("AUTH_USER", supabaseUser);
    console.log("AUTH_SESSION", session);
    if (supabaseUser?.id) {
      setPersistenceUser(supabaseUser.id);
    }
  }, [supabaseUser, supabaseUser?.id, session]);

  useEffect(() => {
    if (supabaseError) {
      setAuthError(supabaseError);
    }
  }, [supabaseError]);

  // ── Resolved user (production: always Supabase user or null) ──────
  const syncedUser = supabaseUser?.id ? supabaseUser : null;
  const isAuthenticated = Boolean(
    (session?.user?.id && supabaseUser?.id)       // production: Supabase session
  );
  const isAdmin = Boolean(supabaseUser?.id && syncedUser?.role === 'admin');

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
      throw new Error('Supabase auth environment variables are missing.');
    }

    const returnTo = options.returnTo || authIntent || location.pathname || '/chat';
    const redirectTo = buildOAuthRedirect(returnTo);
    await signInWithGoogle({ redirectTo });
    return null;
  }, [authIntent, buildOAuthRedirect, isSupabaseConfigured, location.pathname, signInWithGoogle]);

  const handleLoginSuccess = useCallback(() => {
    const destination = authIntent || '/chat';
    navigate(destination, { replace: true });
  }, [authIntent, navigate]);

  const signOut = useCallback(async () => {
    try {
      await signOutSupabase();
    } finally {
      clearPersistenceUser();
      if (typeof window !== 'undefined') {
        if (window.localStorage) localStorage.clear();
        if (window.sessionStorage) sessionStorage.clear();
        // Clear IndexedDB for Supabase if it exists
        try {
          const dbs = await window.indexedDB.databases();
          dbs.forEach((db) => {
            if (db.name.includes('supabase')) {
              window.indexedDB.deleteDatabase(db.name);
            }
          });
        } catch (e) {
          // Ignore indexedDB errors in restricted environments
        }
      }
      
      const { default: useStore } = await import('../stores/useStore');
      useStore.getState().reset();
      
      closeAuthModal();
      navigate('/', { replace: true });
      window.location.reload();
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
    isSupabaseConfigured,
    openAuthModal,
    closeAuthModal,
    handleSignIn,
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
    isSupabaseConfigured,
    openAuthModal,
    closeAuthModal,
    handleSignIn,
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

export default AuthContext;
