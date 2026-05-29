import React, { createContext, useContext, useEffect, useMemo, useState, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router';
import useSupabaseAuth from '../hooks/useSupabaseAuth';
import { clearPersistenceUser, setPersistenceUser } from '../services/sessionPersistence';

interface AuthContextType {
  user: any;
  role: string;
  loading: boolean;
  authResolved: boolean;
  sessionReady: boolean;
  isHydrated: boolean;
  authModalOpen: boolean;
  authIntent: string;
  authError: string;
  isAuthenticated: boolean;
  isAdmin: boolean;
  isUser: boolean;
  isSupabaseConfigured: boolean;
  openAuthModal: (options?: { returnTo?: string; redirectTo?: string; error?: string }) => void;
  closeAuthModal: () => void;
  handleSignIn: (options?: { returnTo?: string }) => Promise<void>;
  requireAuth: (options?: { returnTo?: string }) => boolean;
  setAuthError: (error: string) => void;
  signOut: () => Promise<void>;
  onLoginSuccess: () => void;
  signInWithGoogleOAuth: (options?: { redirectTo?: string }) => Promise<void>;
  signInWithEmail: (email: string, password: string) => Promise<any>;
  signUpWithEmail: (email: string, password: string, name?: string) => Promise<any>;
  resetPasswordForEmail: (email: string) => Promise<any>;
  updateUserPassword: (password: string) => Promise<any>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
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
    signInWithEmail,
    signUpWithEmail,
    resetPasswordForEmail,
    updateUserPassword,
  } = useSupabaseAuth();

  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authIntent, setAuthIntent] = useState('/chat');
  const [authError, setAuthError] = useState('');

  useEffect(() => {
    if (supabaseUser?.id) {
      setPersistenceUser(supabaseUser.id);
    }
  }, [supabaseUser, supabaseUser?.id, session]);

  useEffect(() => {
    if (supabaseError) {
      setAuthError(supabaseError);
    }
  }, [supabaseError]);

  useEffect(() => {
    console.log("[AUTH_PROVIDER]");
    console.log("session", session);
    console.log("user", supabaseUser);
    const isAuth = Boolean(session?.user?.id && supabaseUser?.id);
    console.log("auth", isAuth);
  }, [session, supabaseUser]);

  const syncedUser = supabaseUser?.id ? supabaseUser : null;
  const isAuthenticated = Boolean(session?.user?.id && supabaseUser?.id);
  const isAdmin = Boolean(supabaseUser?.id && syncedUser?.role === 'admin');

  const openAuthModal = useCallback((options: { returnTo?: string; redirectTo?: string; error?: string } = {}) => {
    setAuthIntent(options.returnTo || options.redirectTo || location.pathname || '/chat');
    setAuthError(options.error || '');
    setAuthModalOpen(true);
  }, [location.pathname]);

  const closeAuthModal = useCallback(() => {
    setAuthModalOpen(false);
    setAuthError('');
    setSupabaseError('');
  }, [setSupabaseError]);

  const buildOAuthRedirect = useCallback((destination?: string) => {
    if (typeof window === 'undefined') {
      return destination || '/chat';
    }
    const safePath = destination && destination.startsWith('/') ? destination : '/chat';
    return `${window.location.origin}${safePath}`;
  }, []);

  const handleSignIn = useCallback(async (options: { returnTo?: string } = {}) => {
    if (!isSupabaseConfigured) {
      throw new Error('Supabase auth environment variables are missing.');
    }
    const returnTo = options.returnTo || authIntent || location.pathname || '/chat';
    const redirectTo = buildOAuthRedirect(returnTo);
    await signInWithGoogle({ redirectTo });
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
        try {
          const dbs = await window.indexedDB.databases();
          dbs.forEach((db) => {
            if (db.name && db.name.includes('supabase')) {
              window.indexedDB.deleteDatabase(db.name);
            }
          });
        } catch (e) {
          // Ignore
        }
      }
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

  const authResolved = !loading;
  const sessionReady = authResolved && isAuthenticated && Boolean(syncedUser?.id);

  const value = useMemo(() => ({
    user: syncedUser,
    role: syncedUser?.role || 'user',
    loading,
    authResolved,
    sessionReady,
    isHydrated: authResolved,
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
    signInWithGoogleOAuth: signInWithGoogle,
    signInWithEmail,
    signUpWithEmail,
    resetPasswordForEmail,
    updateUserPassword,
  }), [
    syncedUser, loading, authResolved, sessionReady, authModalOpen, authIntent, authError,
    isAuthenticated, isAdmin, isSupabaseConfigured, openAuthModal, closeAuthModal, handleSignIn,
    requireAuth, handleLoginSuccess, signOut, signInWithGoogle, signInWithEmail, signUpWithEmail,
    resetPasswordForEmail, updateUserPassword,
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
