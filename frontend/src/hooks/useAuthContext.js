/**
 * ============================================================
 * Auth Context Hook
 * ============================================================
 *
 * Global auth state for SuperTokens + backend user profile sync.
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
import {
  AUTH_REQUIRED_EVENT,
  AUTH_STATE_CHANGED_EVENT,
  USER_ROLES,
  SuperTokensWrapper,
  getCurrentUser,
  handleAuthCallbackIfPresent,
  signOutUser,
} from '../services/firebaseAuth';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authIntent, setAuthIntent] = useState('/chat');
  const [authError, setAuthError] = useState('');

  const refreshUser = useCallback(async () => {
    try {
      const currentUser = await getCurrentUser();
      setUser(currentUser);
      return currentUser;
    } catch (error) {
      if (process.env.NODE_ENV === 'development') {
        console.error('Auth refresh failed:', error);
      }
      setUser(null);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;

    const bootstrap = async () => {
      try {
        const handled = await handleAuthCallbackIfPresent();
        if (handled) return;
      } catch (error) {
        if (active) {
          setAuthError(error instanceof Error ? error.message : 'Authentication failed.');
          setAuthModalOpen(true);
        }
      }

      if (active) {
        await refreshUser();
      }
    };

    bootstrap();

    return () => {
      active = false;
    };
  }, [refreshUser]);

  const openAuthModal = useCallback((options = {}) => {
    const nextPath = options.returnTo || options.redirectTo || location.pathname || '/chat';
    setAuthIntent(nextPath);
    setAuthError(options.error || '');
    setAuthModalOpen(true);
  }, [location.pathname]);

  const closeAuthModal = useCallback(() => {
    setAuthModalOpen(false);
    setAuthError('');
  }, []);

  useEffect(() => {
    const onAuthRequired = (event) => {
      openAuthModal(event.detail || {});
    };

    const onAuthStateChanged = async () => {
      const nextUser = await refreshUser();
      if (nextUser) {
        setAuthModalOpen(false);
        setAuthError('');
      }
    };

    window.addEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
    window.addEventListener(AUTH_STATE_CHANGED_EVENT, onAuthStateChanged);

    return () => {
      window.removeEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
      window.removeEventListener(AUTH_STATE_CHANGED_EVENT, onAuthStateChanged);
    };
  }, [openAuthModal, refreshUser]);

  const handleLoginSuccess = useCallback((userData) => {
    setUser(userData);
    setAuthModalOpen(false);
    setAuthError('');

    const destination = authIntent || '/chat';
    if (location.pathname !== destination) {
      navigate(destination);
    }
  }, [authIntent, location.pathname, navigate]);

  const signOut = useCallback(async () => {
    await signOutUser();
    setUser(null);
    setAuthModalOpen(false);
    setAuthError('');
    navigate('/');
  }, [navigate]);

  const requireAuth = useCallback((options = {}) => {
    if (user) {
      return true;
    }
    openAuthModal(options);
    return false;
  }, [openAuthModal, user]);

  const value = useMemo(() => {
    const role = user?.role || null;
    return {
      user,
      role,
      loading,
      authModalOpen,
      authIntent,
      authError,
      isAuthenticated: Boolean(user),
      isAdmin: role === USER_ROLES.ADMIN,
      isUser: role === USER_ROLES.USER,
      openAuthModal,
      closeAuthModal,
      refreshUser,
      requireAuth,
      setAuthError,
      signOut,
      onLoginSuccess: handleLoginSuccess,
    };
  }, [
    authError,
    authIntent,
    authModalOpen,
    closeAuthModal,
    handleLoginSuccess,
    loading,
    openAuthModal,
    refreshUser,
    requireAuth,
    signOut,
    user,
  ]);

  return (
    <AuthContext.Provider value={value}>
      <SuperTokensWrapper>{children}</SuperTokensWrapper>
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
