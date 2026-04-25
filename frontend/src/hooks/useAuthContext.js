/**
 * ============================================================
 * Auth Context Hook (Clerk Edition)
 * ============================================================
 *
 * Global auth state using Clerk + backend user profile sync.
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
import { useUser, useAuth, useClerk } from '@clerk/clerk-react';
import api from '../services/api';

const AuthContext = createContext(null);

export let getClerkToken = async () => null;

export const AuthProvider = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();
  
  const { isLoaded, isSignedIn, user: clerkUser } = useUser();
  const { getToken, signOut: clerkSignOut } = useAuth();
  const clerk = useClerk();

  const [loading, setLoading] = useState(true);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authIntent, setAuthIntent] = useState('/chat');
  const [authError, setAuthError] = useState('');
  const [syncedUser, setSyncedUser] = useState(null);

  // Expose getToken globally for API interceptors
  getClerkToken = async () => {
    try {
      return await getToken();
    } catch (e) {
      return null;
    }
  };

  const refreshUser = useCallback(async () => {
    if (!isSignedIn || !clerkUser) {
      setSyncedUser(null);
      setLoading(false);
      return null;
    }

    try {
      const email = clerkUser.primaryEmailAddress?.emailAddress;
      const name = clerkUser.fullName || email?.split('@')[0];
      
      const payload = {
        email,
        name,
        provider: 'clerk'
      };

      // Ensure backend syncs this user
      const response = await api.post('/api/auth/sync-user', payload);
      setSyncedUser(response.data);
      return response.data;
    } catch (error) {
      if (process.env.NODE_ENV === 'development') {
        console.error('Auth sync failed:', error);
      }
      setSyncedUser(null);
      return null;
    } finally {
      setLoading(false);
    }
  }, [isSignedIn, clerkUser]);

  useEffect(() => {
    if (isLoaded) {
      refreshUser();
    }
  }, [isLoaded, refreshUser]);

  const openAuthModal = useCallback((options = {}) => {
    const nextPath = options.returnTo || options.redirectTo || location.pathname || '/chat';
    setAuthIntent(nextPath);
    setAuthError(options.error || '');
    clerk.openSignIn({ redirectUrl: nextPath });
  }, [clerk, location.pathname]);

  const closeAuthModal = useCallback(() => {
    setAuthModalOpen(false);
    setAuthError('');
  }, []);

  const handleLoginSuccess = useCallback((userData) => {
    setSyncedUser(userData);
    setAuthModalOpen(false);
    setAuthError('');

    const destination = authIntent || '/chat';
    if (location.pathname !== destination) {
      navigate(destination);
    }
  }, [authIntent, location.pathname, navigate]);

  const signOut = useCallback(async () => {
    await clerkSignOut();
    setSyncedUser(null);
    setAuthModalOpen(false);
    setAuthError('');
    navigate('/');
  }, [clerkSignOut, navigate]);

  const requireAuth = useCallback((options = {}) => {
    if (isSignedIn) {
      return true;
    }
    openAuthModal(options);
    return false;
  }, [openAuthModal, isSignedIn]);

  const value = useMemo(() => {
    // If we haven't synced yet but Clerk says we're signed in, provide a temporary user object
    const effectiveUser = syncedUser || (isSignedIn && clerkUser ? {
      user_id: clerkUser.id,
      email: clerkUser.primaryEmailAddress?.emailAddress,
      name: clerkUser.fullName,
      role: 'user',
      provider: 'clerk'
    } : null);

    const role = effectiveUser?.role || null;
    
    return {
      user: effectiveUser,
      role,
      loading: !isLoaded || (isSignedIn && !syncedUser && loading),
      authModalOpen,
      authIntent,
      authError,
      isAuthenticated: isSignedIn,
      isAdmin: role === 'admin',
      isUser: role === 'user',
      openAuthModal,
      closeAuthModal,
      refreshUser,
      requireAuth,
      setAuthError,
      signOut,
      onLoginSuccess: handleLoginSuccess,
    };
  }, [
    syncedUser,
    isSignedIn,
    clerkUser,
    isLoaded,
    loading,
    authModalOpen,
    authIntent,
    authError,
    openAuthModal,
    closeAuthModal,
    refreshUser,
    requireAuth,
    signOut,
    handleLoginSuccess,
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
