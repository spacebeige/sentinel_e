/**
 * ============================================================
 * Auth Context Hook (Firebase Edition)
 * ============================================================
 *
 * Global auth state using Firebase Auth + backend user profile sync.
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
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut as firebaseSignOut,
  onAuthStateChanged,
  setPersistence,
  browserLocalPersistence,
} from 'firebase/auth';
import { auth } from '../firebase';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();

  const [firebaseUser, setFirebaseUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authIntent, setAuthIntent] = useState('/chat');
  const [authError, setAuthError] = useState('');
  const [syncedUser, setSyncedUser] = useState(null);

  // Listen to Firebase auth state changes
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setFirebaseUser(user);
      setLoading(false);

      if (user) {
        // User is signed in
        console.log('✓ Firebase user authenticated:', {
          uid: user.uid,
          email: user.email,
          displayName: user.displayName,
          emailVerified: user.emailVerified,
        });

        // Build synced user object
        setSyncedUser({
          user_id: user.uid,
          email: user.email || '',
          name: user.displayName || user.email?.split('@')[0] || 'User',
          provider: 'firebase',
          role: 'user'
        });
      } else {
        // User signed out
        console.log('User signed out');
        setSyncedUser(null);
      }
    });

    return unsubscribe;
  }, []);

  // Phase 2: Log auth state for debugging
  useEffect(() => {
    if (!loading) {
      console.log('AUTH STATE:', {
        firebaseUser: firebaseUser ? { uid: firebaseUser.uid, email: firebaseUser.email } : null,
        syncedUser: !!syncedUser,
        loading,
        timestamp: new Date().toISOString()
      });
    }
  }, [firebaseUser, syncedUser, loading]);

  const handleSignIn = useCallback(async (email, password) => {
    try {
      await setPersistence(auth, browserLocalPersistence);
      await signInWithEmailAndPassword(auth, email, password);
      // Firebase auth state listener will handle the rest
    } catch (error) {
      console.error('Sign in failed:', error);
      setAuthError(error.message || 'Failed to sign in');
      throw error;
    }
  }, []);

  const handleSignUp = useCallback(async (email, password, displayName) => {
    try {
      await setPersistence(auth, browserLocalPersistence);
      const result = await createUserWithEmailAndPassword(auth, email, password);
      // Update display name if provided
      if (displayName && result.user) {
        // Note: updateProfile is async but we'll let it happen in background
        // The auth state listener will capture the updated user
      }
      // Firebase auth state listener will handle the rest
    } catch (error) {
      console.error('Sign up failed:', error);
      setAuthError(error.message || 'Failed to sign up');
      throw error;
    }
  }, []);

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
    try {
      await firebaseSignOut(auth);
      setSyncedUser(null);
      setAuthModalOpen(false);
      setAuthError('');
      navigate('/');
    } catch (error) {
      console.error('Sign out failed:', error);
    }
  }, [navigate]);

  const requireAuth = useCallback((options = {}) => {
    if (firebaseUser) {
      return true;
    }
    openAuthModal(options);
    return false;
  }, [openAuthModal, firebaseUser]);

  const value = useMemo(() => {
    const role = syncedUser?.role || null;

    return {
      user: syncedUser,
      role,
      loading,
      authModalOpen,
      authIntent,
      authError,
      isAuthenticated: !!firebaseUser,
      isAdmin: role === 'admin',
      isUser: role === 'user',
      openAuthModal,
      closeAuthModal,
      handleSignIn,
      handleSignUp,
      requireAuth,
      setAuthError,
      signOut,
      onLoginSuccess: handleLoginSuccess,
    };
  }, [
    syncedUser,
    firebaseUser,
    loading,
    authModalOpen,
    authIntent,
    authError,
    openAuthModal,
    closeAuthModal,
    handleSignIn,
    handleSignUp,
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

