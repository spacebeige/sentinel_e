/**
 * ============================================================
 * Auth Context Hook — Temporary Guest Mode
 * ============================================================
 *
 * Firebase Auth is intentionally bypassed so the app can run in
 * unauthenticated development mode without deleting auth code.
 */

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { createGuestIdentity, TEMP_AUTH_DISABLED } from '../firebase';
import { restoreGuestSession } from '../services/guestSession';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const location = useLocation();
  const navigate = useNavigate();

  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authIntent, setAuthIntent] = useState('/chat');
  const [authError, setAuthError] = useState('');

  const loading = false;
  const guestSession = useMemo(() => restoreGuestSession(), []);
  const guestSessionId = guestSession?.guestSessionId || 'guest-user';
  // TODO: Replace guest-session persistence with Firebase-auth session persistence later
  const GUEST_SESSION_USER = useMemo(
    () => createGuestIdentity(guestSessionId),
    [guestSessionId]
  );
  const syncedUser = GUEST_SESSION_USER;
  const isGuestMode = TEMP_AUTH_DISABLED;
  const isAuthenticated = TEMP_AUTH_DISABLED;
  const isAdmin = TEMP_AUTH_DISABLED ? true : syncedUser?.role === 'admin';

  const openAuthModal = useCallback((options = {}) => {
    if (!TEMP_AUTH_DISABLED) {
      setAuthIntent(options.returnTo || options.redirectTo || location.pathname || '/chat');
      setAuthError(options.error || '');
      setAuthModalOpen(true);
      return;
    }

    // TODO: Re-enable live Firebase authentication after auth configuration fixes
    console.info('Guest mode active; auth modal suppressed.', {
      returnTo: options.returnTo || options.redirectTo || location.pathname || '/chat',
    });
  }, [location.pathname]);

  const closeAuthModal = useCallback(() => {
    setAuthModalOpen(false);
    setAuthError('');
  }, []);

  const handleSignIn = useCallback(async (..._args) => {
    if (!TEMP_AUTH_DISABLED) {
      throw new Error('Live Firebase authentication is temporarily disabled. Enable REACT_APP_GUEST_MODE=true.');
    }

    // TODO: Re-enable live Firebase authentication after auth configuration fixes
    console.info('Guest mode active; sign-in request bypassed.');
    return syncedUser;
  }, [syncedUser]);

  const handleSignUp = useCallback(async (..._args) => {
    if (!TEMP_AUTH_DISABLED) {
      throw new Error('Live Firebase authentication is temporarily disabled. Enable REACT_APP_GUEST_MODE=true.');
    }

    // TODO: Re-enable live Firebase authentication after auth configuration fixes
    console.info('Guest mode active; sign-up request bypassed.');
    return syncedUser;
  }, [syncedUser]);

  const handleLoginSuccess = useCallback(() => {
    const destination = location.pathname || '/chat';
    navigate(destination, { replace: true });
  }, [location.pathname, navigate]);

  const signOut = useCallback(async () => {
    // TODO: Re-enable live Firebase authentication after auth configuration fixes
    // TODO: Replace guest-session persistence with Firebase-auth session persistence later
    if (TEMP_AUTH_DISABLED) {
      console.info('Guest mode active; Firebase sign-out disabled. Local guest history is preserved.');
    }
    navigate('/', { replace: true });
  }, [navigate]);

  const requireAuth = useCallback((options = {}) => {
    if (TEMP_AUTH_DISABLED) {
      return true;
    }
    openAuthModal(options);
    return false;
  }, [openAuthModal]);

  const value = useMemo(() => ({
    user: syncedUser,
    role: syncedUser.role,
    loading,
    authModalOpen,
    authIntent,
    authError,
    isAuthenticated,
    isAdmin,
    isUser: true,
    isGuestMode,
    openAuthModal,
    closeAuthModal,
    handleSignIn,
    handleSignUp,
    requireAuth,
    setAuthError,
    signOut,
    onLoginSuccess: handleLoginSuccess,
  }), [
    syncedUser,
    loading,
    authModalOpen,
    authIntent,
    authError,
    isAuthenticated,
    isAdmin,
    isGuestMode,
    openAuthModal,
    closeAuthModal,
    handleSignIn,
    handleSignUp,
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
