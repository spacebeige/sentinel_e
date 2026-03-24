/**
 * ============================================================
 * Auth Context Hook
 * ============================================================
 *
 * Global authentication state management
 * Usage: const { user, role, signIn, signOut } = useAuthContext()
 */

import { createContext, useContext, useEffect, useState } from 'react';
import { getCurrentUser, signOutUser, USER_ROLES } from '../services/firebaseAuth';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [role, setRole] = useState(null);
  const [loading, setLoading] = useState(true);

  // Check auth status on mount and listen for changes
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const currentUser = await getCurrentUser();
        if (currentUser) {
          setUser(currentUser);
          setRole(currentUser.role || USER_ROLES.USER);
        }
      } catch (error) {
        console.error('Error checking auth status:', error);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []);

  const handleSignOut = async () => {
    try {
      const result = await signOutUser();
      if (result.success) {
        setUser(null);
        setRole(null);
      }
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  const handleLoginSuccess = (userData) => {
    setUser(userData);
    setRole(userData.role || USER_ROLES.USER);
  };

  const isAdmin = role === USER_ROLES.ADMIN;
  const isUser = role === USER_ROLES.USER;

  return (
    <AuthContext.Provider
      value={{
        user,
        role,
        loading,
        isAdmin,
        isUser,
        signOut: handleSignOut,
        onLoginSuccess: handleLoginSuccess,
      }}
    >
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
