import React, { createContext, useContext, useMemo } from 'react';
import { useSupabaseAuth } from '@hooks/useSupabaseAuth';

interface AuthContextType {
  session: any;
  user: any;
  profile: any;
  role: 'user' | 'admin' | 'owner' | null;
  loading: boolean;
  isAuthenticated: boolean;
  signInWithGoogle: (options?: { redirectTo?: string }) => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { session, user, loading, signInWithGoogle, signOut } = useSupabaseAuth();

  const isAuthenticated = Boolean(session?.user?.id);

  const value = useMemo(() => ({
    session,
    user,
    profile: user,
    role: user?.user_metadata?.role || 'user',
    loading,
    isAuthenticated,
    signInWithGoogle,
    signOut
  }), [session, user, loading, isAuthenticated, signInWithGoogle, signOut]);

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
