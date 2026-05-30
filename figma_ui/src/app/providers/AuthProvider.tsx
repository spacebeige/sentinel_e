import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { supabase, isSupabaseConfigured } from '../lib/supabase';

interface AuthContextType {
  session: any;
  user: any;
  loading: boolean;
  isAuthenticated: boolean;
  isAdmin: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [session, setSession] = useState<any>(null);
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    let unsubscribe = () => {};

    const initialize = async () => {
      if (!isSupabaseConfigured) {
        if (isMounted) {
          setSession(null);
          setUser(null);
          setLoading(false);
        }
        return;
      }

      try {
        const { data } = await supabase.auth.getSession();
        if (!isMounted) return;
        setSession(data?.session ?? null);
        setUser(data?.session?.user ?? null);
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }

      const { data: authListener } = supabase.auth.onAuthStateChange((_event, nextSession) => {
        if (!isMounted) return;
        setSession(nextSession ?? null);
        setUser(nextSession?.user ?? null);
        setLoading(false);
      });

      unsubscribe = () => authListener?.subscription?.unsubscribe();
    };

    void initialize();

    return () => {
      isMounted = false;
      unsubscribe();
    };
  }, []);

  const adminAllowlist = useMemo(() => {
    const raw = (import.meta.env.VITE_RUNTIME_ADMIN_EMAILS || '') as string;
    return raw
      .split(',')
      .map((entry) => entry.trim().toLowerCase())
      .filter(Boolean);
  }, []);

  const isAuthenticated = Boolean(session?.user?.id);
  const isAdmin = Boolean(
    session?.user?.email &&
    adminAllowlist.includes(session.user.email.toLowerCase())
  );

  const value = useMemo(() => ({
    session,
    user,
    loading,
    isAuthenticated,
    isAdmin,
  }), [session, user, loading, isAuthenticated, isAdmin]);

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
