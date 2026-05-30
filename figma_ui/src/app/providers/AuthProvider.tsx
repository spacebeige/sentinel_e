import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { supabase, isSupabaseConfigured } from '../lib/supabase';
import { getCurrentUser } from '../api';

interface AuthContextType {
  session: any;
  user: any;
  profile: any;
  role: 'user' | 'admin' | 'owner' | null;
  loading: boolean;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [session, setSession] = useState<any>(null);
  const [user, setUser] = useState<any>(null);
  const [profile, setProfile] = useState<any>(null);
  const [role, setRole] = useState<'user' | 'admin' | 'owner' | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    let unsubscribe = () => { };

    const initialize = async () => {
      if (!isSupabaseConfigured) {
        if (isMounted) {
          setSession(null);
          setUser(null);
          setLoading(false);
        }
        return;
      }

      const fetchProfile = async (userId: string) => {
        try {
          const userData = await getCurrentUser();
          if (isMounted) {
            if (userData && userData.user) {
              setProfile(userData.user);
              setRole(userData.user.role || 'user');
            } else {
              setProfile({ id: userId });
              setRole('user');
            }
          }
        } catch (err) {
          console.error('[AUTH PROVIDER] Error fetching profile:', err);
          if (isMounted) {
            setRole('user');
          }
        }
      };

      try {
        const { data } = await supabase.auth.getSession();
        if (!isMounted) return;
        setSession(data?.session ?? null);
        setUser(data?.session?.user ?? null);
        if (data?.session?.user?.id) {
          await fetchProfile(data.session.user.id);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }

      const { data: authListener } = supabase.auth.onAuthStateChange(async (_event, nextSession) => {
        if (!isMounted) return;
        setSession(nextSession ?? null);
        setUser(nextSession?.user ?? null);
        
        if (nextSession?.user?.id) {
          await fetchProfile(nextSession.user.id);
        } else {
          setProfile(null);
          setRole(null);
        }
        
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

  const isAuthenticated = Boolean(session?.user?.id);

  const value = useMemo(() => ({
    session,
    user,
    profile,
    role,
    loading,
    isAuthenticated,
  }), [session, user, profile, role, loading, isAuthenticated]);

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
