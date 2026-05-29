import { useCallback, useEffect, useRef, useState } from 'react';
import { isSupabaseConfigured } from '../lib/supabase';
import {
  restoreSupabaseSession,
  readSupabaseSessionSnapshot,
  signInWithGoogleOAuth,
  signInWithEmail as signInWithEmailService,
  signUpWithEmail as signUpWithEmailService,
  signOutSupabaseSession,
  subscribeToSupabaseSession,
  resetPasswordForEmail as resetPasswordForEmailService,
  updateUserPassword as updateUserPasswordService,
} from '../services/supabaseSessionManager';

export function useSupabaseAuth() {
  const [session, setSession] = useState<any>(null);
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    let unsubscribe = () => {};
    console.log("[AUTH] useEffect mounted");

    // ============================================================
    // 1. SYNCHRONOUS OPTIMISTIC HYDRATION
    // ============================================================
    // We do this immediately upon effect execution, outside the async function,
    // to unblock the UI instantly if a valid snapshot exists.
    const snapshot = readSupabaseSessionSnapshot();
    if (snapshot?.user?.id) {
      console.time("OPTIMISTIC_HYDRATION");
      setUser(snapshot.user);
      setSession(snapshot.access_token ? { access_token: snapshot.access_token, user: snapshot.user } : null);
      setLoading(false); 
      console.timeEnd("OPTIMISTIC_HYDRATION");
      console.log('[Auth] Optimistic hydration applied. UI unblocked.');
    }

    async function hydrate() {
      console.log("[AUTH] hydrate start. cancelled:", cancelled);
      console.time("AUTH_TOTAL");

      if (!isSupabaseConfigured) {
        if (!cancelled) {
          console.log("[AUTH] !isSupabaseConfigured, setLoading(false)");
          setLoading(false);
          setError('Supabase auth environment variables are missing.');
        }
        return;
      }

      console.log('[Auth] Hydrating Supabase session...');

      try {
        console.time("GET_SESSION");
        const restored = await restoreSupabaseSession();
        console.timeEnd("GET_SESSION");
        console.log("[AUTH] restore complete. cancelled:", cancelled);
        
        if (!cancelled) {
          setSession(restored.session || null);
          setUser(restored.user || null);
          setError(restored.error?.message || '');
          if (restored.session?.user?.id) {
            console.log('[Auth] Session restored for user:', restored.session.user.id);
          } else {
            console.log('[Auth] No active session found during hydration.');
          }
        } else {
          console.log("[AUTH] cancelled = true, skipped state update after restore");
        }
      } catch (restoreError) {
        if (!cancelled) {
          console.error('[Auth] Session restore failed:', restoreError?.message);
          setError(restoreError?.message || 'Failed to restore auth session.');
        }
      } finally {
        if (!cancelled) {
          console.log("[AUTH] setLoading(false)");
          setLoading(false);
          console.log('[Auth] Hydration complete. loading=false');
        } else {
          console.log("[AUTH] setLoading(false) skipped due to cancelled = true!");
        }
        console.timeEnd("AUTH_TOTAL");
      }

      if (!cancelled) {
        unsubscribe = subscribeToSupabaseSession(({ session: nextSession, user: nextUser }: any) => {
          if (cancelled) return;
          console.time("AUTH_STATE_CHANGE");
          console.log('[Auth] Auth state changed:', nextSession?.user?.id || 'signed out');
          setSession(nextSession || null);
          setUser(nextUser || null);
          console.log("[AUTH] setLoading(false) from subscribe");
          setLoading(false);
          setError('');
          console.timeEnd("AUTH_STATE_CHANGE");
        });
      }
    }

    hydrate();

    return () => {
      console.log("[AUTH] cleanup, setting cancelled = true");
      cancelled = true;
      unsubscribe();
    };
  }, []); // intentionally empty — runs once on mount only

  const signInWithGoogle = useCallback(async ({ redirectTo }: { redirectTo?: string } = {}) => {
    setError('');
    await signInWithGoogleOAuth({ redirectTo });
  }, []);

  const signInWithEmail = useCallback(async (email: string, password: string) => {
    setError('');
    return await signInWithEmailService(email, password);
  }, []);

  const signUpWithEmail = useCallback(async (email: string, password: string, name?: string) => {
    setError('');
    return await signUpWithEmailService(email, password, name);
  }, []);

  const signOut = useCallback(async () => {
    setError('');
    await signOutSupabaseSession();
  }, []);

  const resetPasswordForEmail = useCallback(async (email: string) => {
    setError('');
    return await resetPasswordForEmailService(email);
  }, []);

  const updateUserPassword = useCallback(async (password: string) => {
    setError('');
    return await updateUserPasswordService(password);
  }, []);

  return {
    session,
    user,
    loading,
    error,
    setError,
    isSupabaseConfigured,
    signInWithGoogle,
    signInWithEmail,
    signUpWithEmail,
    signOut,
    resetPasswordForEmail,
    updateUserPassword,
  };
}

export default useSupabaseAuth;
