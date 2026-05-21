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
} from '../services/supabaseSessionManager';

export function useSupabaseAuth() {
  const [session, setSession] = useState(null);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  // Guard: prevent duplicate hydration calls (React StrictMode / HMR double-invoke)
  const hydratedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    let unsubscribe = () => {};
    // Safety timeout: if session restore hangs for >5s (network, cold start),
    // force loading=false so the app doesn't stay blocked indefinitely.
    const safetyTimer = setTimeout(() => {
      if (!cancelled) {
        console.warn('[Auth] Session restore safety timeout hit — forcing loading=false');
        setLoading(false);
      }
    }, 5000);

    async function hydrate() {
      // Ensure hydration runs exactly once per mount
      if (hydratedRef.current) return;
      hydratedRef.current = true;

      if (!isSupabaseConfigured) {
        if (!cancelled) {
          setLoading(false);
          setError('Supabase auth environment variables are missing.');
        }
        return;
      }

      console.log('[Auth] Hydrating Supabase session...');

      try {
        const snapshot = readSupabaseSessionSnapshot();
        if (!cancelled && snapshot?.user?.id) {
          setUser(snapshot.user);
          setSession((prev) => prev || (snapshot.access_token ? { access_token: snapshot.access_token, user: snapshot.user } : null));
        }
        const restored = await restoreSupabaseSession();
        if (!cancelled) {
          setSession(restored.session || null);
          setUser(restored.user || null);
          setError(restored.error?.message || '');
          if (restored.session?.user?.id) {
            console.log('[Auth] Session restored for user:', restored.session.user.id);
          } else {
            console.log('[Auth] No active session found during hydration.');
          }
        }
      } catch (restoreError) {
        if (!cancelled) {
          console.error('[Auth] Session restore failed:', restoreError?.message);
          setError(restoreError?.message || 'Failed to restore auth session.');
        }
      } finally {
        clearTimeout(safetyTimer);
        if (!cancelled) {
          setLoading(false);
          console.log('[Auth] Hydration complete. loading=false');
        }
      }

      if (!cancelled) {
        unsubscribe = subscribeToSupabaseSession(({ session: nextSession, user: nextUser }) => {
          if (cancelled) return;
          console.log('[Auth] Auth state changed:', nextSession?.user?.id || 'signed out');
          setSession(nextSession || null);
          setUser(nextUser || null);
          setLoading(false);
          setError('');
        });
      }
    }

    hydrate();

    return () => {
      cancelled = true;
      clearTimeout(safetyTimer);
      unsubscribe();
    };
  }, []); // intentionally empty — runs once on mount only

  const signInWithGoogle = useCallback(async ({ redirectTo } = {}) => {
    setError('');
    await signInWithGoogleOAuth({ redirectTo });
  }, []);

  const signInWithEmail = useCallback(async ({ email, password }) => {
    setError('');
    return await signInWithEmailService({ email, password });
  }, []);

  const signUpWithEmail = useCallback(async ({ email, password, options }) => {
    setError('');
    return await signUpWithEmailService({ email, password, options });
  }, []);

  const signOut = useCallback(async () => {
    setError('');
    await signOutSupabaseSession();
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
  };
}

export default useSupabaseAuth;
