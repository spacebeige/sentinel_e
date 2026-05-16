import { useCallback, useEffect, useState } from 'react';
import { isSupabaseConfigured } from '../lib/supabase';
import {
  restoreSupabaseSession,
  signInWithGitHubOAuth,
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

  useEffect(() => {
    let cancelled = false;

    async function hydrate() {
      if (!isSupabaseConfigured) {
        if (!cancelled) {
          setLoading(false);
          setError('Supabase auth environment variables are missing.');
        }
        return;
      }

      try {
        const restored = await restoreSupabaseSession();
        if (!cancelled) {
          setSession(restored.session || null);
          setUser(restored.user || null);
          setError(restored.error?.message || '');
        }
      } catch (restoreError) {
        if (!cancelled) {
          setError(restoreError?.message || 'Failed to restore auth session.');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    hydrate();

    const unsubscribe = subscribeToSupabaseSession(({ session: nextSession, user: nextUser }) => {
      if (cancelled) return;
      setSession(nextSession || null);
      setUser(nextUser || null);
      setLoading(false);
      setError('');
    });

    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);

  const signInWithGitHub = useCallback(async ({ redirectTo } = {}) => {
    setError('');
    await signInWithGitHubOAuth({ redirectTo });
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
    signInWithGitHub,
    signInWithEmail,
    signUpWithEmail,
    signOut,
  };
}

export default useSupabaseAuth;
