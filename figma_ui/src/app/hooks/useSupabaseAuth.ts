import { useCallback, useState } from 'react';
import { supabase, isSupabaseConfigured } from '../lib/supabase';
import { useAuthContext } from '../providers/AuthProvider';

export function useSupabaseAuth() {
  const { session, user, loading, isAuthenticated, isAdmin, isOwner, role } = useAuthContext();
  const [error, setError] = useState('');

  const signInWithGoogle = useCallback(async ({ redirectTo }: { redirectTo?: string } = {}) => {
    setError('');
    const { error: signInError } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: redirectTo ? { redirectTo } : undefined,
    });
    if (signInError) {
      setError(signInError.message);
      throw signInError;
    }
  }, []);

  const signInWithEmail = useCallback(async (email: string, password: string) => {
    setError('');
    const { data, error: signInError } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    if (signInError) {
      setError(signInError.message);
      throw signInError;
    }
    return data;
  }, []);

  const signUpWithEmail = useCallback(async (email: string, password: string, name?: string) => {
    setError('');
    const { data, error: signUpError } = await supabase.auth.signUp({
      email,
      password,
      options: name ? { data: { full_name: name } } : undefined,
    });
    if (signUpError) {
      setError(signUpError.message);
      throw signUpError;
    }
    return data;
  }, []);

  const signOut = useCallback(async () => {
    setError('');
    const { error: signOutError } = await supabase.auth.signOut();
    if (signOutError) {
      setError(signOutError.message);
      throw signOutError;
    }
  }, []);

  const resetPasswordForEmail = useCallback(async (email: string, redirectTo?: string) => {
    setError('');
    const { data, error: resetError } = await supabase.auth.resetPasswordForEmail(
      email,
      redirectTo ? { redirectTo } : undefined
    );
    if (resetError) {
      setError(resetError.message);
      throw resetError;
    }
    return data;
  }, []);

  const updateUserPassword = useCallback(async (password: string) => {
    setError('');
    const { data, error: updateError } = await supabase.auth.updateUser({ password });
    if (updateError) {
      setError(updateError.message);
      throw updateError;
    }
    return data;
  }, []);

  return {
    session,
    user,
    loading,
    isAuthenticated,
    isAdmin,
    isOwner,
    role,
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
