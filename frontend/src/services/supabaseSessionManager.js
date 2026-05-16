import { getSupabaseClient, isSupabaseConfigured } from '../lib/supabase';

const AUTH_SNAPSHOT_KEY = 'sentinel-supabase-auth-snapshot';

function safeParse(raw, fallback = null) {
  if (!raw) return fallback;
  try {
    const parsed = JSON.parse(raw);
    return parsed ?? fallback;
  } catch {
    return fallback;
  }
}

function canUseLocalStorage() {
  try {
    if (typeof window === 'undefined' || !window.localStorage) return false;
    const key = '__sentinel_supabase_test__';
    window.localStorage.setItem(key, key);
    window.localStorage.removeItem(key);
    return true;
  } catch {
    return false;
  }
}

export function normalizeSupabaseUser(user) {
  if (!user?.id) return null;

  const role = user?.app_metadata?.role || user?.user_metadata?.role || 'user';
  const provider = user?.app_metadata?.provider
    || user?.identities?.[0]?.provider
    || 'supabase';

  return {
    id: user.id,
    user_id: user.id,
    uid: user.id,
    email: user.email || '',
    name: user.user_metadata?.full_name
      || user.user_metadata?.name
      || user.user_metadata?.user_name
      || user.email?.split('@')[0]
      || 'User',
    provider,
    role,
    is_authenticated: true,
    app_metadata: user.app_metadata || {},
    user_metadata: user.user_metadata || {},
  };
}

export function readSupabaseSessionSnapshot() {
  if (!canUseLocalStorage()) return null;
  return safeParse(window.localStorage.getItem(AUTH_SNAPSHOT_KEY), null);
}

export function persistSupabaseSessionSnapshot(session) {
  if (!canUseLocalStorage()) return;

  try {
    if (!session?.user?.id) {
      window.localStorage.removeItem(AUTH_SNAPSHOT_KEY);
      return;
    }

    const normalizedUser = normalizeSupabaseUser(session.user);
    window.localStorage.setItem(AUTH_SNAPSHOT_KEY, JSON.stringify({
      access_token: session.access_token || null,
      expires_at: session.expires_at || null,
      user: normalizedUser,
      refreshedAt: new Date().toISOString(),
    }));
  } catch (error) {
    console.warn('Unable to persist Supabase auth snapshot', error);
  }
}

export function clearSupabaseSessionSnapshot() {
  if (!canUseLocalStorage()) return;
  window.localStorage.removeItem(AUTH_SNAPSHOT_KEY);
}

export async function restoreSupabaseSession() {
  if (!isSupabaseConfigured) {
    return { session: null, user: null, error: null };
  }

  const supabase = getSupabaseClient();
  const { data, error } = await supabase.auth.getSession();
  const session = data?.session || null;
  const user = normalizeSupabaseUser(session?.user || null);

  if (session) {
    persistSupabaseSessionSnapshot(session);
  } else {
    clearSupabaseSessionSnapshot();
  }

  return { session, user, error: error || null };
}

export async function signInWithGitHubOAuth({ redirectTo } = {}) {
  if (!isSupabaseConfigured) {
    throw new Error('Supabase auth is not configured.');
  }

  const supabase = getSupabaseClient();
  const resolvedRedirect = redirectTo || `${window.location.origin}/chat`;
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'github',
    options: {
      redirectTo: resolvedRedirect,
    },
  });

  if (error) {
    throw error;
  }

  return data;
}

export async function signOutSupabaseSession() {
  if (!isSupabaseConfigured) {
    return;
  }

  const supabase = getSupabaseClient();
  const { error } = await supabase.auth.signOut();
  if (error) {
    throw error;
  }
  clearSupabaseSessionSnapshot();
}

export function subscribeToSupabaseSession(listener) {
  if (!isSupabaseConfigured) {
    return () => {};
  }

  const supabase = getSupabaseClient();
  const { data } = supabase.auth.onAuthStateChange((_event, session) => {
    persistSupabaseSessionSnapshot(session);
    const user = normalizeSupabaseUser(session?.user || null);
    listener({ session: session || null, user });
  });

  return () => data?.subscription?.unsubscribe?.();
}

export async function signInWithEmail({ email, password }) {
  if (!isSupabaseConfigured) {
    throw new Error('Supabase auth is not configured.');
  }

  const supabase = getSupabaseClient();
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });

  if (error) {
    throw error;
  }

  return data;
}

export async function signUpWithEmail({ email, password, options = {} }) {
  if (!isSupabaseConfigured) {
    throw new Error('Supabase auth is not configured.');
  }

  const supabase = getSupabaseClient();
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options,
  });

  if (error) {
    throw error;
  }

  return data;
}
