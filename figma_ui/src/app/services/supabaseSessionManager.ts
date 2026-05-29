// import { getSupabaseClient, isSupabaseConfigured } from '../lib/supabase';

// const AUTH_SNAPSHOT_KEY = 'sentinel-supabase-auth-snapshot';
// const RUNTIME_ADMIN_EMAILS = new Set(
//   String(process.env.REACT_APP_RUNTIME_ADMIN_EMAILS || 'oomkaragarkhed0710@gmail.com')
//     .split(',')
//     .map((email) => email.trim().toLowerCase())
//     .filter(Boolean)
// );

// function safeParse(raw, fallback = null) {
//   if (!raw) return fallback;
//   try {
//     const parsed = JSON.parse(raw);
//     return parsed ?? fallback;
//   } catch {
//     return fallback;
//   }
// }

// function canUseLocalStorage() {
//   try {
//     if (typeof window === 'undefined' || !window.localStorage) return false;
//     const key = '__sentinel_supabase_test__';
//     window.localStorage.setItem(key, key);
//     window.localStorage.removeItem(key);
//     return true;
//   } catch {
//     return false;
//   }
// }

// export function normalizeSupabaseUser(user) {
//   if (!user?.id) return null;

//   const email = user.email || '';
//   const role = RUNTIME_ADMIN_EMAILS.has(email.trim().toLowerCase())
//     ? 'admin'
//     : (user?.app_metadata?.role || user?.user_metadata?.role || 'user');
//   const provider = user?.app_metadata?.provider
//     || user?.identities?.[0]?.provider
//     || 'supabase';

//   return {
//     id: user.id,
//     user_id: user.id,
//     uid: user.id,
//     email,
//     name: user.user_metadata?.full_name
//       || user.user_metadata?.name
//       || user.user_metadata?.user_name
//       || user.email?.split('@')[0]
//       || 'User',
//     provider,
//     role,
//     is_authenticated: true,
//     app_metadata: user.app_metadata || {},
//     user_metadata: user.user_metadata || {},
//   };
// }

// export function readSupabaseSessionSnapshot() {
//   if (!canUseLocalStorage()) return null;
//   return safeParse(window.localStorage.getItem(AUTH_SNAPSHOT_KEY), null);
// }

// export function persistSupabaseSessionSnapshot(session) {
//   if (!canUseLocalStorage()) return;

//   try {
//     if (!session?.user?.id) {
//       window.localStorage.removeItem(AUTH_SNAPSHOT_KEY);
//       return;
//     }

//     const normalizedUser = normalizeSupabaseUser(session.user);
//     window.localStorage.setItem(AUTH_SNAPSHOT_KEY, JSON.stringify({
//       access_token: session.access_token || null,
//       expires_at: session.expires_at || null,
//       user: normalizedUser,
//       refreshedAt: new Date().toISOString(),
//     }));
//   } catch (error) {
//     console.warn('Unable to persist Supabase auth snapshot', error);
//   }
// }

// export function clearSupabaseSessionSnapshot() {
//   if (!canUseLocalStorage()) return;
//   window.localStorage.removeItem(AUTH_SNAPSHOT_KEY);
// }

// export async function restoreSupabaseSession() {
//   if (!isSupabaseConfigured) {
//     return { session: null, user: null, error: null };
//   }

//   const supabase = getSupabaseClient();
//   const { data, error } = await supabase.auth.getSession();
//   const session = data?.session || null;
//   const user = normalizeSupabaseUser(session?.user || null);

//   if (session) {
//     persistSupabaseSessionSnapshot(session);
//   } else {
//     clearSupabaseSessionSnapshot();
//   }

//   return { session, user, error: error || null };
// }

// export async function signInWithGoogleOAuth({ redirectTo } = {}) {
//   if (!isSupabaseConfigured) {
//     throw new Error('Supabase auth is not configured.');
//   }

//   const supabase = getSupabaseClient();
//   const resolvedRedirect = redirectTo || `${window.location.origin}/chat`;
//   const { data, error } = await supabase.auth.signInWithOAuth({
//     provider: 'google',
//     options: {
//       redirectTo: resolvedRedirect,
//     },
//   });

//   if (error) {
//     throw error;
//   }

//   return data;
// }

// export async function signOutSupabaseSession() {
//   if (!isSupabaseConfigured) {
//     return;
//   }

//   const supabase = getSupabaseClient();
//   const { error } = await supabase.auth.signOut();
//   if (error) {
//     throw error;
//   }
//   clearSupabaseSessionSnapshot();
// }

// export function subscribeToSupabaseSession(listener) {
//   if (!isSupabaseConfigured) {
//     return () => {};
//   }

//   const supabase = getSupabaseClient();
//   const { data } = supabase.auth.onAuthStateChange((_event, session) => {
//     persistSupabaseSessionSnapshot(session);
//     const user = normalizeSupabaseUser(session?.user || null);
//     listener({ session: session || null, user });
//   });

//   return () => data?.subscription?.unsubscribe?.();
// }

// export async function signInWithEmail({ email, password }) {
//   if (!isSupabaseConfigured) {
//     throw new Error('Supabase auth is not configured.');
//   }

//   // Validate before calling Supabase to prevent 422 unprocessable-entity errors
//   if (!email || typeof email !== 'string' || !email.includes('@')) {
//     throw new Error('A valid email address is required.');
//   }
//   if (!password || typeof password !== 'string' || password.length < 6) {
//     throw new Error('Password must be at least 6 characters.');
//   }

//   const supabase = getSupabaseClient();
//   const { data, error } = await supabase.auth.signInWithPassword({
//     email: email.trim().toLowerCase(),
//     password,
//   });

//   if (error) {
//     console.error('[Supabase] signInWithEmail error:', error.status, error.message);
//     throw error;
//   }

//   return data;
// }

// export async function signUpWithEmail({ email, password, options = {} }) {
//   if (!isSupabaseConfigured) {
//     throw new Error('Supabase auth is not configured.');
//   }

//   // Validate before calling Supabase to prevent 422 unprocessable-entity errors
//   if (!email || typeof email !== 'string' || !email.includes('@')) {
//     throw new Error('A valid email address is required.');
//   }
//   if (!password || typeof password !== 'string' || password.length < 6) {
//     throw new Error('Password must be at least 6 characters.');
//   }

//   const supabase = getSupabaseClient();
//   const { data, error } = await supabase.auth.signUp({
//     email: email.trim().toLowerCase(),
//     password,
//     options,
//   });

//   if (error) {
//     console.error('[Supabase] signUpWithEmail error:', error.status, error.message);
//     throw error;
//   }

//   // Detect duplicate signup (Supabase returns a fake user with no session)
//   if (data?.user && !data.session && data.user.identities?.length === 0) {
//     console.warn('[Supabase] signUpWithEmail: email already registered (no identities returned)');
//     throw new Error('An account with this email already exists. Please sign in instead.');
//   }

//   return data;
// }

import { getSupabaseClient, isSupabaseConfigured } from '../lib/supabase';

const AUTH_SNAPSHOT_KEY = 'sentinel-supabase-auth-snapshot-v2';

const RUNTIME_ADMIN_EMAILS = new Set(
  String(
    import.meta.env.VITE_RUNTIME_ADMIN_EMAILS ||
    'oomkaragarkhed0710@gmail.com'
  )
    .split(',')
    .map((email) => email.trim().toLowerCase())
    .filter(Boolean)
);

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
    if (typeof window === 'undefined' || !window.localStorage) {
      return false;
    }

    const key = '__sentinel_supabase_test__';

    window.localStorage.setItem(key, key);
    window.localStorage.removeItem(key);

    return true;
  } catch {
    return false;
  }
}

/**
 * ============================================================
 * HARD AUTH NORMALIZATION
 * ============================================================
 * Prevent:
 * - stale guest hydration
 * - ghost sessions
 * - invalid fallback users
 * - invalid fallback users
 * ============================================================
 */

export function normalizeSupabaseUser(user) {
  if (!user?.id) {
    return null;
  }

  const email = String(user.email || '').trim().toLowerCase();

  const derivedName =
    user?.user_metadata?.full_name ||
    user?.user_metadata?.name ||
    user?.user_metadata?.user_name ||
    user?.email?.split('@')[0] ||
    '';

  const normalizedName = String(derivedName).trim().toLowerCase();

  /**
   * ============================================================
   * HARD BLOCK LEGACY / INVALID USERS
   * ============================================================
   */

  const invalidGuestValues = new Set([
    'fallback',
    'local',
    'temp',
  ]);

  if (
    invalidGuestValues.has(email) ||
    invalidGuestValues.has(normalizedName) ||
    email.includes('guest') ||
    normalizedName.includes('guest')
  ) {
    console.warn(
      '[Sentinel-E Auth] Blocked invalid guest hydration:',
      {
        email,
        normalizedName,
      }
    );

    return null;
  }

  const role = RUNTIME_ADMIN_EMAILS.has(email)
    ? 'admin'
    : (
        user?.app_metadata?.role ||
        user?.user_metadata?.role ||
        'user'
      );

  const provider =
    user?.app_metadata?.provider ||
    user?.identities?.[0]?.provider ||
    'supabase';

  return {
    id: user.id,
    user_id: user.id,
    uid: user.id,

    email,

    name:
      derivedName ||
      email.split('@')[0] ||
      'User',

    provider,
    role,

    is_authenticated: true,

    app_metadata: user.app_metadata || {},
    user_metadata: user.user_metadata || {},
  };
}

export function readSupabaseSessionSnapshot() {
  if (!canUseLocalStorage()) {
    return null;
  }

  const parsed = safeParse(
    window.localStorage.getItem(AUTH_SNAPSHOT_KEY),
    null
  );
  
  console.log("[READ PAYLOAD]", parsed);

  /**
   * ============================================================
   * HARD VALIDATION
   * ============================================================
   */

  if (!parsed?.user?.id) {
    console.log("[DIAGNOSTIC] readSupabaseSessionSnapshot: parsed.user.id missing");
    return null;
  }

  if (
    parsed?.user?.email === 'fallback' ||
    parsed?.user?.name === 'fallback'
  ) {
    console.log("[DIAGNOSTIC] readSupabaseSessionSnapshot: fallback user detected, clearing");
    clearSupabaseSessionSnapshot();
    return null;
  }

  console.log("[DIAGNOSTIC] readSupabaseSessionSnapshot: successful read", parsed);
  return parsed;
}

export function persistSupabaseSessionSnapshot(session) {
  if (!canUseLocalStorage()) {
    return;
  }

  try {
    /**
     * ============================================================
     * INVALID SESSION
     * ============================================================
     */

    if (!session?.user?.id) {
      clearSupabaseSessionSnapshot();
      return;
    }

    const normalizedUser = normalizeSupabaseUser(session.user);

    /**
     * ============================================================
     * INVALID USER
     * ============================================================
     */

    if (!normalizedUser?.id) {
      console.log("[DIAGNOSTIC] persistSupabaseSessionSnapshot: invalid normalized user, clearing snapshot");
      clearSupabaseSessionSnapshot();
      return;
    }

    console.log("[DIAGNOSTIC] SAVING SESSION", session);
    const payloadToSave = {
      access_token: session.access_token || null,
      expires_at: session.expires_at || null,
      user: normalizedUser,
      refreshedAt: new Date().toISOString(),
    };
    console.log("[SAVE PAYLOAD]", payloadToSave);

    window.localStorage.setItem(
      AUTH_SNAPSHOT_KEY,
      JSON.stringify(payloadToSave)
    );
    console.log("[DIAGNOSTIC] SESSION SAVED");
  } catch (error) {
    console.warn(
      '[Sentinel-E Auth] Failed to persist snapshot:',
      error
    );

    clearSupabaseSessionSnapshot();
  }
}

export function clearSupabaseSessionSnapshot() {
  if (!canUseLocalStorage()) {
    return;
  }

  window.localStorage.removeItem(AUTH_SNAPSHOT_KEY);
}

export async function restoreSupabaseSession() {
  if (!isSupabaseConfigured) {
    return {
      session: null,
      user: null,
      error: null,
    };
  }

  try {
    const supabase = getSupabaseClient();

    const { data, error } =
      await supabase.auth.getSession();

    const session = data?.session || null;

    const normalizedUser = normalizeSupabaseUser(
      session?.user || null
    );

    /**
     * ============================================================
     * HARD INVALIDATION
     * ============================================================
     */

    if (!normalizedUser?.id) {
      clearSupabaseSessionSnapshot();

      return {
        session: null,
        user: null,
        error: error || null,
      };
    }

    persistSupabaseSessionSnapshot(session);
    console.log("[DIAGNOSTIC] RESTORED SESSION", session);

    return {
      session,
      user: normalizedUser,
      error: error || null,
    };
  } catch (error) {
    clearSupabaseSessionSnapshot();

    return {
      session: null,
      user: null,
      error,
    };
  }
}

export function subscribeToSupabaseSession(listener) {
  if (!isSupabaseConfigured) {
    return () => {};
  }

  const supabase = getSupabaseClient();
  const { data } = supabase.auth.onAuthStateChange((_event, session) => {
    console.log("[DIAGNOSTIC] CALLBACK SESSION (onAuthStateChange event:", _event, " session:", session, ")");
    persistSupabaseSessionSnapshot(session);
    const user = normalizeSupabaseUser(session?.user || null);
    listener({ session: session || null, user });
  });

  return () => data?.subscription?.unsubscribe?.();
}

export async function signInWithGoogleOAuth({ redirectTo } = {}) {
  if (!isSupabaseConfigured) {
    throw new Error('Supabase auth is not configured.');
  }

  const supabase = getSupabaseClient();
  const resolvedRedirect = redirectTo || `${window.location.origin}/chat`;
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'google',
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

export async function signInWithEmail(email: string, password: string) {
  if (!isSupabaseConfigured) {
    throw new Error('Supabase auth is not configured.');
  }

  if (!email || typeof email !== 'string' || !email.includes('@')) {
    throw new Error('A valid email address is required.');
  }
  if (!password || typeof password !== 'string' || password.length < 6) {
    throw new Error('Password must be at least 6 characters.');
  }

  const supabase = getSupabaseClient();
  const { data, error } = await supabase.auth.signInWithPassword({
    email: email.trim().toLowerCase(),
    password,
  });

  console.log("[RAW LOGIN RESULT]", { data, error });
  console.log("[RAW SESSION]", data?.session);
  console.log("[RAW USER]", data?.user);

  if (error) {
    console.error('[Supabase] signInWithEmail error:', error.status, error.message);
    throw error;
  }

  return data;
}

export async function signUpWithEmail(email: string, password: string, name?: string) {
  if (!isSupabaseConfigured) {
    throw new Error('Supabase auth is not configured.');
  }

  if (!email || typeof email !== 'string' || !email.includes('@')) {
    throw new Error('A valid email address is required.');
  }
  if (!password || typeof password !== 'string' || password.length < 6) {
    throw new Error('Password must be at least 6 characters.');
  }

  const supabase = getSupabaseClient();
  const { data, error } = await supabase.auth.signUp({
    email: email.trim().toLowerCase(),
    password,
    options: {
      data: {
        full_name: name
      }
    },
  });

  console.log("[RAW SIGNUP RESULT]", { data, error });
  console.log("[RAW SESSION]", data?.session);
  console.log("[RAW USER]", data?.user);

  if (error) {
    console.error('[Supabase] signUpWithEmail error:', error.status, error.message);
    throw error;
  }

  if (data?.user && !data.session && data.user.identities?.length === 0) {
    console.warn('[Supabase] signUpWithEmail: email already registered (no identities returned)');
    throw new Error('An account with this email already exists. Please sign in instead.');
  }

  return data;
}
export async function resetPasswordForEmail(email: string) {
  if (!isSupabaseConfigured) {
    throw new Error('Supabase auth is not configured.');
  }

  if (!email || typeof email !== 'string' || !email.includes('@')) {
    throw new Error('A valid email address is required.');
  }

  const supabase = getSupabaseClient();
  const { data, error } = await supabase.auth.resetPasswordForEmail(email.trim().toLowerCase(), {
    redirectTo: `${window.location.origin}/reset-password`,
  });

  if (error) {
    throw error;
  }

  return data;
}

export async function updateUserPassword(password: string) {
  if (!isSupabaseConfigured) {
    throw new Error('Supabase auth is not configured.');
  }

  if (!password || typeof password !== 'string' || password.length < 6) {
    throw new Error('Password must be at least 6 characters.');
  }

  const supabase = getSupabaseClient();
  const { data, error } = await supabase.auth.updateUser({
    password,
  });

  if (error) {
    throw error;
  }

  return data;
}
