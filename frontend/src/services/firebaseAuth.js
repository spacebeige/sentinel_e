/**
 * ============================================================
 * Auth Service (legacy path retained for compatibility)
 * ============================================================
 *
 * This file replaces the previous Firebase implementation with
 * a SuperTokens + FastAPI session flow while keeping the existing
 * import path stable across the app.
 */

import SuperTokens, { SuperTokensWrapper } from 'supertokens-auth-react';
import Session from 'supertokens-auth-react/recipe/session';
import ThirdParty, {
  Github,
  Google,
} from 'supertokens-auth-react/recipe/thirdparty';

const API_DOMAIN = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const AUTH_API_BASE_PATH = process.env.REACT_APP_AUTH_API_BASE_PATH || '/auth';
const AUTH_WEBSITE_BASE_PATH = process.env.REACT_APP_AUTH_WEBSITE_BASE_PATH || '/auth';

const AUTH_PROVIDER_KEY = 'sentinel-e-auth-provider';
const AUTH_RETURN_TO_KEY = 'sentinel-e-auth-return-to';
const AUTH_POPUP_EVENT = 'sentinel-e:auth-popup';
export const AUTH_REQUIRED_EVENT = 'sentinel-e:auth-required';
export const AUTH_STATE_CHANGED_EVENT = 'sentinel-e:auth-state-changed';

let initialized = false;

export const USER_ROLES = {
  ADMIN: 'admin',
  USER: 'user',
};

export const db = null;

function getWebsiteDomain() {
  if (typeof window !== 'undefined') {
    return window.location.origin;
  }
  return process.env.REACT_APP_WEBSITE_URL || 'http://localhost:3000';
}

export function initAuthClient() {
  if (initialized || typeof window === 'undefined') {
    return;
  }

  SuperTokens.init({
    appInfo: {
      appName: 'Sentinel-E',
      apiDomain: API_DOMAIN,
      websiteDomain: getWebsiteDomain(),
      apiBasePath: AUTH_API_BASE_PATH,
      websiteBasePath: AUTH_WEBSITE_BASE_PATH,
    },
    recipeList: [
      ThirdParty.init({
        signInAndUpFeature: {
          providers: [Google.init(), Github.init()],
        },
      }),
      Session.init(),
    ],
  });

  initialized = true;
}

initAuthClient();

function dispatchAuthStateChanged(user = null) {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(
    new CustomEvent(AUTH_STATE_CHANGED_EVENT, {
      detail: { user },
    })
  );
}

export function requestAuthModal(detail = {}) {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(
    new CustomEvent(AUTH_REQUIRED_EVENT, {
      detail,
    })
  );
}

function buildPopupFeatures(width = 520, height = 720) {
  const dualScreenLeft = window.screenLeft ?? window.screenX ?? 0;
  const dualScreenTop = window.screenTop ?? window.screenY ?? 0;
  const viewportWidth = window.innerWidth || document.documentElement.clientWidth || window.screen.width;
  const viewportHeight = window.innerHeight || document.documentElement.clientHeight || window.screen.height;

  const left = Math.max(0, dualScreenLeft + (viewportWidth - width) / 2);
  const top = Math.max(0, dualScreenTop + (viewportHeight - height) / 2);

  return [
    'popup=yes',
    `width=${width}`,
    `height=${height}`,
    `left=${Math.round(left)}`,
    `top=${Math.round(top)}`,
    'resizable=yes',
    'scrollbars=yes',
  ].join(',');
}

function getAuthCallbackUrl() {
  const url = new URL('/', getWebsiteDomain());
  url.searchParams.set('auth_callback', '1');
  return url.toString();
}

function cleanupCallbackParams() {
  if (typeof window === 'undefined') return;
  const url = new URL(window.location.href);
  ['auth_callback', 'code', 'state', 'error', 'error_description'].forEach((key) => {
    url.searchParams.delete(key);
  });
  window.history.replaceState({}, document.title, `${url.pathname}${url.search}${url.hash}`);
}

function deriveProfileFromResponse(user, fallbackProvider = null) {
  const email =
    user?.email ||
    user?.emails?.[0] ||
    user?.loginMethods?.find((method) => Array.isArray(method?.email) ? method.email.length > 0 : method?.email)?.email ||
    null;

  const provider =
    user?.thirdParty?.id ||
    user?.thirdParty?.[0]?.id ||
    user?.loginMethods?.find((method) => method?.thirdParty?.id)?.thirdParty?.id ||
    fallbackProvider;

  const name =
    user?.name ||
    user?.rawUserInfoFromProvider?.fromUserInfoAPI?.name ||
    user?.loginMethods?.find((method) => method?.rawUserInfoFromProvider?.fromUserInfoAPI?.name)
      ?.rawUserInfoFromProvider?.fromUserInfoAPI?.name ||
    (email ? email.split('@')[0] : 'User');

  return { email, name, provider };
}

export async function syncCurrentUser(profile = {}) {
  const response = await fetch(`${API_DOMAIN}/api/auth/sync-user`, {
    method: 'POST',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(profile),
  });

  if (!response.ok) {
    throw new Error('Unable to finish sign in.');
  }

  return response.json();
}

function waitForPopupResult(popup) {
  return new Promise((resolve, reject) => {
    let settled = false;

    const cleanup = () => {
      settled = true;
      window.removeEventListener('message', onMessage);
      window.clearInterval(closePoll);
    };

    const onMessage = (event) => {
      if (event.origin !== getWebsiteDomain()) return;
      if (event.data?.type !== AUTH_POPUP_EVENT) return;
      cleanup();
      if (event.data.success) {
        resolve(event.data.user || null);
      } else {
        reject(new Error(event.data.error || 'Sign in was cancelled.'));
      }
    };

    const closePoll = window.setInterval(() => {
      if (!popup || popup.closed) {
        if (!settled) {
          cleanup();
          reject(new Error('Sign in was cancelled.'));
        }
      }
    }, 250);

    window.addEventListener('message', onMessage);
  });
}

export async function authenticateWithProvider(provider, options = {}) {
  initAuthClient();

  const returnTo = options.returnTo || '/chat';
  window.sessionStorage.setItem(AUTH_PROVIDER_KEY, provider);
  window.sessionStorage.setItem(AUTH_RETURN_TO_KEY, returnTo);

  const authorizationUrl =
    await ThirdParty.getAuthorisationURLWithQueryParamsAndSetState({
      thirdPartyId: provider,
      frontendRedirectURI: getAuthCallbackUrl(),
    });

  const popup = window.open(
    authorizationUrl,
    `sentinel-e-${provider}-auth`,
    buildPopupFeatures()
  );

  if (!popup) {
    throw new Error('Popup blocked. Please allow popups and try again.');
  }

  popup.focus();
  const syncedUser = await waitForPopupResult(popup);
  dispatchAuthStateChanged(syncedUser);
  return syncedUser;
}

export async function handleAuthCallbackIfPresent() {
  if (typeof window === 'undefined') return false;

  const url = new URL(window.location.href);
  const hasOAuthParams = url.searchParams.has('code') || url.searchParams.has('error');

  if (!hasOAuthParams) {
    return false;
  }

  try {
    initAuthClient();

    if (url.searchParams.has('error')) {
      throw new Error(url.searchParams.get('error_description') || 'Authentication failed.');
    }

    const response = await ThirdParty.signInAndUp();
    if (response.status !== 'OK') {
      throw new Error('Unable to finish sign in.');
    }

    const fallbackProvider = window.sessionStorage.getItem(AUTH_PROVIDER_KEY);
    const profile = deriveProfileFromResponse(response.user, fallbackProvider);
    const syncedUser = await syncCurrentUser(profile);

    window.sessionStorage.removeItem(AUTH_PROVIDER_KEY);

    if (window.opener && !window.opener.closed) {
      window.opener.postMessage(
        {
          type: AUTH_POPUP_EVENT,
          success: true,
          user: syncedUser,
        },
        getWebsiteDomain()
      );
      window.close();
      return true;
    }

    cleanupCallbackParams();
    dispatchAuthStateChanged(syncedUser);
    const returnTo = window.sessionStorage.getItem(AUTH_RETURN_TO_KEY) || '/chat';
    window.sessionStorage.removeItem(AUTH_RETURN_TO_KEY);
    window.location.replace(returnTo);
    return true;
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Authentication failed.';
    if (window.opener && !window.opener.closed) {
      window.opener.postMessage(
        {
          type: AUTH_POPUP_EVENT,
          success: false,
          error: message,
        },
        getWebsiteDomain()
      );
      window.close();
      return true;
    }

    cleanupCallbackParams();
    throw error;
  }
}

export async function getCurrentUser() {
  initAuthClient();

  const hasSession = await Session.doesSessionExist().catch(() => false);
  if (!hasSession) {
    return null;
  }

  const response = await fetch(`${API_DOMAIN}/api/auth/me`, {
    method: 'GET',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
    },
    cache: 'no-store',
  });

  if (response.status === 401) {
    return null;
  }

  if (!response.ok) {
    throw new Error('Unable to load your account.');
  }

  return response.json();
}

export async function signOutUser() {
  initAuthClient();
  await Session.signOut();
  dispatchAuthStateChanged(null);
  return { success: true };
}

export async function attemptSessionRefresh() {
  initAuthClient();
  return Session.attemptRefreshingSession();
}

export async function isUserAdmin() {
  const currentUser = await getCurrentUser();
  return currentUser?.role === USER_ROLES.ADMIN;
}

export async function getAllUsers() {
  return [];
}

export async function updateUserProfile() {
  return { success: false, error: 'Profile editing is not implemented.' };
}

export async function getUserProfile() {
  return getCurrentUser();
}

const authService = {
  AUTH_REQUIRED_EVENT,
  AUTH_STATE_CHANGED_EVENT,
  USER_ROLES,
  SuperTokensWrapper,
  attemptSessionRefresh,
  authenticateWithProvider,
  getCurrentUser,
  getUserProfile,
  handleAuthCallbackIfPresent,
  initAuthClient,
  isUserAdmin,
  requestAuthModal,
  signOutUser,
  syncCurrentUser,
};

export { SuperTokensWrapper };
export default authService;
