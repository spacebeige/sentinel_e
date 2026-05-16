/**
 * ============================================================
 * sessionPersistence.js — Authenticated User Session Storage
 * ============================================================
 *
 * PRODUCTION ARCHITECTURE:
 *   - All persistence is user-scoped: conversation:${user.id}
 *   - Keys NEVER use 'guest-session' for authenticated users
 *   - Guest state NEVER hydrates into authenticated user slots
 *   - No mixed-key corruption possible when auth is active
 *
 * HIDDEN GUEST FALLBACK (dev/emergency only):
 *   - Isolated under 'guest-session' namespace
 *   - Only accessible when HIDDEN_GUEST_FALLBACK_ENABLED=true
 *   - Guest keys and auth user keys are strictly separated
 *
 * Storage strategy:
 * - localStorage: long-lived conversation/session state per authenticated user
 * - sessionStorage: hydration in-flight markers to prevent race conditions
 *
 * TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
 */

const STORAGE_PREFIX = 'sentinel';
const AUTH_USER_KEY = `${STORAGE_PREFIX}-auth-user-id`;
// Legacy guest keys — preserved for hidden dev fallback only
const LEGACY_SESSION_KEY = 'sentinel-guest-session';
const LEGACY_CONVERSATIONS_KEY = 'sentinel-guest-conversations';
const LEGACY_ACTIVE_CHAT_KEY = 'sentinel-active-chat-id';
const LEGACY_UI_STATE_KEY = 'sentinel-guest-ui-state';
const LEGACY_MIGRATION_KEY_PREFIX = `${STORAGE_PREFIX}-legacy-migrated`;
// DEFAULT_FALLBACK_USER: isolated namespace for hidden guest fallback (dev/emergency only)
const DEFAULT_FALLBACK_USER = 'guest-session';
const guestModeRaw = String(process.env.REACT_APP_GUEST_MODE ?? 'false').trim().toLowerCase();
// TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
const HIDDEN_GUEST_FALLBACK_ENABLED = guestModeRaw === 'true' && process.env.NODE_ENV !== 'production';

let activePersistenceUserId = null;

function canUseStorage(storage) {
  try {
    if (!storage) return false;
    const key = '__sentinel_storage_test__';
    storage.setItem(key, key);
    storage.removeItem(key);
    return true;
  } catch {
    return false;
  }
}

const hasLocalStorage = () => typeof window !== 'undefined' && canUseStorage(window.localStorage);
const hasSessionStorage = () => typeof window !== 'undefined' && canUseStorage(window.sessionStorage);

function safeParse(raw, fallback) {
  if (!raw) return fallback;
  try {
    const parsed = JSON.parse(raw);
    return parsed ?? fallback;
  } catch {
    return fallback;
  }
}

function readLocal(key, fallback) {
  if (!hasLocalStorage()) return fallback;
  return safeParse(window.localStorage.getItem(key), fallback);
}

function writeLocal(key, value) {
  if (!hasLocalStorage()) return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.warn(`Unable to persist ${key}`, error);
  }
}

function readLocalRaw(key, fallback = null) {
  if (!hasLocalStorage()) return fallback;
  const value = window.localStorage.getItem(key);
  return value ?? fallback;
}

function writeLocalRaw(key, value) {
  if (!hasLocalStorage()) return;
  if (value === null || value === undefined) {
    window.localStorage.removeItem(key);
    return;
  }
  window.localStorage.setItem(key, String(value));
}

function writeSessionRaw(key, value) {
  if (!hasSessionStorage()) return;
  if (value === null || value === undefined) {
    window.sessionStorage.removeItem(key);
    return;
  }
  window.sessionStorage.setItem(key, String(value));
}

function nowIso() {
  return new Date().toISOString();
}

function generateId(prefix = '') {
  const id = typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 12)}`;
  return `${prefix}${id}`;
}

function storageKeysForUser(userId) {
  return {
    session: `session:${userId}`,
    conversations: `conversation:${userId}`,
    activeChat: `activeConversation:${userId}`,
    hydration: `hydration:${userId}`,
    uiState: `uiState:${userId}`,
  };
}

function legacyStorageKeysForUser(userId) {
  return {
    session: `${STORAGE_PREFIX}-session-${userId}`,
    conversations: `${STORAGE_PREFIX}-conversations-${userId}`,
    activeChat: `${STORAGE_PREFIX}-active-chat-${userId}`,
    hydration: `${STORAGE_PREFIX}-hydrating-${userId}`,
    uiState: `${STORAGE_PREFIX}-ui-state-${userId}`,
  };
}

function normalizeUserId(rawUserId) {
  if (!rawUserId) return null;
  const value = String(rawUserId).trim();
  if (!value) return null;
  return value.slice(0, 200);
}

function isGuestSessionId(userId) {
  if (!userId) return false;
  const normalized = String(userId).toLowerCase();
  return normalized === DEFAULT_FALLBACK_USER || normalized.startsWith('guest');
}

function resolveUserId(userId = null) {
  const explicit = normalizeUserId(userId);
  if (explicit) {
    // Authenticated user IDs always take priority
    if (!isGuestSessionId(explicit)) return explicit;
    // Guest ID: only allowed when fallback is explicitly enabled
    if (HIDDEN_GUEST_FALLBACK_ENABLED) return explicit;
    return null;
  }

  const active = normalizeUserId(activePersistenceUserId);
  if (active) {
    if (!isGuestSessionId(active)) return active;
    if (HIDDEN_GUEST_FALLBACK_ENABLED) return active;
    return null;
  }

  const stored = normalizeUserId(readLocalRaw(AUTH_USER_KEY, null));
  if (stored) {
    // Defensive: never use a stored guest key as auth user in production
    if (isGuestSessionId(stored) && !HIDDEN_GUEST_FALLBACK_ENABLED) return null;
    return stored;
  }

  // Production: return null (no implicit guest fallback)
  // TODO: Remove hidden guest fallback after production auth architecture fully stabilizes
  if (HIDDEN_GUEST_FALLBACK_ENABLED) {
    return DEFAULT_FALLBACK_USER;
  }

  return null;
}

function getConversationsMap(userId) {
  if (!userId) return {};
  const keys = storageKeysForUser(userId);
  const legacyKeys = legacyStorageKeysForUser(userId);
  const stored = readLocal(keys.conversations, readLocal(legacyKeys.conversations, {}));
  return stored && typeof stored === 'object' && !Array.isArray(stored) ? stored : {};
}

function persistConversationsMap(userId, conversations) {
  if (!userId) return;
  const keys = storageKeysForUser(userId);
  writeLocal(keys.conversations, conversations && typeof conversations === 'object' ? conversations : {});
}

function normalizeMessages(messages) {
  return Array.isArray(messages) ? messages.filter(Boolean) : [];
}

function normalizeConversation(conversation = {}, fallbackId = null) {
  const id = conversation.id || fallbackId || generateId();
  const messages = normalizeMessages(conversation.messages);
  const createdAt = conversation.createdAt || conversation.created_at || nowIso();
  const updatedAt = conversation.updatedAt || conversation.updated_at || createdAt;

  return {
    id,
    title: conversation.title || conversation.chat_name || conversation.preview || 'New Chat',
    mode: conversation.mode || 'standard',
    subMode: conversation.subMode ?? conversation.sub_mode ?? null,
    createdAt,
    updatedAt,
    messages,
    currentResult: conversation.currentResult ?? null,
    sessionState: conversation.sessionState ?? null,
    lastQueryText: conversation.lastQueryText || '',
    lastResponseText: conversation.lastResponseText || '',
    governanceVerdict: conversation.governanceVerdict ?? null,
    orchestrationState: conversation.orchestrationState ?? null,
    analyticsState: conversation.analyticsState ?? null,
    debateState: conversation.debateState ?? null,
    tacticalMapState: conversation.tacticalMapState ?? null,
    metadata: {
      messageCount: messages.length,
      ...(conversation.metadata || {}),
    },
  };
}

function titleFromMessages(messages) {
  const firstUserMessage = normalizeMessages(messages).find((message) => message.role === 'user');
  const content = String(firstUserMessage?.content || '').trim();
  if (!content) return 'New Chat';
  return content.length > 54 ? `${content.slice(0, 54)}...` : content;
}

function createSessionShell(userId) {
  const timestamp = nowIso();
  return {
    id: `session-${userId}`,
    userId,
    createdAt: timestamp,
    updatedAt: timestamp,
    activeChatId: null,
    metadata: {
      client: 'web',
      mode: 'authenticated',
    },
  };
}

function migrateLegacyGuestState(userId) {
  if (!HIDDEN_GUEST_FALLBACK_ENABLED || !isGuestSessionId(userId)) {
    return;
  }
  const migrationKey = `${LEGACY_MIGRATION_KEY_PREFIX}-${userId}`;
  if (!hasLocalStorage() || readLocalRaw(migrationKey, null) === 'true') return;

  const keys = storageKeysForUser(userId);
  if (readLocalRaw(keys.session, null)) {
    writeLocalRaw(migrationKey, 'true');
    return;
  }

  const legacySession = readLocal(LEGACY_SESSION_KEY, null);
  const legacyConversations = readLocal(LEGACY_CONVERSATIONS_KEY, null);
  const legacyActiveChat = readLocalRaw(LEGACY_ACTIVE_CHAT_KEY, null);
  const legacyUiState = readLocal(LEGACY_UI_STATE_KEY, null);

  if (legacySession && typeof legacySession === 'object') {
    writeLocal(keys.session, {
      ...createSessionShell(userId),
      activeChatId: legacyActiveChat || legacySession.activeChatId || null,
      updatedAt: nowIso(),
      metadata: {
        ...(legacySession.metadata || {}),
        migratedFromGuest: true,
      },
    });
  }

  if (legacyConversations && typeof legacyConversations === 'object') {
    writeLocal(keys.conversations, legacyConversations);
  }

  if (legacyActiveChat) {
    writeLocalRaw(keys.activeChat, legacyActiveChat);
  }

  if (legacyUiState && typeof legacyUiState === 'object') {
    writeLocal(keys.uiState, legacyUiState);
  }

  writeLocalRaw(migrationKey, 'true');
}

function migrateLegacyUserScopedState(userId) {
  if (!userId || isGuestSessionId(userId)) return;
  const migrationKey = `${LEGACY_MIGRATION_KEY_PREFIX}-userscoped-${userId}`;
  if (!hasLocalStorage() || readLocalRaw(migrationKey, null) === 'true') return;

  const keys = storageKeysForUser(userId);
  const legacyKeys = legacyStorageKeysForUser(userId);

  const legacySession = readLocal(legacyKeys.session, null);
  const legacyConversations = readLocal(legacyKeys.conversations, null);
  const legacyActiveChat = readLocalRaw(legacyKeys.activeChat, null);
  const legacyUiState = readLocal(legacyKeys.uiState, null);

  if (!readLocal(keys.session, null) && legacySession && typeof legacySession === 'object') {
    writeLocal(keys.session, { ...legacySession, userId });
  }

  if (!readLocal(keys.conversations, null) && legacyConversations && typeof legacyConversations === 'object') {
    writeLocal(keys.conversations, legacyConversations);
  }

  if (!readLocalRaw(keys.activeChat, null) && legacyActiveChat) {
    writeLocalRaw(keys.activeChat, legacyActiveChat);
  }

  if (!readLocal(keys.uiState, null) && legacyUiState && typeof legacyUiState === 'object') {
    writeLocal(keys.uiState, legacyUiState);
  }

  writeLocalRaw(migrationKey, 'true');
}

export function setPersistenceUser(userId) {
  const resolved = resolveUserId(userId);
  if (!resolved) return null;

  const stored = normalizeUserId(readLocalRaw(AUTH_USER_KEY, null));

  // DEFENSIVE GUARD: Guest session NEVER overwrites an authenticated user's stored key.
  // If a real (non-guest) user ID is already stored, reject the guest takeover.
  if (isGuestSessionId(resolved) && stored && !isGuestSessionId(stored)) {
    // Real user exists — do not allow guest to override
    activePersistenceUserId = stored;
    return stored;
  }

  activePersistenceUserId = resolved;
  // Only persist non-guest IDs to localStorage
  if (!isGuestSessionId(resolved)) {
    writeLocalRaw(AUTH_USER_KEY, resolved);
  }
  return resolved;
}

export function getPersistenceUserId() {
  return resolveUserId();
}

export function clearPersistenceUser() {
  activePersistenceUserId = null;
  writeLocalRaw(AUTH_USER_KEY, null);
}

export function restoreUserSession(options = {}) {
  const userId = setPersistenceUser(options.userId);
  if (!userId) return null;

  if (isGuestSessionId(userId) && !HIDDEN_GUEST_FALLBACK_ENABLED) {
    return null;
  }

  migrateLegacyUserScopedState(userId);
  migrateLegacyGuestState(userId);

  const keys = storageKeysForUser(userId);
  const wasHydrating = isSessionHydrating({ userId });
  if (!wasHydrating) {
    writeSessionRaw(keys.hydration, 'true');
  }

  const existing = readLocal(keys.session, null);
  const session = existing?.id
    ? {
      ...createSessionShell(userId),
      ...existing,
      userId,
    }
    : createSessionShell(userId);

  writeLocal(keys.session, session);

  const activeChatId = readLocalRaw(keys.activeChat, null) || session.activeChatId || null;
  const uiState = readLocal(keys.uiState, {});
  const conversations = getConversationsMap(userId);

  if (!wasHydrating) {
    writeSessionRaw(keys.hydration, null);
  }

  return {
    ...session,
    activeChatId,
    uiState: uiState && typeof uiState === 'object' ? uiState : {},
    conversations,
    userId,
  };
}

export function loadConversationState(conversationId, options = {}) {
  if (!conversationId) return null;
  const userId = resolveUserId(options.userId);
  if (!userId) return null;
  const conversations = getConversationsMap(userId);
  const conversation = conversations[conversationId];
  return conversation ? normalizeConversation(conversation, conversationId) : null;
}

export function saveConversationState(conversationId, state = {}, options = {}) {
  if (!conversationId) return null;
  const userId = setPersistenceUser(options.userId);
  if (!userId) return null;
  const keys = storageKeysForUser(userId);
  const session = restoreUserSession({ userId });
  if (!session) return null;
  const conversations = getConversationsMap(userId);
  const previous = conversations[conversationId] || {};
  const normalized = normalizeConversation({
    ...previous,
    ...state,
    id: conversationId,
    title: state.title || previous.title || titleFromMessages(state.messages || previous.messages),
    updatedAt: nowIso(),
  }, conversationId);

  conversations[conversationId] = normalized;
  persistConversationsMap(userId, conversations);

  const nextSession = {
    ...session,
    activeChatId: conversationId,
    updatedAt: nowIso(),
  };
  writeLocal(keys.session, nextSession);
  writeLocalRaw(keys.activeChat, conversationId);

  if (state.uiState && typeof state.uiState === 'object') {
    writeLocal(keys.uiState, state.uiState);
  }

  return normalized;
}

export function saveConversationHistory(conversationId, messages = [], state = {}, options = {}) {
  return saveConversationState(conversationId, {
    ...state,
    messages: normalizeMessages(messages),
  }, options);
}

export function loadConversationHistory(conversationId = null, options = {}) {
  if (conversationId) {
    return loadConversationState(conversationId, options);
  }
  return listConversationHistory(options);
}

export function listConversationHistory(options = {}) {
  const userId = resolveUserId(options.userId);
  if (!userId) return [];
  const conversations = getConversationsMap(userId);
  return Object.values(conversations)
    .map((conversation) => normalizeConversation(conversation))
    .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
}

export function switchConversation(conversationId, options = {}) {
  if (!conversationId) return null;

  const userId = setPersistenceUser(options.userId);
  if (!userId) return null;
  const keys = storageKeysForUser(userId);
  const session = restoreUserSession({ userId });
  if (!session) return null;
  const conversations = getConversationsMap(userId);

  if (!conversations[conversationId] && options.createIfMissing) {
    conversations[conversationId] = normalizeConversation({
      id: conversationId,
      mode: options.mode || 'standard',
      subMode: options.subMode || null,
      messages: [],
    }, conversationId);
    persistConversationsMap(userId, conversations);
  }

  const nextSession = {
    ...session,
    activeChatId: conversationId,
    updatedAt: nowIso(),
  };
  writeLocal(keys.session, nextSession);
  writeLocalRaw(keys.activeChat, conversationId);

  return loadConversationState(conversationId, { userId });
}

export function createNewConversation(options = {}) {
  const userId = setPersistenceUser(options.userId);
  if (!userId) return null;
  const id = options.id || generateId();
  const timestamp = nowIso();

  const conversation = normalizeConversation({
    id,
    title: options.title || 'New Chat',
    mode: options.mode || 'standard',
    subMode: options.subMode || null,
    createdAt: timestamp,
    updatedAt: timestamp,
    messages: [],
    sessionState: null,
    currentResult: null,
    lastQueryText: '',
    lastResponseText: '',
    governanceVerdict: null,
  }, id);

  saveConversationState(id, conversation, { userId });
  return conversation;
}

export function persistSessionState(state = {}, options = {}) {
  const userId = setPersistenceUser(options.userId);
  if (!userId) return { session: null, conversation: null };
  const keys = storageKeysForUser(userId);
  const session = restoreUserSession({ userId });
  if (!session) return { session: null, conversation: null };
  const conversationId = state.conversationId || state.activeChatId || session.activeChatId || null;
  const sessionMetadata = state.metadata && typeof state.metadata === 'object' ? state.metadata : {};

  const nextSession = {
    ...session,
    activeChatId: conversationId,
    updatedAt: nowIso(),
    metadata: {
      ...(session.metadata || {}),
      ...sessionMetadata,
    },
  };

  writeLocal(keys.session, nextSession);

  if (conversationId) {
    writeLocalRaw(keys.activeChat, conversationId);
  } else {
    writeLocalRaw(keys.activeChat, null);
  }

  if (state.uiState && typeof state.uiState === 'object') {
    writeLocal(keys.uiState, state.uiState);
  }

  let conversation = null;
  if (conversationId) {
    const conversationState = { ...state };
    delete conversationState.conversationId;
    delete conversationState.activeChatId;
    delete conversationState.metadata;
    conversation = saveConversationState(conversationId, conversationState, { userId });
  }

  return { session: nextSession, conversation };
}

export function isSessionHydrating(options = {}) {
  const userId = resolveUserId(options.userId);
  if (!userId) return false;
  const keys = storageKeysForUser(userId);
  if (!hasSessionStorage()) return false;
  return window.sessionStorage.getItem(keys.hydration) === 'true';
}

// Compatibility shims for legacy imports.
export function createGuestSession() {
  const session = restoreUserSession({ userId: DEFAULT_FALLBACK_USER, allowGuest: true });
  if (!session) {
    return {
      ...createSessionShell(DEFAULT_FALLBACK_USER),
      guestSessionId: DEFAULT_FALLBACK_USER,
      uiState: {},
      conversations: {},
    };
  }
  return {
    ...session,
    guestSessionId: session.userId,
  };
}

export function restoreGuestSession() {
  const session = restoreUserSession({ userId: DEFAULT_FALLBACK_USER, allowGuest: true });
  if (!session) {
    return {
      ...createSessionShell(DEFAULT_FALLBACK_USER),
      guestSessionId: DEFAULT_FALLBACK_USER,
      uiState: {},
      conversations: {},
    };
  }
  return {
    ...session,
    guestSessionId: session.userId,
  };
}

export function getGuestSessionId() {
  return getPersistenceUserId();
}

export function isGuestHydrating() {
  return isSessionHydrating();
}
