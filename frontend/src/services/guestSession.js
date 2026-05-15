/**
 * Temporary local guest-session persistence for auth-disabled Sentinel-E.
 *
 * TODO: Replace guest-session persistence with Firebase-auth session persistence later
 */

const SESSION_KEY = 'sentinel-guest-session';
const CONVERSATIONS_KEY = 'sentinel-guest-conversations';
const ACTIVE_CHAT_KEY = 'sentinel-active-chat-id';
const HYDRATION_KEY = 'sentinel-guest-hydrating';
const UI_STATE_KEY = 'sentinel-guest-ui-state';

const FALLBACK_SESSION_ID = 'guest-session';

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

function nowIso() {
  return new Date().toISOString();
}

function generateId(prefix = '') {
  const id = typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 12)}`;
  return `${prefix}${id}`;
}

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

function writeSession(key, value) {
  if (!hasSessionStorage()) return;
  try {
    window.sessionStorage.setItem(key, value);
  } catch {
    /* storage is best-effort */
  }
}

function removeSession(key) {
  if (!hasSessionStorage()) return;
  try {
    window.sessionStorage.removeItem(key);
  } catch {
    /* storage is best-effort */
  }
}

function getConversationsMap() {
  const stored = readLocal(CONVERSATIONS_KEY, {});
  return stored && typeof stored === 'object' && !Array.isArray(stored) ? stored : {};
}

function persistConversationsMap(conversations) {
  writeLocal(CONVERSATIONS_KEY, conversations && typeof conversations === 'object' ? conversations : {});
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

export function createGuestSession() {
  const existing = readLocal(SESSION_KEY, null);
  if (existing?.id) {
    return existing;
  }

  const timestamp = nowIso();
  const session = {
    id: generateId('guest-'),
    createdAt: timestamp,
    updatedAt: timestamp,
    activeChatId: null,
    metadata: {
      client: 'web',
      mode: 'guest',
    },
  };

  writeLocal(SESSION_KEY, session);
  return session;
}

export function restoreGuestSession() {
  const wasHydrating = isGuestHydrating();
  if (!wasHydrating) {
    writeSession(HYDRATION_KEY, 'true');
  }
  const session = createGuestSession();
  const activeChatId = (hasLocalStorage() && window.localStorage.getItem(ACTIVE_CHAT_KEY)) || session.activeChatId || null;
  const uiState = readLocal(UI_STATE_KEY, {});
  const conversations = getConversationsMap();

  const restored = {
    ...session,
    activeChatId,
    uiState: uiState && typeof uiState === 'object' ? uiState : {},
    conversations,
    guestSessionId: session.id || FALLBACK_SESSION_ID,
  };

  if (!wasHydrating) {
    removeSession(HYDRATION_KEY);
  }
  return restored;
}

export function getGuestSessionId() {
  return restoreGuestSession().guestSessionId || FALLBACK_SESSION_ID;
}

export function saveConversationState(conversationId, state = {}) {
  if (!conversationId) return null;
  const session = createGuestSession();
  const conversations = getConversationsMap();
  const previous = conversations[conversationId] || {};
  const normalized = normalizeConversation({
    ...previous,
    ...state,
    id: conversationId,
    title: state.title || previous.title || titleFromMessages(state.messages || previous.messages),
    updatedAt: nowIso(),
  }, conversationId);

  conversations[conversationId] = normalized;
  persistConversationsMap(conversations);

  const nextSession = {
    ...session,
    activeChatId: conversationId,
    updatedAt: nowIso(),
  };
  writeLocal(SESSION_KEY, nextSession);
  if (hasLocalStorage()) window.localStorage.setItem(ACTIVE_CHAT_KEY, conversationId);

  if (state.uiState && typeof state.uiState === 'object') {
    writeLocal(UI_STATE_KEY, state.uiState);
  }

  return normalized;
}

export function loadConversationState(conversationId) {
  if (!conversationId) return null;
  const conversations = getConversationsMap();
  const conversation = conversations[conversationId];
  return conversation ? normalizeConversation(conversation, conversationId) : null;
}

export function saveConversationHistory(conversationId, messages = [], state = {}) {
  return saveConversationState(conversationId, {
    ...state,
    messages: normalizeMessages(messages),
  });
}

export function loadConversationHistory(conversationId = null) {
  if (conversationId) {
    return loadConversationState(conversationId);
  }
  return listConversationHistory();
}

export function switchConversation(conversationId, options = {}) {
  if (!conversationId) return null;
  const session = createGuestSession();
  const conversations = getConversationsMap();

  if (!conversations[conversationId] && options.createIfMissing) {
    conversations[conversationId] = normalizeConversation({
      id: conversationId,
      mode: options.mode || 'standard',
      subMode: options.subMode || null,
      messages: [],
    }, conversationId);
    persistConversationsMap(conversations);
  }

  const nextSession = {
    ...session,
    activeChatId: conversationId,
    updatedAt: nowIso(),
  };
  writeLocal(SESSION_KEY, nextSession);
  if (hasLocalStorage()) window.localStorage.setItem(ACTIVE_CHAT_KEY, conversationId);
  return loadConversationState(conversationId);
}

export function createNewConversation(options = {}) {
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

  saveConversationState(id, conversation);
  return conversation;
}

export function persistSessionState(state = {}) {
  const session = createGuestSession();
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

  writeLocal(SESSION_KEY, nextSession);
  if (hasLocalStorage()) {
    if (conversationId) {
      window.localStorage.setItem(ACTIVE_CHAT_KEY, conversationId);
    } else {
      window.localStorage.removeItem(ACTIVE_CHAT_KEY);
    }
  }

  if (state.uiState && typeof state.uiState === 'object') {
    writeLocal(UI_STATE_KEY, state.uiState);
  }

  let conversation = null;
  if (conversationId) {
    const conversationState = { ...state };
    delete conversationState.conversationId;
    delete conversationState.activeChatId;
    delete conversationState.metadata;
    conversation = saveConversationState(conversationId, conversationState);
  }

  return { session: nextSession, conversation };
}

export function listConversationHistory() {
  const conversations = getConversationsMap();
  return Object.values(conversations)
    .map((conversation) => normalizeConversation(conversation))
    .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
}

export function isGuestHydrating() {
  if (!hasSessionStorage()) return false;
  return window.sessionStorage.getItem(HYDRATION_KEY) === 'true';
}
