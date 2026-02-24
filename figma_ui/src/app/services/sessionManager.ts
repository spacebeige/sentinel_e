// ============================================================
// Session Manager — localStorage + in-memory state
// Persists session across refreshes; backend remains authoritative
// ============================================================

const STORAGE_KEY = "sentinel_e_session";

export interface PersistedSession {
  chatId: string | null;
  mode: string;
  subMode: string | null;
  selectedModelId: string;
  /** Debate rounds snapshot (so we don't lose rounds on refresh) */
  debateRounds: DebateRoundSnapshot[];
  /** Glass kill_override state */
  killOverride: boolean;
  /** Timestamp of last persistence */
  savedAt: string;
}

export interface DebateRoundSnapshot {
  round: number;
  positions: Array<{
    model: string;
    position: string;
    confidence: number;
    key_points: string[];
  }>;
  consensus?: string;
}

const DEFAULT_SESSION: PersistedSession = {
  chatId: null,
  mode: "standard",
  subMode: null,
  selectedModelId: "sentinel-std",
  debateRounds: [],
  killOverride: false,
  savedAt: new Date().toISOString(),
};

/**
 * Read persisted session from localStorage. Returns defaults if missing / corrupt.
 */
export function loadSession(): PersistedSession {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_SESSION };
    const parsed = JSON.parse(raw) as Partial<PersistedSession>;
    return {
      chatId: parsed.chatId ?? null,
      mode: parsed.mode ?? "standard",
      subMode: parsed.subMode ?? null,
      selectedModelId: parsed.selectedModelId ?? "sentinel-std",
      debateRounds: Array.isArray(parsed.debateRounds) ? parsed.debateRounds : [],
      killOverride: Boolean(parsed.killOverride),
      savedAt: parsed.savedAt ?? new Date().toISOString(),
    };
  } catch {
    return { ...DEFAULT_SESSION };
  }
}

/**
 * Persist session to localStorage.
 */
export function saveSession(session: Partial<PersistedSession>): void {
  try {
    const current = loadSession();
    const merged: PersistedSession = {
      ...current,
      ...session,
      savedAt: new Date().toISOString(),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
  } catch {
    // Storage full or unavailable — silently degrade
  }
}

/**
 * Clear persisted session (e.g. new chat).
 */
export function clearSession(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // noop
  }
}

/**
 * Save only the chatId (most common operation).
 */
export function saveChatId(chatId: string | null): void {
  saveSession({ chatId });
}

/**
 * Store debate round snapshot.
 */
export function appendDebateRound(round: DebateRoundSnapshot): void {
  const current = loadSession();
  const rounds = [...current.debateRounds, round];
  saveSession({ debateRounds: rounds });
}

/**
 * Clear debate rounds (new topic or mode change).
 */
export function clearDebateRounds(): void {
  saveSession({ debateRounds: [] });
}
