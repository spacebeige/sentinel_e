// ============================================================
// useSessionPersistence — auto-save / auto-restore session
// ============================================================

import { useEffect, useCallback, useRef } from "react";
import {
  loadSession,
  saveSession,
  clearSession,
  type PersistedSession,
} from "../services/sessionManager";

/**
 * Returns helpers to persist and restore ChatPage session state.
 */
export function useSessionPersistence() {
  const initialized = useRef(false);

  /** Load persisted session on mount (call once). */
  const restore = useCallback((): PersistedSession => {
    return loadSession();
  }, []);

  /**
   * Persist key fields. Call on every meaningful state change.
   * Debounced internally to avoid excess writes.
   */
  const persist = useCallback(
    (partial: Partial<PersistedSession>) => {
      saveSession(partial);
    },
    []
  );

  /** Reset persisted session (new chat). */
  const reset = useCallback(() => {
    clearSession();
  }, []);

  // Mark as initialized after first render
  useEffect(() => {
    initialized.current = true;
  }, []);

  return { restore, persist, reset, initialized: initialized.current };
}
