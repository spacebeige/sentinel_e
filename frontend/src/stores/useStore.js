/**
 * ============================================================
 * useStore.js — Authenticated User Session Store
 * ============================================================
 *
 * Persistence keys are user-scoped to prevent stale guest-state mixing.
 * Defensive guards block any legacy guest userId from being rehydrated
 * into an authenticated user's session slot.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import api from '../services/api';
import { getPersistenceUserId } from '../services/sessionPersistence';

async function fetchWithRetry(fetcher, retries = 3, baseDelayMs = 1500) {
  const shouldRetry = (error) => {
    const status = error?.status || error?.response?.status;
    if (status === 401 || status === 403 || status === 404) return false;
    return true;
  };

  let lastError;
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      return await fetcher();
    } catch (error) {
      lastError = error;
      if (!shouldRetry(error)) throw error;
      if (attempt < retries - 1) {
        const delay = baseDelayMs * (attempt + 1);
        // eslint-disable-next-line no-await-in-loop
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }
  throw lastError;
}

const useStore = create(
  persist(
    (set, get) => ({
      userId: null,
      chats: [],
      messages: [],
      memory: [],
      preferences: null,
      contextWindow: null,
      isLoaded: false,
      hasHydrated: false,
      isLoading: false,
      error: null,

      setUserId: (id) => set({ userId: id }),
      setLoading: (isLoading) => set({ isLoading }),

      setHistory: (chats, messages) => {
        const prev = get();
        const safeChats = Array.isArray(chats) ? chats : [];
        const safeMessages = Array.isArray(messages) ? messages : [];

        // Never blow away cached data with an empty response (cold start / transient failure)
        const nextChats = (safeChats.length === 0 && prev.chats.length > 0) ? prev.chats : safeChats;
        const nextMessages = (safeMessages.length === 0 && prev.messages.length > 0) ? prev.messages : safeMessages;

        set({ chats: nextChats, messages: nextMessages, isLoaded: true });
      },

      setChatsGuarded: (newChats) => {
        // PHASE 5: Guard against invalid overwrites
        if (!Array.isArray(newChats)) {
          console.warn('setChatsGuarded: Invalid chats array, skipping overwrite');
          return;
        }
        set({ chats: newChats });
      },

      setChats: (chats) => set({ chats }),
      setMessages: (messages) => set({ messages }),
      setMemory: (memory) => set({ memory }),
      setPreferences: (preferences) => set({ preferences }),
      setContextWindow: (contextWindow) => set({ contextWindow }),

      // Called once on app init when user is already signed in
      initializeSession: async () => {
        if (get().isLoaded || get().isLoading) return;
        await get().reloadHistory();
      },

      // Force reload from server (used on login / refresh)
      reloadHistory: async () => {
        if (get().isLoading) return;
        set({ isLoading: true, error: null });
        try {
          const prev = get();
          const results = await Promise.allSettled([
            fetchWithRetry(() => api.get('/api/history'), 4, 1500),
            api.get('/api/user/memory'),
            api.get('/api/user/preferences')
          ]);

          const historyResponse = results[0].status === 'fulfilled'
            ? (results[0].value ?? { chats: [], messages: [] })
            : { chats: [], messages: [] };
          const memoryData = results[1].status === 'fulfilled'
            ? (results[1].value ?? [])
            : [];
          const prefsData = results[2].status === 'fulfilled'
            ? (results[2].value ?? {})
            : {};

          // Always parse from response.chats / response.messages
          const chats = Array.isArray(historyResponse?.chats) ? historyResponse.chats : [];
          const messages = Array.isArray(historyResponse?.messages) ? historyResponse.messages : [];

          get().setHistory(chats, messages);

          set({
            memory: Array.isArray(memoryData) ? memoryData : [],
            preferences: prefsData || {},
            chats: get().chats.length > 0 ? get().chats : prev.chats,
            messages: get().messages.length > 0 ? get().messages : prev.messages,
            isLoaded: true,
            isLoading: false,
            error: null,
          });
        } catch (err) {
          console.error('Failed to load session:', err);
          set({ error: err.message, isLoading: false, isLoaded: true });
        }
      },

      addMessage: (message) => set((state) => ({
        messages: [...state.messages, message]
      })),

      addChat: (chat) => set((state) => ({
        chats: [chat, ...state.chats]
      })),

      updateMemory: (memory) => set({ memory }),

      clearSession: () => set({
        chats: [],
        messages: [],
        memory: [],
        preferences: null,
        contextWindow: null,
        isLoaded: false,
        isLoading: false,
        error: null,
      }),

      // Hard reset for user switch or logout
      resetForNewUser: () => get().clearSession(),

      reset: () => set({
        chats: [],
        messages: [],
        memory: [],
        preferences: null,
        contextWindow: null,
        isLoaded: false,
        hasHydrated: false,
        error: null
      })
    }),
    {
      // Persist key is stable per deployment — user ID is stored inside the state.
      // Authenticated users ALWAYS get fresh state from the server on login.
      name: 'sentinel-session',
      storage: {
        getItem: (name) => {
          const userId = getPersistenceUserId();
          if (!userId) return null;
          return localStorage.getItem(`${name}-${userId}`);
        },
        setItem: (name, value) => {
          const userId = getPersistenceUserId();
          if (!userId) return;
          localStorage.setItem(`${name}-${userId}`, value);
        },
        removeItem: (name) => {
          const userId = getPersistenceUserId();
          if (!userId) return;
          localStorage.removeItem(`${name}-${userId}`);
        }
      },
      onRehydrateStorage: () => (state) => {
        // Ensure loading is cleared even if state is undefined (Zustand edge case)
        if (state) state.setLoading(false);
        // hasHydrated must always be set — even on null state — to unblock ProtectedRoute
        useStore.setState({ hasHydrated: true });
      },
      partialize: (state) => ({
        userId: state.userId,
        chats: state.chats,
        messages: state.messages,
        memory: state.memory,
        preferences: state.preferences,
        isLoaded: state.isLoaded,
      }),
    }
  )
);

export default useStore;
