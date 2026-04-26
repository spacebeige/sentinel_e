import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import api from '../services/api';

const useStore = create(
  persist(
    (set, get) => ({
      chats: [],
      messages: [],
      memory: [],
      preferences: null,
      contextWindow: null,
      isInitialized: false,
      isLoading: false,
      error: null,

      setChats: (chats) => set({ chats }),
      setMessages: (messages) => set({ messages }),
      setMemory: (memory) => set({ memory }),
      setPreferences: (preferences) => set({ preferences }),
      setContextWindow: (contextWindow) => set({ contextWindow }),

      initializeSession: async () => {
        if (get().isInitialized || get().isLoading) return;
        
        set({ isLoading: true, error: null });
        try {
          const results = await Promise.allSettled([
            api.get('/api/history'),
            api.get('/api/user/memory'),
            api.get('/api/user/preferences')
          ]);

          const historyData = results[0].status === 'fulfilled' ? results[0].value.data.data : { chats: [], messages: [] };
          const memoryData = results[1].status === 'fulfilled' ? results[1].value.data.data : [];
          const prefsData = results[2].status === 'fulfilled' ? results[2].value.data.data : {};

          set({
            chats: historyData?.chats || [],
            messages: historyData?.messages || [],
            memory: Array.isArray(memoryData) ? memoryData : [],
            preferences: prefsData || {},
            isInitialized: true,
            isLoading: false
          });
        } catch (err) {
          console.error('Failed to initialize session:', err);
          set({ error: err.message, isLoading: false, isInitialized: true });
        }
      },

      addMessage: (message) => set((state) => ({ 
        messages: [...state.messages, message] 
      })),

      addChat: (chat) => set((state) => ({
        chats: [chat, ...state.chats]
      })),

      updateMemory: (memory) => set({ memory }),
      
      reset: () => set({
        chats: [],
        messages: [],
        memory: [],
        preferences: null,
        contextWindow: null,
        isInitialized: false,
        error: null
      })
    }),
    {
      name: 'sentinel-session-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ 
        chats: state.chats, 
        messages: state.messages, 
        memory: state.memory, 
        preferences: state.preferences 
      }),
    }
  )
);

export default useStore;
