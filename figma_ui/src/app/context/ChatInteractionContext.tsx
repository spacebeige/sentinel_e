import React, { createContext, useContext, useState, ReactNode } from 'react';

interface ChatInteractionContextType {
  isHistoryOpen: boolean;
  toggleHistory: () => void;
  activeSubMode: string | null;
  setActiveSubMode: (mode: string | null) => void;
  isProMode: boolean;
  setIsProMode: (isPro: boolean) => void;
  triggerNewChat: () => void;
  newChatTriggered: number; // Increment to trigger
}

const ChatInteractionContext = createContext<ChatInteractionContextType | undefined>(undefined);

export function ChatInteractionProvider({ children }: { children: ReactNode }) {
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [activeSubMode, setActiveSubMode] = useState<string | null>(null);
  const [isProMode, setIsProMode] = useState(false);
  const [newChatTriggered, setNewChatTriggered] = useState(0);

  const toggleHistory = () => setIsHistoryOpen((prev) => !prev);
  const triggerNewChat = () => setNewChatTriggered((prev) => prev + 1);

  return (
    <ChatInteractionContext.Provider
      value={{
        isHistoryOpen,
        toggleHistory,
        activeSubMode,
        setActiveSubMode,
        isProMode,
        setIsProMode,
        triggerNewChat,
        newChatTriggered,
      }}
    >
      {children}
    </ChatInteractionContext.Provider>
  );
}

export function useChatInteraction() {
  const context = useContext(ChatInteractionContext);
  if (context === undefined) {
    throw new Error('useChatInteraction must be used within a ChatInteractionProvider');
  }
  return context;
}
