import React, { useState, useEffect, useMemo } from 'react';

import FigmaChatShell, { MODELS } from '../figma_shell/FigmaChatShell';

import { useAuthContext } from '../hooks/useAuthContext';
import useStore from '../stores/useStore';
import {
  sendStandard,
  sendExperimental,
  sendKill,
} from '../services/api';



export default function ChatEngine() {
  useAuthContext();
  const { 
    chats, 
    messages, 
    addMessage, 
    addChat, 
    isLoading: storeLoading 
  } = useStore();

  const [mode, setMode] = useState('standard');
  const [subMode, setSubMode] = useState(null);
  const [killActive, setKillActive] = useState(false);
  const [rounds] = useState(3);
  const [activeChatId, setActiveChatId] = useState(null);
  const [currentResult, setCurrentResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [serverStatus] = useState('unknown');
  const [sessionState] = useState(null);
  const [input, setInput] = useState('');
  const [selectedModel, setSelectedModel] = useState(MODELS[0]);

  // Derived state: filter messages for active chat
  const activeMessages = useMemo(() => {
    if (!activeChatId) return [];
    return messages.filter(m => m.chat_id === activeChatId);
  }, [messages, activeChatId]);

  // Derived state: format history for UI
  const history = useMemo(() => {
    return (chats || []).map(item => ({
      id: item.id,
      timestamp: item.updated_at || item.created_at,
      mode: item.mode,
      summary: item.chat_name || item.preview || 'Chat',
      name: item.chat_name,
      filename: item.id,
    }));
  }, [chats]);

  // Model routing logic
  useEffect(() => {
    if (selectedModel.category === 'standard' && mode !== 'standard') {
      setMode('standard');
    } else if (selectedModel.category === 'experimental' && mode !== 'experimental') {
      setMode('experimental');
    }
  }, [selectedModel, mode]);

  const handleSend = async (text, attachments = []) => {
    if (!text && attachments.length === 0) return;
    setLoading(true);
    
    const userMsg = {
      id: Math.random().toString(36).substr(2, 9),
      chat_id: activeChatId,
      role: 'user',
      content: text,
      created_at: new Date().toISOString()
    };
    addMessage(userMsg);
    setInput('');

    try {
      let result;
      if (killActive) {
        result = await sendKill(text, activeChatId);
      } else if (mode === 'standard') {
        result = await sendStandard(text, activeChatId, selectedModel.id);
      } else {
        result = await sendExperimental(text, activeChatId, subMode, rounds);
      }

      const assistantMsg = {
        id: result.message_id || Math.random().toString(36).substr(2, 9),
        chat_id: result.chat_id,
        role: 'assistant',
        content: result.formatted_output,
        created_at: new Date().toISOString()
      };
      addMessage(assistantMsg);
      setCurrentResult(result);
      
      if (!activeChatId) {
        setActiveChatId(result.chat_id);
        addChat({
          id: result.chat_id,
          chat_name: text.substring(0, 30),
          mode,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        });
      }
    } catch (err) {
      console.error('Send failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectRun = (run) => {
    setActiveChatId(run.id);
    setMode(run.mode === 'standard' ? 'standard' : 'experimental');
  };

  const handleNewChat = () => {
    setActiveChatId(null);
    setCurrentResult(null);
    setKillActive(false);
  };

  return (
    <FigmaChatShell
      input={input}
      setInput={setInput}
      selectedModel={selectedModel}
      setSelectedModel={setSelectedModel}
      loading={loading || storeLoading}
      response={currentResult}
      handleSubmit={handleSend}
      messages={activeMessages}
      serverStatus={serverStatus}
      activeChatId={activeChatId}
      history={history}
      sessionState={sessionState}
      mode={mode}
      setMode={setMode}
      subMode={subMode}
      setSubMode={setSubMode}
      killActive={killActive}
      setKillActive={setKillActive}
      handleNewChat={handleNewChat}
      handleSelectRun={handleSelectRun}
    />
  );
}
