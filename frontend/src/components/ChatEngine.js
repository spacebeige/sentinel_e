/**
 * ============================================================
 * ChatEngine — Logic Authority (extracted from App.js)
 * ============================================================
 *
 * This component owns ALL backend logic:
 *   - State management (messages, mode, subMode, history, etc.)
 *   - API calls (handleSend, fetchHistory, checkHealth, etc.)
 *   - Model routing (standard / experimental / omega kill)
 *
 * It renders the FigmaChatShell as a controlled visual component.
 *
 * DO NOT MODIFY:
 *   - API clients
 *   - Model routing logic
 *   - handleSend / handleSelectRun
 *   - Inference parameters
 *   - Debate engine logic
 *
 * ============================================================
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { API_BASE } from '../config';
import FigmaChatShell, { MODELS } from '../figma_shell/FigmaChatShell';
import { getDefaultPipelineSteps } from '../engines/modeController';
import memoryManager from '../engines/memoryManager';
import { injectContext } from '../engines/contextInjector';
import { evaluateResponse } from '../engines/cognitiveGovernor';
import { useCognitiveStore } from '../stores/cognitiveStore';
import { useAuthContext } from '../hooks/useAuthContext';
import {
  getHistory,
  getChatMessages,
  getSessionDescriptive,
  getOmegaSession,
  checkHealth as apiCheckHealth,
  sendStandard,
  sendExperimental,
  sendKill,
} from '../services/api';

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export default function ChatEngine() {
  const { addDebateResult } = useCognitiveStore();
  const { isAuthenticated, syncedUser } = useAuthContext();

  const [mode, setMode] = useState('standard');
  const [subMode, setSubMode] = useState(null);
  const [killActive, setKillActive] = useState(false);
  const [rounds, setRounds] = useState(3);
  const [history, setHistory] = useState([]);
  const [messages, setMessages] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [currentResult, setCurrentResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showLearning, setShowLearning] = useState(false);
  const [serverStatus, setServerStatus] = useState('unknown');
  const [sessionState, setSessionState] = useState(null);
  const [lastResponseText, setLastResponseText] = useState('');
  const [lastQueryText, setLastQueryText] = useState('');
  const [governanceVerdict, setGovernanceVerdict] = useState(null);

  // === Dynamic model registry ===
  const [chatModels, setChatModels] = useState([]);
  const [mcoModels, setMcoModels] = useState([]);

  // === FIGMA SHELL BINDINGS ===
  const [input, setInput] = useState('');
  const [selectedModel, setSelectedModel] = useState(MODELS[0]);

  // Sync selectedModel.category ↔ mode (bidirectional adapter)
  useEffect(() => {
    if (selectedModel.category === 'standard' && mode !== 'standard') {
      setMode('standard');
    } else if (selectedModel.category === 'experimental' && mode !== 'experimental') {
      setMode('experimental');
    }
  }, [selectedModel]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (mode === 'standard' && selectedModel.category !== 'standard') {
      setSelectedModel(MODELS[0]); // sentinel-std
    } else if (mode === 'experimental' && selectedModel.category !== 'experimental') {
      setSelectedModel(MODELS[1]); // sentinel-exp
    }
  }, [mode]); // eslint-disable-line react-hooks/exhaustive-deps

  // === Mode isolation: subMode only applies to experimental ===
  useEffect(() => {
    if (selectedModel.category === 'standard') {
      setSubMode(null); // Standard mode NEVER has a subMode
    } else if (selectedModel.category === 'experimental' && !subMode) {
      setSubMode('debate'); // Default experimental subMode
    }
  }, [selectedModel]); // eslint-disable-line react-hooks/exhaustive-deps

  // Pipeline steps for ThinkingAnimation (tied to current mode)
  const pipelineSteps = useMemo(() => getDefaultPipelineSteps(mode, subMode), [mode, subMode]);

  // === Health Check (uses auth-aware api service) ===
  const checkHealthCb = useCallback(async () => {
    const status = await apiCheckHealth();
    setServerStatus(status);
  }, []);

  useEffect(() => {
    checkHealthCb();
    const interval = setInterval(checkHealthCb, 30000);
    return () => clearInterval(interval);
  }, [checkHealthCb]);

  // === Fetch model registry ===
  useEffect(() => {
    if (serverStatus !== 'online' || !isAuthenticated) return;
    const fetchModels = async () => {
      try {
        const { default: api } = await import('../services/api');
        // Use the default api instance which has auth interceptors
        const res = await api.get('/api/models');
        const models = (res.data?.models || []).map(m => ({
          id: m.id,
          name: m.name,
          provider: m.provider,
          role: m.role,
          tier: m.tier,
          enabled: m.enabled,
          color: m.role === 'anchor' ? '#3b82f6' : m.role === 'debate' ? '#ef4444' : '#10b981',
          category: 'individual',
          synthesis_only: m.synthesis_only,
          active: m.enabled,
          context_window: m.context_window,
          max_output_tokens: m.max_output_tokens,
        }));
        setChatModels(models);
      } catch { /* ignore — models are optional */ }
    };
    fetchModels();
  }, [serverStatus, isAuthenticated]);

  // === Claude toggle handler ===
  const handleToggleClaude = useCallback(async () => {
    try {
      const { default: api } = await import('../services/api');
      const res = await api.post('/api/models/claude/toggle');
      if (res.data) {
        setChatModels(prev => prev.map(m =>
          m.id === 'claude-sonnet-4.6' ? { ...m, active: res.data.active } : m
        ));
      }
    } catch (err) {
      console.error('Claude toggle failed:', err);
    }
  }, []);

  // ============================================================
  // CHAT HISTORY — Auth-aware, reloads on login & page refresh
  // Uses the api service which attaches Clerk JWT tokens.
  // ============================================================
  useEffect(() => {
    if (!isAuthenticated) {
      setHistory([]);
      return;
    }

    const fetchHistoryData = async () => {
      try {
        const data = await getHistory(50, 0);
        const formatted = (data || []).map(item => ({
          id: item.id,
          timestamp: item.updated_at || item.created_at || new Date().toISOString(),
          mode: item.mode,
          sub_mode: item.sub_mode || null,
          summary: item.chat_name || item.preview || 'Chat',
          name: item.chat_name,
          filename: item.id,
          data: null,
        }));
        setHistory(formatted);
      } catch (err) {
        console.error('Failed to load history:', err);
      }
    };
    fetchHistoryData();
  }, [isAuthenticated, syncedUser?.id]); // Re-fetch when auth state changes

  // Fetch descriptive session state for right panel
  useEffect(() => {
    if (!activeChatId || !isAuthenticated) { setSessionState(null); return; }
    const fetchSession = async () => {
      try {
        const data = await getSessionDescriptive(activeChatId);
        if (data && !data.error) setSessionState(data);
      } catch {
        // Fall back to legacy endpoint
        try {
          const data = await getOmegaSession(activeChatId);
          if (data?.session_state) setSessionState(data.session_state);
        } catch { /* ignore */ }
      }
    };
    fetchSession();
  }, [activeChatId, isAuthenticated]);

  const handleSend = async ({ text, file }) => {
    if (!text && !file) return;
    const chatId = activeChatId;
    setLoading(true);
    setShowLearning(false);
    setLastQueryText(text || '');

    const userMsg = { role: 'user', content: text || `[File: ${file?.name}]`, timestamp: new Date().toISOString() };
    if (file && file.type?.startsWith('image/')) {
      try {
        const reader = new FileReader();
        const dataUrl = await new Promise((resolve) => {
          reader.onload = (e) => resolve(e.target.result);
          reader.readAsDataURL(file);
        });
        const b64 = dataUrl.split(',')[1];
        userMsg.image_b64 = b64;
        userMsg.image_mime = file.type;
      } catch { /* ignore preview failure */ }
    }
    setMessages(prev => [...prev, userMsg]);

    // Record user message in memory layer
    memoryManager.recordMessage(userMsg, mode, subMode);

    try {
      let result;

      if (killActive && chatId) {
        result = await sendKill(text, chatId);
      } else if (mode === 'experimental') {
        result = await sendExperimental(text, {
          chatId, file, 
          context: null,
          mode: 'experimental',
          subMode: subMode || 'debate',
          rounds: Math.max(rounds, 3),
          killSwitch: false,
        });
      } else if (mode === 'ensemble') {
        // Use the api service for ensemble
        const { default: api } = await import('../services/api');
        const formData = new FormData();
        formData.append('text', text);
        formData.append('rounds', Math.max(rounds, 3));
        if (chatId) formData.append('chat_id', chatId);
        if (file) formData.append('file', file);
        injectContext(formData, text || '', mode, subMode);
        const res = await api.post('/run/ensemble', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        result = res.data;
      } else {
        result = await sendStandard(text, chatId, file, null);
      }

      const returnedChatId = result.chat_id ? String(result.chat_id) : null;

      const answerText = result.formatted_output
        || result.data?.priority_answer
        || result.priority_answer
        || 'No response.';

      const assistantMsg = {
        role: 'assistant',
        content: answerText,
        timestamp: new Date().toISOString(),
        reasoning_json: result.omega_metadata || null,
      };
      setMessages(prev => [...prev, assistantMsg]);
      setCurrentResult(result);
      setLastResponseText(answerText);
      if (result.session_state) setSessionState(result.session_state);

      // Record assistant response in memory layer
      memoryManager.recordMessage(assistantMsg, mode, subMode);
      memoryManager.recordAnalytics(result);

      // Pipe ensemble results into global cognitive store (v7.0)
      addDebateResult(result);

      // Self-governance evaluation (Section XII)
      const verdict = evaluateResponse({
        userQuery: text || '',
        responseText: answerText,
        responseData: result,
        mode, subMode,
      });
      setGovernanceVerdict(verdict);
      if (verdict.flags.length > 0) {
        console.debug('[CognitiveGovernor]', verdict.flags, verdict.suggestions);
      }

      if (returnedChatId && UUID_REGEX.test(returnedChatId)) {
        setActiveChatId(returnedChatId);
      }

      const effectiveChatId = chatId || returnedChatId;
      if (effectiveChatId && UUID_REGEX.test(effectiveChatId)) {
        setHistory(prev => {
          const exists = prev.some(item => item.id === effectiveChatId);
          if (exists) {
            return prev.map(item =>
              item.id === effectiveChatId
                ? { ...item, timestamp: new Date().toISOString(), summary: text ? text.substring(0, 40) : item.summary, data: result }
                : item
            );
          }
          return [{ id: effectiveChatId, timestamp: new Date().toISOString(), mode, summary: text ? text.substring(0, 40) : 'Chat', filename: effectiveChatId, data: result }, ...prev];
        });
      }

      setServerStatus('online');
    } catch (error) {
      console.error(error);
      setServerStatus('offline');
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  const handleSelectRun = async (run) => {
    setShowLearning(false);
    if (run.mode === 'standard' || run.mode === 'conversational') {
      setMode('standard');
    } else {
      setMode('experimental');
    }
    if (run.sub_mode) setSubMode(run.sub_mode);
    setActiveChatId(run.id);
    setMessages([]);
    setCurrentResult(null);
    setLoading(true);
    try {
      const data = await getChatMessages(run.id);
      const loaded = (data || [])
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => ({
          role: m.role,
          content: m.content,
          timestamp: m.timestamp || run.timestamp,
          image_b64: m.image_b64 || null,
          image_mime: m.image_mime || null,
          reasoning_json: m.reasoning_json || null,
        }));
      setMessages(loaded);
    } catch (err) {
      console.error('Failed to load messages:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleNewChat = () => {
    setActiveChatId(null);
    setMessages([]);
    setCurrentResult(null);
    setShowLearning(false);
    setSessionState(null);
    setKillActive(false);
    setLastResponseText('');
    setLastQueryText('');
    // Reset short-term memory for new session (preserves analytical + profile)
    memoryManager.newSession();
  };

  return (
    <FigmaChatShell
      input={input}
      setInput={setInput}
      selectedModel={selectedModel}
      setSelectedModel={setSelectedModel}
      loading={loading}
      response={currentResult}
      handleSubmit={handleSend}
      messages={messages}
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
      onNewChat={handleNewChat}
      onSelectRun={handleSelectRun}
      pipelineSteps={pipelineSteps}
      lastQueryText={lastQueryText}
      lastResponseText={lastResponseText}
      governanceVerdict={governanceVerdict}
      chatModels={chatModels}
      mcoModels={mcoModels}
      onToggleClaude={handleToggleClaude}
    />
  );
}
