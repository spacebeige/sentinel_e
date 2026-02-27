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
import axios from 'axios';
import { API_BASE } from '../config';
import FigmaChatShell, { MODELS } from '../figma_shell/FigmaChatShell';
import { getDefaultPipelineSteps } from '../engines/modeController';
import memoryManager from '../engines/memoryManager';
import { injectContext } from '../engines/contextInjector';
import { evaluateResponse } from '../engines/cognitiveGovernor';
import { useCognitiveStore } from '../stores/cognitiveStore';

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export default function ChatEngine() {
  const { addDebateResult } = useCognitiveStore();
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

  // === FIGMA SHELL BINDINGS ===
  const [input, setInput] = useState('');
  const [selectedModel, setSelectedModel] = useState(MODELS[0]);

  // Use API_BASE directly from config

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
      setSelectedModel(MODELS[4]); // sentinel-exp
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

  const checkHealth = useCallback(async () => {
    try {
      await axios.get(`${API_BASE}/`, { timeout: 3000 });
      setServerStatus('online');
    } catch {
      setServerStatus('offline');
    }
  }, [API_BASE_URL]);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get(`${API_BASE}/api/history`);
        const formatted = res.data.map(item => ({
          id: item.id,
          timestamp: item.updated_at || item.created_at || new Date().toISOString(),
          mode: item.mode,
          summary: item.chat_name || item.preview || 'Chat',
          filename: item.id,
          data: null,
        }));
        setHistory(formatted);
      } catch (err) {
        console.error('Failed to load history:', err);
      }
    };
    fetchHistory();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch descriptive session state for right panel
  useEffect(() => {
    if (!activeChatId) { setSessionState(null); return; }
    const fetchSession = async () => {
      try {
        const res = await axios.get(`${API_BASE}/api/session/${activeChatId}/descriptive`);
        if (res.data && !res.data.error) setSessionState(res.data);
      } catch {
        // Fall back to legacy endpoint
        try {
          const res = await axios.get(`${API_BASE}/api/omega/session/${activeChatId}`);
          if (res.data?.session_state) setSessionState(res.data.session_state);
        } catch { /* ignore */ }
      }
    };
    fetchSession();
  }, [activeChatId, API_BASE_URL]);

  const handleSend = async ({ text, file }) => {
    if (!text && !file) return;
    const chatId = activeChatId;
    setLoading(true);
    setShowLearning(false);
    setLastQueryText(text || '');

    const userMsg = { role: 'user', content: text || `[File: ${file?.name}]`, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);

    // Record user message in memory layer
    memoryManager.recordMessage(userMsg, mode, subMode);

    const formData = new FormData();
    if (text) formData.append('text', text);
    if (file) formData.append('file', file);
    if (chatId && UUID_REGEX.test(chatId)) formData.append('chat_id', chatId);

    // Inject stateful context (memory + preferences + adaptive params)
    injectContext(formData, text || '', mode, subMode);

    let endpoint;
    // ── COGNITIVE ENSEMBLE v7.0: Single endpoint, no mode routing ──
    endpoint = `${API_BASE}/run/ensemble`;
    formData.append('rounds', Math.max(rounds, 3));  // enforce minimum 3 rounds
    // No mode-based branching. All requests route through ensemble engine.

    try {
      const response = await axios.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const result = response.data;
      const returnedChatId = result.chat_id ? String(result.chat_id) : null;

      const answerText = result.formatted_output
        || result.data?.priority_answer
        || result.priority_answer
        || 'No response.';

      const assistantMsg = { role: 'assistant', content: answerText, timestamp: new Date().toISOString() };
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
      const res = await axios.get(`${API_BASE}/api/chat/${run.id}/messages`);
      const loaded = (res.data || [])
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => ({ role: m.role, content: m.content, timestamp: m.timestamp || run.timestamp }));
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
    />
  );
}
