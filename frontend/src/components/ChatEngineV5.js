/**
 * ============================================================
 * ChatEngine v5 — Secure Logic Authority
 * ============================================================
 * 
 * SECURITY CHANGES from v4:
 *   - All API calls through services/api.js (JWT auth)
 *   - No API keys in frontend
 *   - No system prompt exposure
 *   - No model routing logic client-side
 *   - No internal state exposed via dev tools
 *   - Presentation layer only
 * 
 * MODE ISOLATION:
 *   - Single Model Mode: shows only model output
 *   - Standard: shows output + basic confidence
 *   - Experimental: shows output + full analytics (collapsible)
 *   - Advanced diagnostics hidden behind developer mode
 * ============================================================
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import FigmaChatShell, { MODELS } from '../figma_shell/FigmaChatShell';
import { getDefaultPipelineSteps } from '../engines/modeController';
import memoryManager from '../engines/memoryManager';
import { buildContextPayload } from '../engines/contextInjector';
import { evaluateResponse } from '../engines/cognitiveGovernor';
import {
  initSession, checkHealth as apiCheckHealth,
  sendStandard, sendExperimental, sendKill,
  getHistory, getChatMessages, getSessionDescriptive, getOmegaSession,
} from '../services/api';

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export default function ChatEngineV5() {
  const [mode, setMode] = useState('standard');
  const [subMode, setSubMode] = useState(null);
  const [killActive, setKillActive] = useState(false);
  const [rounds] = useState(3);
  const [history, setHistory] = useState([]);
  const [messages, setMessages] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [currentResult, setCurrentResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showLearning] = useState(false);
  const [serverStatus, setServerStatus] = useState('unknown');
  const [sessionState, setSessionState] = useState(null);
  const [lastResponseText, setLastResponseText] = useState('');
  const [lastQueryText, setLastQueryText] = useState('');
  const [governanceVerdict, setGovernanceVerdict] = useState(null);
  const [error, setError] = useState(null);

  const [input, setInput] = useState('');
  const [selectedModel, setSelectedModel] = useState(MODELS[0]);

  // ── Session Bootstrap ────────────────────────────────────
  useEffect(() => {
    const bootstrap = async () => {
      try {
        await initSession();
      } catch {
        // Will work without auth in dev mode
      }
    };
    bootstrap();
  }, []);

  // ── Mode Sync ────────────────────────────────────────────
  useEffect(() => {
    if (selectedModel.category === 'standard' && mode !== 'standard') {
      setMode('standard');
    } else if (selectedModel.category === 'experimental' && mode !== 'experimental') {
      setMode('experimental');
    }
  }, [selectedModel, mode]);

  useEffect(() => {
    if (mode === 'standard' && selectedModel.category !== 'standard') {
      setSelectedModel(MODELS[0]);
    } else if (mode === 'experimental' && selectedModel.category !== 'experimental') {
      setSelectedModel(MODELS[4]);
    }
  }, [mode, selectedModel]);

  useEffect(() => {
    if (selectedModel.category === 'standard') {
      setSubMode(null);
    } else if (selectedModel.category === 'experimental' && !subMode) {
      setSubMode('debate');
    }
  }, [selectedModel, subMode]);

  const pipelineSteps = useMemo(
    () => getDefaultPipelineSteps(mode, subMode),
    [mode, subMode]
  );

  // ── Health Check ─────────────────────────────────────────
  const checkHealth = useCallback(async () => {
    const status = await apiCheckHealth();
    setServerStatus(status);
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  // ── History Loading ──────────────────────────────────────
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const data = await getHistory();
        const formatted = data.map(item => ({
          id: item.id,
          timestamp: item.updated_at || item.created_at || new Date().toISOString(),
          mode: item.mode,
          summary: item.chat_name || item.preview || 'Chat',
          filename: item.id,
          data: null,
        }));
        setHistory(formatted);
      } catch (err) {
        // Silent fail on history load
      }
    };
    fetchHistory();
  }, []);

  // ── Session State ────────────────────────────────────────
  useEffect(() => {
    if (!activeChatId) { setSessionState(null); return; }
    const fetchSession = async () => {
      try {
        const data = await getSessionDescriptive(activeChatId);
        if (data && !data.error) setSessionState(data);
      } catch {
        try {
          const data = await getOmegaSession(activeChatId);
          if (data?.session_state) setSessionState(data.session_state);
        } catch { /* silent */ }
      }
    };
    fetchSession();
  }, [activeChatId]);

  // ── Send Handler ─────────────────────────────────────────
  const handleSend = async ({ text, file }) => {
    if (!text && !file) return;
    const chatId = activeChatId;
    setLoading(true);
    setError(null);
    setShowLearning(false);
    setLastQueryText(text || '');

    const userMsg = {
      role: 'user',
      content: text || `[File: ${file?.name}]`,
      timestamp: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMsg]);
    memoryManager.recordMessage(userMsg, mode, subMode);

    // Build context payload (sanitized client-side memory)
    const context = buildContextPayload(text || '', mode, subMode);

    try {
      let result;

      if (mode === 'experimental' && subMode === 'glass' && killActive) {
        result = await sendKill(text, chatId);
      } else if (mode === 'experimental') {
        result = await sendExperimental(text, {
          chatId, file, context, mode: 'experimental',
          subMode, rounds,
        });
      } else {
        result = await sendStandard(text, chatId, file, context);
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
      };
      setMessages(prev => [...prev, assistantMsg]);
      setCurrentResult(result);
      setLastResponseText(answerText);
      if (result.session_state) setSessionState(result.session_state);

      memoryManager.recordMessage(assistantMsg, mode, subMode);
      memoryManager.recordAnalytics(result);

      const verdict = evaluateResponse({
        userQuery: text || '',
        responseText: answerText,
        responseData: result,
        mode, subMode,
      });
      setGovernanceVerdict(verdict);

      if (returnedChatId && UUID_REGEX.test(returnedChatId)) {
        setActiveChatId(returnedChatId);
      }

      // Update history
      const effectiveChatId = chatId || returnedChatId;
      if (effectiveChatId && UUID_REGEX.test(effectiveChatId)) {
        setHistory(prev => {
          const exists = prev.some(item => item.id === effectiveChatId);
          if (exists) {
            return prev.map(item =>
              item.id === effectiveChatId
                ? { ...item, timestamp: new Date().toISOString(), summary: text ? text.substring(0, 40) : item.summary }
                : item
            );
          }
          return [{
            id: effectiveChatId,
            timestamp: new Date().toISOString(),
            mode,
            summary: text ? text.substring(0, 40) : 'Chat',
            filename: effectiveChatId,
            data: null,
          }, ...prev];
        });
      }

      setServerStatus('online');
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.');
      setMessages(prev => prev.slice(0, -1)); // Remove optimistic user msg
      if (err.message?.includes('Unable to reach')) {
        setServerStatus('offline');
      }
    } finally {
      setLoading(false);
    }
  };

  // ── Select Run ───────────────────────────────────────────
  const handleSelectRun = async (run) => {
    setShowLearning(false);
    setError(null);
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
        }));
      setMessages(loaded);
    } catch (err) {
      setError('Failed to load chat history.');
    } finally {
      setLoading(false);
    }
  };

  // ── New Chat ─────────────────────────────────────────────
  const handleNewChat = () => {
    setActiveChatId(null);
    setMessages([]);
    setCurrentResult(null);
    setShowLearning(false);
    setSessionState(null);
    setKillActive(false);
    setLastResponseText('');
    setLastQueryText('');
    setError(null);
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
      error={error}
    />
  );
}
