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
import FigmaChatShell from '../figma_shell/FigmaChatShell';
import useModels from '../hooks/useModels';
import { getDefaultPipelineSteps } from '../engines/modeController';
import memoryManager from '../engines/memoryManager';
import { evaluateResponse } from '../engines/cognitiveGovernor';
import {
  initSession, checkHealth as apiCheckHealth,
  sendMCOQuery, sendDirectModelQuery,
  getHistory, getChatMessages, getSessionDescriptive, getOmegaSession,
} from '../services/api';

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export default function ChatEngineV5() {
  const { chatModels, mcoModels } = useModels();
  const [mode, setMode] = useState('standard');
  const [subMode, setSubMode] = useState(null);
  const [killActive, setKillActive] = useState(false);
  const [history, setHistory] = useState([]);
  const [messages, setMessages] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [currentResult, setCurrentResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [, setShowLearning] = useState(false);
  const [serverStatus, setServerStatus] = useState('unknown');
  const [sessionState, setSessionState] = useState(null);
  const [lastResponseText, setLastResponseText] = useState('');
  const [lastQueryText, setLastQueryText] = useState('');
  const [governanceVerdict, setGovernanceVerdict] = useState(null);
  const [error, setError] = useState(null);

  const [input, setInput] = useState('');
  // Default to Sentinel Standard aggregate mode (not an individual model)
  const SENTINEL_STD = { id: 'sentinel-std', name: 'Sentinel-E Standard', provider: 'Aggregated', color: '#3b82f6', category: 'standard', isMeta: true, enabled: true };
  const [selectedModel, setSelectedModel] = useState(SENTINEL_STD);

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
  // Individual models (tier-based, no category) always run in standard mode.
  // Meta modes: sentinel-std → standard, sentinel-exp → experimental.
  useEffect(() => {
    const cat = selectedModel.category;
    const isIndividualModel = !cat && selectedModel.tier;
    if (isIndividualModel && mode !== 'standard') {
      setMode('standard');
    } else if (cat === 'standard' && mode !== 'standard') {
      setMode('standard');
    } else if (cat === 'experimental' && mode !== 'experimental') {
      setMode('experimental');
    }
  }, [selectedModel, mode]);

  useEffect(() => {
    // Only force model switch when toggling between meta modes.
    // Do NOT override an individually-selected model.
    const isIndividualModel = !selectedModel.category && selectedModel.tier;
    if (isIndividualModel) return; // individual model — no override
    if (mode === 'standard' && selectedModel.category !== 'standard') {
      const stdModel = chatModels.find(m => m.category === 'standard') || chatModels[0];
      if (stdModel) setSelectedModel(stdModel);
    } else if (mode === 'experimental' && selectedModel.category !== 'experimental') {
      const expModel = chatModels.find(m => m.category === 'experimental') || chatModels[chatModels.length - 1];
      if (expModel) setSelectedModel(expModel);
    }
  }, [mode, selectedModel, chatModels]);

  useEffect(() => {
    const cat = selectedModel.category;
    const isIndividualModel = !cat && selectedModel.tier;
    if (isIndividualModel || cat === 'standard') {
      setSubMode(null);
    } else if (cat === 'experimental' && !subMode) {
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

    try {
      let result;

      // Determine if single model focus mode
      const isSingleModel = selectedModel && !selectedModel.isMeta && selectedModel.id !== 'sentinel-std' && selectedModel.id !== 'sentinel-exp';

      if (isSingleModel) {
        // Single Model Focus: route directly to /chat/{model_id}
        result = await sendDirectModelQuery(selectedModel.id, text, chatId);
      } else if (mode === 'experimental') {
        // ALL experimental sub-modes (debate, evidence, glass, kill) → MCO
        result = await sendMCOQuery(text, {
          chatId,
          mode: 'experimental',
          subMode: (subMode === 'glass' && killActive) ? 'glass' : subMode,
        });
      } else {
        // Standard mode → MCO
        result = await sendMCOQuery(text, {
          chatId,
          mode: 'standard',
        });
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
      chatModels={chatModels}
      mcoModels={mcoModels}
    />
  );
}
