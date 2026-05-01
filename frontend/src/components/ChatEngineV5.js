/**
 * ============================================================
 * ChatEngine v5 — Secure Logic Authority
 * ============================================================
 * 
 * SECURITY CHANGES from v4:
 *   - All API calls through services/api.js (session auth)
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
import { auth } from '../firebase';
import FigmaChatShell from '../figma_shell/FigmaChatShell';
import useModels from '../hooks/useModels';
import { getDefaultPipelineSteps } from '../engines/modeController';
import memoryManager from '../engines/memoryManager';
import { evaluateResponse } from '../engines/cognitiveGovernor';
import {
  checkHealth as apiCheckHealth,
  sendMCOQuery, sendDirectModelQuery,
  getChatMessages, getSessionDescriptive, getOmegaSession,
} from '../services/api';
import useStore from '../stores/useStore';
import { validateResponseShape, Schemas } from '../utils/validation';

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export default function ChatEngineV5() {
  const { chatModels, mcoModels, toggleClaude: onToggleClaude } = useModels();
  const [mode, setMode] = useState('standard');
  const [subMode, setSubMode] = useState(null);
  const [killActive, setKillActive] = useState(false);
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
  const persistedChats = useStore((state) => state.chats);
  const persistedMessages = useStore((state) => state.messages);
  const historyLoading = useStore((state) => state.isLoading);
  const reloadHistory = useStore((state) => state.reloadHistory);

  const history = useMemo(
    () => (persistedChats || []).map((item) => ({
      id: item.id,
      timestamp: item.updated_at || item.created_at || new Date().toISOString(),
      mode: item.mode,
      summary: item.chat_name || item.preview || 'Chat',
      filename: item.id,
      data: null,
      sub_mode: item.sub_mode || null,
    })),
    [persistedChats]
  );
  // Default to Sentinel Standard aggregate mode (not an individual model)
  const SENTINEL_STD = { id: 'sentinel-std', name: 'Sentinel-E Standard', provider: 'Aggregated', color: '#3b82f6', category: 'standard', isMeta: true, enabled: true };
  const [selectedModel, setSelectedModel] = useState(SENTINEL_STD);

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

  useEffect(() => {
    const savedChatId = localStorage.getItem('sentinel-active-chat-id');
    if (savedChatId && UUID_REGEX.test(savedChatId)) {
      setActiveChatId(savedChatId);
    }
  }, []);

  useEffect(() => {
    if (activeChatId) {
      localStorage.setItem('sentinel-active-chat-id', activeChatId);
    } else {
      localStorage.removeItem('sentinel-active-chat-id');
    }
  }, [activeChatId]);

  useEffect(() => {
    if (!activeChatId || messages.length > 0) return;
    const cached = (persistedMessages || [])
      .filter((m) => String(m.chat_id) === String(activeChatId))
      .map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        timestamp: m.created_at || new Date().toISOString(),
        image_b64: m.image_b64 || null,
        image_mime: m.image_mime || null,
        reasoning_json: m.reasoning_json || null,
      }));

    if (cached.length > 0) {
      setMessages(cached);
    }
  }, [activeChatId, persistedMessages, messages.length]);

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
    // PHASE 1: Log current user id for diagnostics
    try {
      console.log('FRONTEND: SEND - USER_ID', auth.currentUser?.uid || null);
    } catch (e) {
      /* ignore */
    }

    // PATCH 3: Block send if no userId
    if (!auth.currentUser?.uid) {
      console.error('NO USER_ID — blocking send');
      setError('Not authenticated. Please sign in.');
      return;
    }

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
    if (file && (file.type?.startsWith('image/') || file.type === 'application/pdf')) {
      try {
        const reader = new FileReader();
        const dataUrl = await new Promise((resolve) => {
          reader.onload = (e) => resolve(e.target.result);
          reader.readAsDataURL(file);
        });
        userMsg.image_b64 = dataUrl.split(',')[1];
        userMsg.image_mime = file.type;
        if (file.type === 'application/pdf') {
          userMsg.pdf_filename = file.name;
        }
      } catch { /* ignore preview failure */ }
    }
    setMessages(prev => [...prev, userMsg]);
    memoryManager.recordMessage(userMsg, mode, subMode);

    try {
      let result;

      // Determine if single model focus mode
      const isSingleModel = selectedModel && !selectedModel.isMeta && selectedModel.id !== 'sentinel-std' && selectedModel.id !== 'sentinel-exp';

      if (isSingleModel) {
        // Single Model Focus: route directly to /chat/{model_id}
        result = await sendDirectModelQuery(selectedModel.id, text, chatId, {
          image_b64: userMsg.image_b64 || null,
          image_mime: userMsg.image_mime || null,
        });
      } else if (mode === 'experimental') {
        // ALL experimental sub-modes (debate, evidence, glass, kill) → MCO
        result = await sendMCOQuery(text, {
          chatId,
          mode: 'experimental',
          subMode: (subMode === 'glass' && killActive) ? 'glass' : subMode,
          image_b64: userMsg.image_b64 || null,
          image_mime: userMsg.image_mime || null,
        });
      } else {
        // Standard mode → MCO
        result = await sendMCOQuery(text, {
          chatId,
          mode: 'standard',
          image_b64: userMsg.image_b64 || null,
          image_mime: userMsg.image_mime || null,
        });
      }

      const returnedChatId = result.chat_id ? String(result.chat_id) : null;
      
      // Validate result shape before rendering
      const isValid = validateResponseShape(result, Schemas.CHAT_RUN, 'sendMCOQuery');
      
      const answerText = isValid 
        ? (result.formatted_output || result.data?.priority_answer || result.priority_answer || 'No response.')
        : 'Error: Received invalid response shape from server.';

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

      // Keep sidebar and cached state in sync with authoritative backend history.
      reloadHistory();

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
    const cached = (persistedMessages || [])
      .filter((m) => String(m.chat_id) === String(run.id))
      .map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        timestamp: m.created_at || run.timestamp,
        image_b64: m.image_b64 || null,
        image_mime: m.image_mime || null,
        reasoning_json: m.reasoning_json || null,
      }));

    setMessages(cached);
    setCurrentResult(null);
    setLoading(true);

    try {
      const data = await getChatMessages(run.id);
      const loaded = (data || [])
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => ({
          id: m.id,
          role: m.role,
          content: m.content,
          timestamp: m.timestamp || run.timestamp,
          image_b64: m.image_b64 || null,
          image_mime: m.image_mime || null,
          reasoning_json: m.reasoning_json || null,
        }));
      setMessages(loaded);
    } catch (err) {
      if (cached.length === 0) {
        setError('Failed to load chat history.');
      }
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
      historyLoading={historyLoading}
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
      onToggleClaude={onToggleClaude}
    />
  );
}
