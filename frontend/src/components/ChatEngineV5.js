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
import {
  createNewConversation,
  loadConversationHistory,
  loadConversationState,
  persistSessionState,
  restoreGuestSession,
  saveConversationHistory,
  switchConversation,
} from '../services/guestSession';
import useStore from '../stores/useStore';
import { validateResponseShape, Schemas } from '../utils/validation';

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function normalizeLocalMessages(messages) {
  return Array.isArray(messages) ? messages.filter(Boolean) : [];
}

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
  const [guestHydrated, setGuestHydrated] = useState(false);
  const [localConversations, setLocalConversations] = useState([]);
  const guestBootstrapRef = React.useRef(false);

  const [input, setInput] = useState('');
  const persistedChats = useStore((state) => state.chats);
  const persistedMessages = useStore((state) => state.messages);
  const historyLoading = useStore((state) => state.isLoading);
  const reloadHistory = useStore((state) => state.reloadHistory);

  const history = useMemo(
    () => {
      const backendHistory = (persistedChats || []).map((item) => ({
      id: item.id,
      timestamp: item.updated_at || item.created_at || new Date().toISOString(),
      mode: item.mode,
      summary: item.chat_name || item.preview || 'Chat',
      filename: item.id,
      data: null,
      sub_mode: item.sub_mode || null,
      }));

      const backendIds = new Set(backendHistory.map((item) => String(item.id)));
      const localHistory = (localConversations || [])
        .filter((item) => item?.id && !backendIds.has(String(item.id)))
        .map((item) => ({
          id: item.id,
          timestamp: item.updatedAt || item.createdAt || new Date().toISOString(),
          mode: item.mode || 'standard',
          summary: item.title || 'New Chat',
          filename: item.id,
          data: null,
          sub_mode: item.subMode || null,
          isLocalGuest: true,
        }));

      return [...localHistory, ...backendHistory].sort(
        (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
    },
    [persistedChats, localConversations]
  );
  // Default to Sentinel Standard aggregate mode (not an individual model)
  const SENTINEL_STD = { id: 'sentinel-std', name: 'Sentinel-E Standard', provider: 'Aggregated', color: '#3b82f6', category: 'standard', isMeta: true, enabled: true };
  const [selectedModel, setSelectedModel] = useState(SENTINEL_STD);

  const persistActiveConversation = useCallback((overrides = {}) => {
    const safeActiveChatId = overrides.activeChatId || activeChatId;
    if (!guestHydrated || !safeActiveChatId) return null;

    // TODO: Replace guest-session persistence with Firebase-auth session persistence later
    const saved = persistSessionState({
      conversationId: safeActiveChatId,
      mode,
      subMode,
      messages: Array.isArray(messages) ? messages : [],
      currentResult,
      sessionState,
      lastQueryText,
      lastResponseText,
      governanceVerdict,
      orchestrationState: currentResult?.omega_metadata || currentResult?.orchestration_state || null,
      analyticsState: currentResult?.analytics || currentResult?.omega_metadata || null,
      debateState: subMode === 'debate' ? (currentResult?.debate || currentResult?.omega_metadata || null) : null,
      tacticalMapState: currentResult?.tactical_map || currentResult?.map_state || null,
      uiState: {
        mode,
        subMode,
        killActive,
        selectedModelId: selectedModel?.id || 'sentinel-std',
      },
      ...overrides,
    });
    setLocalConversations(loadConversationHistory());
    return saved?.conversation || null;
  }, [
    activeChatId,
    currentResult,
    governanceVerdict,
    guestHydrated,
    killActive,
    lastQueryText,
    lastResponseText,
    messages,
    mode,
    selectedModel?.id,
    sessionState,
    subMode,
  ]);

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
    if (guestBootstrapRef.current) return;
    guestBootstrapRef.current = true;

    const restored = restoreGuestSession();
    const conversations = loadConversationHistory();
    let conversation = restored.activeChatId ? loadConversationHistory(restored.activeChatId) : null;

    if (!conversation && conversations.length > 0) {
      conversation = conversations[0];
    }

    if (!conversation) {
      conversation = createNewConversation({ mode: 'standard' });
    }

    switchConversation(conversation.id, { createIfMissing: true, mode: conversation.mode, subMode: conversation.subMode });
    persistSessionState({
      conversationId: conversation.id,
      uiState: {
        mode: conversation.mode || restored.uiState?.mode || 'standard',
        subMode: conversation.subMode ?? restored.uiState?.subMode ?? null,
        killActive: Boolean(restored.uiState?.killActive),
      },
    });
    setActiveChatId(conversation.id);
    setMessages(Array.isArray(conversation.messages) ? conversation.messages : []);
    setCurrentResult(conversation.currentResult || null);
    setSessionState(conversation.sessionState || null);
    setLastQueryText(conversation.lastQueryText || '');
    setLastResponseText(conversation.lastResponseText || '');
    setGovernanceVerdict(conversation.governanceVerdict || null);
    setMode(conversation.mode || restored.uiState?.mode || 'standard');
    setSubMode(conversation.subMode ?? restored.uiState?.subMode ?? null);
    setKillActive(Boolean(restored.uiState?.killActive));
    setLocalConversations(loadConversationHistory());
    setGuestHydrated(true);
  }, []);

  useEffect(() => {
    if (!guestHydrated) return;
    if (activeChatId) {
      localStorage.setItem('sentinel-active-chat-id', activeChatId);
    } else {
      localStorage.removeItem('sentinel-active-chat-id');
    }
  }, [activeChatId, guestHydrated]);

  useEffect(() => {
    persistActiveConversation();
  }, [persistActiveConversation]);

  useEffect(() => {
    if (!guestHydrated || !activeChatId || messages.length > 0) return;
    const localConversation = loadConversationHistory(activeChatId);
    if (localConversation?.messages?.length > 0) {
      setMessages(localConversation.messages);
      return;
    }

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
  }, [activeChatId, persistedMessages, messages.length, guestHydrated]);

  // ── Session State ────────────────────────────────────────
  useEffect(() => {
    if (!guestHydrated || !activeChatId) return;
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
  }, [activeChatId, guestHydrated]);

  // ── Send Handler ─────────────────────────────────────────
  const handleSend = async ({ text, file }) => {
    const activeUserId = restoreGuestSession()?.guestSessionId || 'guest-user';

    // PHASE 1: Log current user id for diagnostics
    try {
      console.log('FRONTEND: SEND - USER_ID', activeUserId);
    } catch (e) {
      /* ignore */
    }

    // TODO: Restore Firebase Auth after configuration fixes
    // Original auth gate preserved below.
    //
    // if (!activeUserId) {
    //   console.error('NO USER_ID — blocking send');
    //   setError('Not authenticated. Please sign in.');
    //   return;
    // }

    if (!text && !file) return;
    const ensuredConversation = activeChatId
      ? loadConversationState(activeChatId)
      : createNewConversation({ mode, subMode });
    const chatId = activeChatId || ensuredConversation?.id || createNewConversation({ mode, subMode }).id;
    if (!activeChatId) setActiveChatId(chatId);
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
    const optimisticMessages = [...(Array.isArray(messages) ? messages : []), userMsg];
    setMessages(optimisticMessages);
    saveConversationHistory(chatId, optimisticMessages, {
      mode,
      subMode,
      lastQueryText: text || '',
    });
    setLocalConversations(loadConversationHistory());
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
      const completedMessages = [...optimisticMessages, assistantMsg];
      setMessages(completedMessages);
      setCurrentResult(result);
      setLastResponseText(answerText);
      if (result.session_state) setSessionState(result.session_state);
      saveConversationHistory(returnedChatId || chatId, completedMessages, {
        mode,
        subMode,
        currentResult: result,
        sessionState: result.session_state || sessionState,
        lastQueryText: text || '',
        lastResponseText: answerText,
        governanceVerdict: null,
        orchestrationState: result.omega_metadata || result.orchestration_state || null,
        analyticsState: result.analytics || result.omega_metadata || null,
        debateState: subMode === 'debate' ? (result.debate || result.omega_metadata || null) : null,
        tacticalMapState: result.tactical_map || result.map_state || null,
      });
      setLocalConversations(loadConversationHistory());

      memoryManager.recordMessage(assistantMsg, mode, subMode);
      memoryManager.recordAnalytics(result);

      const verdict = evaluateResponse({
        userQuery: text || '',
        responseText: answerText,
        responseData: result,
        mode, subMode,
      });
      setGovernanceVerdict(verdict);
      saveConversationHistory(returnedChatId || chatId, completedMessages, {
        currentResult: result,
        sessionState: result.session_state || sessionState,
        lastQueryText: text || '',
        lastResponseText: answerText,
        governanceVerdict: verdict,
      });
      setLocalConversations(loadConversationHistory());

      if (returnedChatId && UUID_REGEX.test(returnedChatId)) {
        setActiveChatId(returnedChatId);
      }

      // Keep sidebar and cached state in sync with authoritative backend history.
      reloadHistory();

      setServerStatus('online');
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.');
      const rolledBackMessages = optimisticMessages.slice(0, -1);
      setMessages(rolledBackMessages); // Remove optimistic user msg
      saveConversationHistory(chatId, rolledBackMessages, { mode, subMode });
      setLocalConversations(loadConversationHistory());
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
      if (!run.sub_mode) setSubMode(null);
    } else {
      setMode('experimental');
    }
    if (run.sub_mode) setSubMode(run.sub_mode);
    setActiveChatId(run.id);
    switchConversation(run.id, { createIfMissing: true, mode: run.mode || 'standard', subMode: run.sub_mode || null });
    const localConversation = loadConversationState(run.id);
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

    const safeCached = cached.length > 0 ? cached : normalizeLocalMessages(localConversation?.messages);
    setMessages(safeCached);
    setCurrentResult(localConversation?.currentResult || null);
    setSessionState(localConversation?.sessionState || null);
    setLastQueryText(localConversation?.lastQueryText || '');
    setLastResponseText(localConversation?.lastResponseText || '');
    setGovernanceVerdict(localConversation?.governanceVerdict || null);
    setLoading(true);

    if (run.isLocalGuest) {
      setLoading(false);
      return;
    }

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
      saveConversationHistory(run.id, loaded, {
        mode: run.mode || mode,
        subMode: run.sub_mode || subMode,
      });
      setLocalConversations(loadConversationHistory());
    } catch (err) {
      if (safeCached.length === 0) {
        setError('Failed to load chat history.');
      }
    } finally {
      setLoading(false);
    }
  };

  // ── New Chat ─────────────────────────────────────────────
  const handleNewChat = () => {
    const conversation = createNewConversation({ mode, subMode });
    setActiveChatId(conversation.id);
    setMessages([]);
    setCurrentResult(null);
    setShowLearning(false);
    setSessionState(null);
    setKillActive(false);
    setLastResponseText('');
    setLastQueryText('');
    setError(null);
    setLocalConversations(loadConversationHistory());
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
