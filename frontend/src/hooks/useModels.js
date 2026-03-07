/**
 * useModels — Dynamic Model Registry Hook
 * 
 * Fetches cognitive models from:
 *   1. GET /chat/models/available  — individual-model (Standard Mode) registry w/ tier info
 *   2. GET /api/mco/models         — MCO registry for backward-compat (aggregated modes)
 * 
 * Falls back to DEFAULT_CHAT_MODELS (all 7 official ensemble models) if backend
 * is unreachable so the UI never renders an empty model list.
 * 
 * Returns:
 *   mcoModels        — Raw MCO registry models (backend data)
 *   chatModels       — UI-formatted models for the model selector
 *   debateModels     — Subset of chatModels with debate_eligible=true (Standard + Debate mode)
 *   loading          — Whether fetch is in progress
 *   error            — Error message if fetch failed
 *   refetch          — Manual refetch trigger
 *   ROLE_COLORS      — Role → colour map
 *   ROLE_LABELS      — Role → human label map
 *   TIER_LABELS      — Tier number → human label map
 */

import { useState, useEffect, useCallback } from 'react';
import { fetchMCOModels, fetchChatModels } from '../services/api';

// ── Role → color mapping ────────────────────────────────────
const ROLE_COLORS = {
  code: '#10b981',
  vision: '#06b6d4',
  baseline: '#6366f1',
  conceptual: '#8b5cf6',
  longctx: '#f59e0b',
  fast: '#ef4444',
  general: '#3b82f6',
};

const ROLE_LABELS = {
  code: 'Code',
  vision: 'Vision',
  baseline: 'Reasoning',
  conceptual: 'Conceptual',
  longctx: 'Long Context',
  fast: 'Speed',
  general: 'General',
};

const TIER_LABELS = {
  1: 'Anchor',
  2: 'Debate',
  3: 'Specialist',
};

const TIER_COLORS = {
  1: '#f59e0b',   // amber  — anchor
  2: '#3b82f6',   // blue   — debate
  3: '#10b981',   // emerald — specialist
};

/**
 * Official 7-model ensemble fallback.
 * Used when the backend /chat/models/available endpoint is unreachable.
 * Mirrors COGNITIVE_MODEL_REGISTRY in backend/metacognitive/cognitive_gateway.py.
 */
export const DEFAULT_CHAT_MODELS = [
  // Tier 1 Anchors
  {
    id: 'llama-3.3',
    name: 'Llama 3.3 70B Versatile',
    provider: 'groq',
    color: TIER_COLORS[1],
    role: 'baseline',
    tier: 1,
    tierLabel: 'Anchor',
    enabled: true,
    debate_eligible: true,
    context_window: 131072,
    max_output_tokens: 8192,
  },
  {
    id: 'deepseek-chat',
    name: 'DeepSeek Chat',
    provider: 'openrouter',
    color: TIER_COLORS[1],
    role: 'general',
    tier: 1,
    tierLabel: 'Anchor',
    enabled: true,
    debate_eligible: true,
    context_window: 65536,
    max_output_tokens: 4096,
  },
  // Tier 2 Debate
  {
    id: 'groq-small',
    name: 'Llama 3.1 8B Instant',
    provider: 'groq',
    color: TIER_COLORS[2],
    role: 'fast',
    tier: 2,
    tierLabel: 'Debate',
    enabled: true,
    debate_eligible: true,
    context_window: 131072,
    max_output_tokens: 8192,
  },
  {
    id: 'mixtral-8x7b',
    name: 'Mixtral 8x7B Instruct',
    provider: 'openrouter',
    color: TIER_COLORS[2],
    role: 'general',
    tier: 2,
    tierLabel: 'Debate',
    enabled: true,
    debate_eligible: true,
    context_window: 32768,
    max_output_tokens: 4096,
  },
  {
    id: 'qwen2.5-32b',
    name: 'Qwen2.5 32B Instruct',
    provider: 'openrouter',
    color: TIER_COLORS[2],
    role: 'conceptual',
    tier: 2,
    tierLabel: 'Debate',
    enabled: true,
    debate_eligible: true,
    context_window: 131072,
    max_output_tokens: 8192,
  },
  // Tier 3 Specialists
  {
    id: 'deepseek-coder-v2',
    name: 'DeepSeek Coder V2 Lite',
    provider: 'openrouter',
    color: TIER_COLORS[3],
    role: 'code',
    tier: 3,
    tierLabel: 'Specialist',
    enabled: true,
    debate_eligible: true,
    context_window: 65536,
    max_output_tokens: 4096,
  },
  {
    id: 'qwen2.5-coder-32b',
    name: 'Qwen2.5 Coder 32B',
    provider: 'openrouter',
    color: TIER_COLORS[3],
    role: 'code',
    tier: 3,
    tierLabel: 'Specialist',
    enabled: true,
    debate_eligible: true,
    context_window: 131072,
    max_output_tokens: 8192,
  },
];

/**
 * Transform backend /chat/models/available response into UI chat models.
 * Preserves tier metadata and debate_eligible flag.
 */
function transformChatModels(backendModels) {
  if (!backendModels || backendModels.length === 0) return DEFAULT_CHAT_MODELS;

  return backendModels.map((m) => ({
    id: m.id,
    name: m.name,
    provider: m.provider,
    color: m.enabled
      ? (TIER_COLORS[m.tier] || ROLE_COLORS[m.role] || '#6366f1')
      : '#6b7280',  // grey when disabled
    role: m.role,
    tier: m.tier,
    tierLabel: TIER_LABELS[m.tier] || 'Debate',
    enabled: m.enabled,
    debate_eligible: m.enabled,
    context_window: m.context_window,
    max_output_tokens: m.max_output_tokens,
  }));
}

/**
 * Transform MCO backend models into legacy chat UI format.
 * Kept for backward-compatibility with aggregated MCO mode.
 */
function transformMCOForChat(mcoModels) {
  if (!mcoModels || mcoModels.length === 0) return [];

  return mcoModels.map((m) => ({
    id: m.key,
    name: m.name,
    provider: m.provider,
    color: ROLE_COLORS[m.role] || '#6366f1',
    role: m.role,
    tier: null,
    tierLabel: null,
    enabled: m.enabled,
    debate_eligible: false,   // MCO models go through aggregated pipeline
    active: m.active,
    modelId: m.model_id,
    contextWindow: m.context_window,
    maxOutputTokens: m.max_output_tokens,
    isMCO: true,
  }));
}

export default function useModels() {
  const [mcoModels, setMcoModels] = useState([]);
  const [chatModels, setChatModels] = useState(DEFAULT_CHAT_MODELS);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch both endpoints in parallel; either can fail gracefully
      const [chatResult, mcoResult] = await Promise.allSettled([
        fetchChatModels(),
        fetchMCOModels(),
      ]);

      // Chat / Standard Mode models (primary)
      if (chatResult.status === 'fulfilled') {
        const models = chatResult.value?.models || [];
        if (models.length > 0) {
          setChatModels(transformChatModels(models));
        }
        // else keep DEFAULT_CHAT_MODELS
      }

      // MCO models (legacy / aggregated mode)
      if (mcoResult.status === 'fulfilled') {
        const models = mcoResult.value?.models || [];
        setMcoModels(models);
      }
    } catch (err) {
      setError(err.message || 'Failed to fetch models');
      setChatModels(DEFAULT_CHAT_MODELS);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // Derive debate-eligible models (tier 1, 2, 3 and enabled)
  const debateModels = chatModels.filter((m) => m.debate_eligible && m.enabled);

  return {
    mcoModels,
    chatModels,
    debateModels,
    loading,
    error,
    refetch: fetchModels,
    ROLE_COLORS,
    ROLE_LABELS,
    TIER_LABELS,
    TIER_COLORS,
  };
}

