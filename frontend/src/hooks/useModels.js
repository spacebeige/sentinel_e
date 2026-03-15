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
import { fetchMCOModels, fetchChatModels, toggleClaude } from '../services/api';

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
  3: 'Fallback',
};

const TIER_COLORS = {
  1: '#f59e0b',   // amber  — anchor
  2: '#3b82f6',   // blue   — debate
  3: '#10b981',   // emerald — fallback
};

/**
 * Official 6-model ensemble fallback.
 * Used when the backend /chat/models/available endpoint is unreachable.
 * Mirrors COGNITIVE_MODEL_REGISTRY in backend/metacognitive/cognitive_gateway.py.
 */
export const DEFAULT_CHAT_MODELS = [
  // Tier 1 Anchor — Analysis
  {
    id: 'llama33-70b',
    name: 'Llama 3.3 70B',
    provider: 'groq',
    color: TIER_COLORS[1],
    role: 'conceptual',
    tier: 1,
    tierLabel: 'Anchor',
    enabled: true,
    active: true,
    disable_reason: null,
    debate_eligible: true,
    context_window: 131072,
    max_output_tokens: 2000,
  },
  // Tier 2 Critique — Diverse argument generators
  {
    id: 'mixtral-8x7b',
    name: 'Mixtral 8x7B',
    provider: 'groq',
    color: TIER_COLORS[2],
    role: 'conceptual',
    tier: 2,
    tierLabel: 'Debate',
    enabled: true,
    active: true,
    disable_reason: null,
    debate_eligible: true,
    context_window: 32768,
    max_output_tokens: 1500,
  },
  {
    id: 'llama4-scout',
    name: 'Llama 4 Scout 17B',
    provider: 'groq',
    color: TIER_COLORS[2],
    role: 'general',
    tier: 2,
    tierLabel: 'Debate',
    enabled: true,
    active: true,
    disable_reason: null,
    debate_eligible: true,
    context_window: 8192,
    max_output_tokens: 1500,
  },
  {
    id: 'qwen-2.5-vl',
    name: 'Qwen 2.5 VL 7B',
    provider: 'qwen',
    color: TIER_COLORS[2],
    role: 'vision',
    tier: 2,
    tierLabel: 'Debate',
    enabled: true,
    active: true,
    disable_reason: null,
    debate_eligible: true,
    context_window: 32768,
    max_output_tokens: 1500,
  },
  // Tier 3 Synthesis + Verification
  {
    id: 'gemini-flash',
    name: 'Gemini Flash 2.0',
    provider: 'gemini',
    color: TIER_COLORS[3],
    role: 'general',
    tier: 3,
    tierLabel: 'Fallback',
    enabled: true,
    active: true,
    disable_reason: null,
    debate_eligible: true,
    context_window: 1048576,
    max_output_tokens: 2000,
  },
  {
    id: 'llama31-8b',
    name: 'Llama 3.1 8B Instant',
    provider: 'groq',
    color: TIER_COLORS[3],
    role: 'fast',
    tier: 3,
    tierLabel: 'Fallback',
    enabled: true,
    active: true,
    disable_reason: null,
    debate_eligible: true,
    context_window: 131072,
    max_output_tokens: 1500,
  },
  // NVIDIA Models
  {
    id: 'mistral-large-675b',
    name: 'Mistral Large 3 675B',
    provider: 'nvidia',
    color: TIER_COLORS[1],
    role: 'conceptual',
    tier: 1,
    tierLabel: 'Anchor',
    enabled: true,
    active: true,
    disable_reason: null,
    debate_eligible: true,
    context_window: 1000,
    max_output_tokens: 1000,
  },
  {
    id: 'kimi-k2-thinking',
    name: 'Kimi K2 Thinking',
    provider: 'nvidia',
    color: TIER_COLORS[2],
    role: 'conceptual',
    tier: 2,
    tierLabel: 'Debate',
    enabled: true,
    active: true,
    disable_reason: null,
    debate_eligible: true,
    context_window: 1000,
    max_output_tokens: 1000,
  },
  // Anthropic (Synthesis only)
  {
    id: 'claude-sonnet-4.6',
    name: 'Claude Sonnet 4.6',
    provider: 'anthropic',
    color: TIER_COLORS[3],
    role: 'general',
    tier: 3,
    tierLabel: 'Synthesis',
    enabled: true,
    active: true,
    disable_reason: null,
    debate_eligible: false,
    synthesis_only: true,
    context_window: 500,
    max_output_tokens: 500,
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
    tierLabel: m.synthesis_only ? 'Synthesis' : (TIER_LABELS[m.tier] || 'Debate'),
    enabled: m.enabled,
    active: m.active ?? m.enabled,
    disable_reason: m.disable_reason || null,
    debate_eligible: m.synthesis_only ? false : m.enabled,
    synthesis_only: m.synthesis_only || false,
    context_window: m.context_window,
    max_output_tokens: m.max_output_tokens,
  }));
}

/**
 * Transform MCO backend models into legacy chat UI format.
 * Kept for backward-compatibility with aggregated MCO mode.
 */

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

  const handleToggleClaude = useCallback(async () => {
    try {
      await toggleClaude();
      await fetchModels();
    } catch (err) {
      console.error('Failed to toggle Claude:', err);
    }
  }, [fetchModels]);

  return {
    mcoModels,
    chatModels,
    debateModels,
    loading,
    error,
    refetch: fetchModels,
    toggleClaude: handleToggleClaude,
    ROLE_COLORS,
    ROLE_LABELS,
    TIER_LABELS,
    TIER_COLORS,
  };
}

