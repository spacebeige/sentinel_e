/**
 * useModels — Dynamic Model Registry Hook
 * 
 * Fetches cognitive models from GET /api/mco/models.
 * Falls back to DEFAULT_MODELS if backend is unreachable.
 * 
 * Returns:
 *   mcoModels   — Raw MCO registry models (backend data)
 *   chatModels  — UI-formatted models for the chat dropdown
 *   loading     — Whether fetch is in progress
 *   error       — Error message if fetch failed
 *   refetch     — Manual refetch trigger
 */

import { useState, useEffect, useCallback } from 'react';
import { fetchMCOModels } from '../services/api';

// ── Role → color mapping ────────────────────────────────────
const ROLE_COLORS = {
  code: '#10b981',
  vision: '#06b6d4',
  baseline: '#6366f1',
  conceptual: '#8b5cf6',
  longctx: '#f59e0b',
};

const ROLE_LABELS = {
  code: 'Code',
  vision: 'Vision',
  baseline: 'Reasoning',
  conceptual: 'Conceptual',
  longctx: 'Long Context',
};

/**
 * Default models used when backend is offline.
 * These match the existing MODELS constant from FigmaChatShell.
 */
export const DEFAULT_CHAT_MODELS = [
  { id: 'sentinel-std', name: 'Sentinel-E Standard', provider: 'Standard', color: '#3b82f6', category: 'standard', enabled: true, role: 'baseline' },
  { id: 'sentinel-exp', name: 'Sentinel-E Pro', provider: 'Experimental', color: '#8b5cf6', category: 'experimental', enabled: true, role: 'conceptual' },
];

/**
 * Transform MCO backend models into chat UI models.
 */
function transformForChat(mcoModels) {
  if (!mcoModels || mcoModels.length === 0) return DEFAULT_CHAT_MODELS;

  // Always include the two meta-models (Standard + Pro)
  const chatModels = [
    { id: 'sentinel-std', name: 'Sentinel-E Standard (Aggregated)', provider: 'All Models', color: '#3b82f6', category: 'standard', enabled: true, role: 'baseline', isMeta: true },
  ];

  // Add individual MCO models under standard category
  mcoModels.forEach((m) => {
    chatModels.push({
      id: m.key,
      name: m.name,
      provider: m.provider,
      color: ROLE_COLORS[m.role] || '#6366f1',
      category: 'standard',
      enabled: m.enabled,
      active: m.active,
      role: m.role,
      modelId: m.model_id,
      contextWindow: m.context_window,
      maxOutputTokens: m.max_output_tokens,
    });
  });

  // Add experimental meta-model
  chatModels.push({
    id: 'sentinel-exp', name: 'Sentinel-E Pro', provider: 'Experimental', color: '#8b5cf6', category: 'experimental', enabled: true, role: 'conceptual', isMeta: true,
  });

  return chatModels;
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
      const data = await fetchMCOModels();
      const models = data?.models || [];
      setMcoModels(models);
      setChatModels(transformForChat(models));
    } catch (err) {
      setError(err.message || 'Failed to fetch models');
      // Keep defaults on error
      setChatModels(DEFAULT_CHAT_MODELS);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  return {
    mcoModels,
    chatModels,
    loading,
    error,
    refetch: fetchModels,
    ROLE_COLORS,
    ROLE_LABELS,
  };
}
