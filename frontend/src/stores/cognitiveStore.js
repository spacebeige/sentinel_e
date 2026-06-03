/**
 * ============================================================
 * Cognitive Store — Global State for Sentinel-E v8.0
 * ============================================================
 * React Context-based store that persists across all cognitive
 * ensemble responses in a session.
 *
 * State:
 *   - debates: full response payloads
 *   - entropyHistory: entropy per query
 *   - driftHistory: mean drift per query
 *   - confidenceHistory: calibrated confidence per query
 *   - participationHistory: participation rate per query
 *   — v8.0 additions —
 *   - orchestrationRunHistory: OrchestrationRun summaries
 *   - cognitiveGrades: response quality grades (A/B/C/D)
 *   - phaseLabels: last known cognitive phase labels
 *   - contradictionHistory: contradiction density per query
 *
 * Updated via addDebateResult(response) on every response.
 * ============================================================
 */

import React, { createContext, useContext, useReducer, useCallback } from 'react';

// ── Initial State ───────────────────────────────────────────
const initialState = {
  debates: [],
  entropyHistory: [],
  driftHistory: [],
  confidenceHistory: [],
  participationHistory: [],
  totalDebates: 0,
  lastResponse: null,
  // ── v8.0 ─────────────────────────────────────────────────
  orchestrationRunHistory: [],  // OrchestrationRun summaries
  cognitiveGrades: [],          // 'A'/'B'/'C'/'D' per query
  phaseLabels: [],              // last phase label per query
  contradictionHistory: [],     // contradiction density per query
  lastOrchRun: null,            // most recent OrchestrationRun summary
  lastCognitiveArtifact: null,  // most recent cognitive artifact
};

// ── Actions ─────────────────────────────────────────────────
const ACTIONS = {
  ADD_DEBATE: 'ADD_DEBATE',
  CLEAR: 'CLEAR',
};

// ── Reducer ─────────────────────────────────────────────────
function cognitiveReducer(state, action) {
  switch (action.type) {
    case ACTIONS.ADD_DEBATE: {
      const response = action.payload;
      const meta = response.omega_metadata || response;

      // ── Existing metrics (preserved) ───────────────────────
      const entropy = response.entropy ?? meta.entropy ?? meta.ensemble_metrics?.disagreement_entropy ?? 0;
      const confidence = response.confidence ?? meta.confidence ?? 0.5;
      const fragility = response.fragility ?? meta.fragility ?? meta.ensemble_metrics?.fragility_score ?? 0;
      const participation = meta.ensemble_metrics?.participation_rate ?? 1.0;
      const drift = response.drift_metrics?.mean_drift ?? meta.drift_metrics?.mean_drift ?? 0;

      // ── v8.0: Extract new cognitive runtime fields ─────────
      const orchRun = meta.orchestration_run || null;
      const cogArtifact = meta.cognitive_artifact || null;
      const grade = cogArtifact?.quality_indicators?.response_grade || null;
      const phaseLabel = orchRun?.phase_label || null;
      const contradictionDensity = cogArtifact?.contradiction_analysis?.density ?? 0;

      return {
        ...state,
        debates: [...state.debates, {
          timestamp: response.timestamp || new Date().toISOString(),
          confidence,
          entropy,
          fragility,
          participation,
          drift,
          modelsExecuted: response.models_executed ?? meta.ensemble_metrics?.model_count ?? 0,
          modelsSucceeded: response.models_succeeded ?? meta.ensemble_metrics?.successful_models ?? 0,
          roundCount: response.debate_rounds?.length ?? meta.debate_rounds?.length ?? 0,
          error: response.error || null,
          // v8.0 additions (non-breaking)
          grade,
          phaseLabel,
          contradictionDensity,
          orchestrationRunId: orchRun?.orchestration_run_id || null,
        }],
        entropyHistory: [...state.entropyHistory, entropy],
        driftHistory: [...state.driftHistory, drift],
        confidenceHistory: [...state.confidenceHistory, confidence],
        participationHistory: [...state.participationHistory, participation],
        totalDebates: state.totalDebates + 1,
        lastResponse: response,
        // ── v8.0 ──────────────────────────────────────────────
        orchestrationRunHistory: orchRun
          ? [...state.orchestrationRunHistory.slice(-19), orchRun]
          : state.orchestrationRunHistory,
        cognitiveGrades: grade
          ? [...state.cognitiveGrades.slice(-19), grade]
          : state.cognitiveGrades,
        phaseLabels: phaseLabel
          ? [...state.phaseLabels.slice(-19), phaseLabel]
          : state.phaseLabels,
        contradictionHistory: [...state.contradictionHistory.slice(-19), contradictionDensity],
        lastOrchRun: orchRun || state.lastOrchRun,
        lastCognitiveArtifact: cogArtifact || state.lastCognitiveArtifact,
      };
    }

    case ACTIONS.CLEAR:
      return initialState;

    default:
      return state;
  }
}

// ── Context ─────────────────────────────────────────────────
const CognitiveStoreContext = createContext(null);

// ── Provider ────────────────────────────────────────────────
export function CognitiveStoreProvider({ children }) {
  const [state, dispatch] = useReducer(cognitiveReducer, initialState);

  const addDebateResult = useCallback((response) => {
    dispatch({ type: ACTIONS.ADD_DEBATE, payload: response });
  }, []);

  const clearStore = useCallback(() => {
    dispatch({ type: ACTIONS.CLEAR });
  }, []);

  const value = {
    ...state,
    addDebateResult,
    clearStore,
  };

  return (
    <CognitiveStoreContext.Provider value={value}>
      {children}
    </CognitiveStoreContext.Provider>
  );
}

// ── Hook ────────────────────────────────────────────────────
export function useCognitiveStore() {
  const context = useContext(CognitiveStoreContext);
  if (!context) {
    // Return a no-op store if used outside provider
    return {
      debates: [],
      entropyHistory: [],
      driftHistory: [],
      confidenceHistory: [],
      participationHistory: [],
      totalDebates: 0,
      lastResponse: null,
      orchestrationRunHistory: [],
      cognitiveGrades: [],
      phaseLabels: [],
      contradictionHistory: [],
      lastOrchRun: null,
      lastCognitiveArtifact: null,
      addDebateResult: () => {},
      clearStore: () => {},
    };
  }
  return context;
}

export default CognitiveStoreContext;
