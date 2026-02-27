/**
 * Mode Controller - Sentinel-E Cognitive Engine v7.0
 *
 * All responses are ensemble mode. No mode-based routing.
 * Always renders CognitiveDashboard (via EnsembleView).
 */

/**
 * Resolve rendering mode from backend response.
 * Always returns ensemble - there is no other mode.
 */
export function resolveRenderMode(result) {
  if (!result || !result.omega_metadata) {
    return { mode: 'ensemble', engine: 'CognitiveCoreEngine', data: null };
  }

  const meta = result.omega_metadata;

  return {
    mode: 'ensemble',
    engine: 'CognitiveCoreEngine',
    data: meta,
    boundary: result.boundary_result || meta.boundary_result || {},
    confidence: result.confidence ?? meta.confidence ?? 0.5,
    entropy: result.entropy ?? meta.entropy ?? 0,
    fragility: result.fragility ?? meta.fragility ?? 0,
    ensembleMetrics: meta.ensemble_metrics || {},
    debateRounds: meta.debate_rounds || [],
    agreementMatrix: meta.agreement_matrix || {},
    driftMetrics: meta.drift_metrics || {},
    modelStatus: meta.model_status || [],
    sessionIntelligence: meta.session_intelligence || {},
    tacticalMap: meta.tactical_map || {},
    calibratedConfidence: meta.confidence_graph || meta.calibrated_confidence || {},
    modelOutputs: meta.model_outputs || [],
    modelsExecuted: result.models_executed ?? meta.ensemble_metrics?.model_count ?? 0,
    modelsSucceeded: result.models_succeeded ?? meta.ensemble_metrics?.successful_models ?? 0,
    modelsFailed: result.models_failed ?? meta.ensemble_metrics?.failed_models ?? 0,
    error: result.error || meta.error || null,
    errorCode: result.error_code || meta.error_code || null,
  };
}

/**
 * Extract pipeline steps from result for ThinkingAnimation
 */
export function extractPipelineSteps(result) {
  if (!result) return [];
  const meta = result.omega_metadata;
  if (!meta) return [];
  if (meta.pipeline_steps) {
    return meta.pipeline_steps.map((step, i) => ({
      id: i,
      step: step.step || step.name || `step_${i}`,
      label: step.label || step.name || 'Processing...',
      status: step.status || 'pending',
    }));
  }
  return [];
}

/**
 * Default pipeline steps - always the cognitive ensemble pipeline.
 */
export function getDefaultPipelineSteps() {
  return [
    { id: 0, step: 'validate', label: 'Validating Providers (4+ models)', status: 'pending' },
    { id: 1, step: 'execute', label: 'Executing All Models', status: 'pending' },
    { id: 2, step: 'structure', label: 'Parsing Structured Outputs', status: 'pending' },
    { id: 3, step: 'debate', label: 'Running Structured Debate (3+ rounds)', status: 'pending' },
    { id: 4, step: 'agreement', label: 'Computing Agreement Matrix', status: 'pending' },
    { id: 5, step: 'drift', label: 'Tracking Stance Drift', status: 'pending' },
    { id: 6, step: 'metrics', label: 'Computing Ensemble Metrics', status: 'pending' },
    { id: 7, step: 'calibrate', label: 'Calibrating Confidence', status: 'pending' },
    { id: 8, step: 'tactical', label: 'Building Tactical Map', status: 'pending' },
    { id: 9, step: 'synthesis', label: 'Synthesizing Consensus', status: 'pending' },
    { id: 10, step: 'session', label: 'Updating Session Intelligence', status: 'pending' },
    { id: 11, step: 'rendering', label: 'Rendering', status: 'pending' },
  ];
}

const modeController = { resolveRenderMode, extractPipelineSteps, getDefaultPipelineSteps };
export default modeController;
