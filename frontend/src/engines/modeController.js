/**
 * Mode Controller - Sentinel-E
 *
 * Resolves rendering mode from backend response metadata.
 * Supports: standard, debate, glass, evidence, ensemble, forensic
 */

/**
 * Resolve rendering mode from backend response.
 * Reads omega_metadata.sub_mode / mode to determine which view to render.
 */
export function resolveRenderMode(result) {
  if (!result || !result.omega_metadata) {
    return { mode: 'standard', engine: 'StandardKernel', data: null };
  }

  const meta = result.omega_metadata;
  const mode = meta.sub_mode || meta.mode || result.sub_mode || result.mode || 'standard';

  const base = {
    mode,
    data: meta,
    boundary: result.boundary_result || meta.boundary_result || {},
    confidence: result.confidence ?? meta.confidence ?? 0.5,
  };

  // Ensemble mode — full cognitive pipeline data
  if (mode === 'ensemble' || mode === 'cognitive' || meta.ensemble_metrics) {
    return {
      ...base,
      mode: 'ensemble',
      engine: 'CognitiveCoreEngine',
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

  // Debate mode
  if (mode === 'debate') {
    return {
      ...base,
      engine: 'DebateEngine',
      debateResult: meta.debate_result || meta.aggregation_result || {},
      rounds: meta.debate_result?.rounds || [],
      modelOutputs: meta.model_outputs || meta.aggregation_result?.model_outputs || [],
    };
  }

  // Glass mode
  if (mode === 'glass') {
    return {
      ...base,
      engine: 'GlassEngine',
      auditResult: meta.audit_result || {},
      reasoningTrace: result.reasoning_trace || meta.reasoning_trace || {},
    };
  }

  // Evidence mode
  if (mode === 'evidence') {
    return {
      ...base,
      engine: 'EvidenceEngine',
      forensicResult: meta.forensic_result || {},
      sources: meta.forensic_result?.sources || [],
    };
  }

  // Standard / conversational (default)
  return {
    ...base,
    engine: 'StandardKernel',
    aggregationResult: meta.aggregation_result || {},
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
 * Default pipeline steps based on mode.
 */
export function getDefaultPipelineSteps(mode, subMode) {
  const effectiveMode = subMode || mode || 'standard';

  if (effectiveMode === 'ensemble') {
    return [
      { id: 0, step: 'validate', label: 'Validating Providers', status: 'pending' },
      { id: 1, step: 'execute', label: 'Executing All Models', status: 'pending' },
      { id: 2, step: 'structure', label: 'Parsing Structured Outputs', status: 'pending' },
      { id: 3, step: 'debate', label: 'Running Structured Debate', status: 'pending' },
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

  if (effectiveMode === 'debate') {
    return [
      { id: 0, step: 'context', label: 'Stabilizing Context', status: 'pending' },
      { id: 1, step: 'routing', label: 'Routing to Debate Engine', status: 'pending' },
      { id: 2, step: 'round1', label: 'Debate Round 1', status: 'pending' },
      { id: 3, step: 'round2', label: 'Debate Round 2', status: 'pending' },
      { id: 4, step: 'round3', label: 'Debate Round 3', status: 'pending' },
      { id: 5, step: 'synthesis', label: 'Synthesizing Arguments', status: 'pending' },
      { id: 6, step: 'rendering', label: 'Rendering', status: 'pending' },
    ];
  }

  if (effectiveMode === 'glass') {
    return [
      { id: 0, step: 'context', label: 'Stabilizing Context', status: 'pending' },
      { id: 1, step: 'audit', label: 'Generating Blind Audit', status: 'pending' },
      { id: 2, step: 'trace', label: 'Extracting Reasoning Trace', status: 'pending' },
      { id: 3, step: 'comparison', label: 'Comparing Outputs', status: 'pending' },
      { id: 4, step: 'rendering', label: 'Rendering', status: 'pending' },
    ];
  }

  if (effectiveMode === 'evidence') {
    return [
      { id: 0, step: 'context', label: 'Stabilizing Context', status: 'pending' },
      { id: 1, step: 'research', label: 'Extracting Evidence', status: 'pending' },
      { id: 2, step: 'citation', label: 'Verifying Citations', status: 'pending' },
      { id: 3, step: 'scoring', label: 'Scoring Evidence', status: 'pending' },
      { id: 4, step: 'rendering', label: 'Rendering', status: 'pending' },
    ];
  }

  // Standard / conversational
  return [
    { id: 0, step: 'context', label: 'Stabilizing Context', status: 'pending' },
    { id: 1, step: 'routing', label: 'Routing Query', status: 'pending' },
    { id: 2, step: 'inference', label: 'Running Inference', status: 'pending' },
    { id: 3, step: 'boundary', label: 'Boundary Analysis', status: 'pending' },
    { id: 4, step: 'rendering', label: 'Rendering', status: 'pending' },
  ];
}

const modeController = { resolveRenderMode, extractPipelineSteps, getDefaultPipelineSteps };
export default modeController;
