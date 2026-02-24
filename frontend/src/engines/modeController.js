/**
 * Mode Controller — Frontend Engine Layer
 * 
 * Processes structured response data from backend v4 engines.
 * Determines which output component to render based on response metadata.
 * No mode logic in UI — this layer routes to the correct renderer.
 */

/**
 * Determin which rendering mode to use based on omega_metadata
 */
export function resolveRenderMode(result) {
  if (!result || !result.omega_metadata) {
    return { mode: 'standard', engine: null, data: null };
  }

  const meta = result.omega_metadata;

  if (meta.aggregation_result) {
    return {
      mode: 'aggregation',
      engine: 'AggregationEngine',
      data: meta.aggregation_result,
      pipelineSteps: meta.pipeline_steps || [],
      boundary: result.boundary_result || meta.boundary_result || {},
      confidence: result.confidence,
    };
  }

  if (meta.forensic_result) {
    return {
      mode: 'evidence',
      engine: 'ForensicEvidenceEngine',
      data: meta.forensic_result,
      pipelineSteps: meta.pipeline_steps || [],
      boundary: result.boundary_result || meta.boundary_result || {},
      confidence: result.confidence,
    };
  }

  if (meta.audit_result) {
    return {
      mode: 'glass',
      engine: 'BlindAuditEngine',
      data: meta.audit_result,
      pipelineSteps: meta.pipeline_steps || [],
      boundary: result.boundary_result || meta.boundary_result || {},
      confidence: result.confidence,
    };
  }

  if (meta.debate_result) {
    return {
      mode: 'debate',
      engine: 'DebateOrchestrator',
      data: meta.debate_result,
      boundary: result.boundary_result || meta.boundary_result || {},
      confidence: result.confidence,
    };
  }

  if (meta.kill_active) {
    return {
      mode: 'kill',
      engine: 'KillDiagnostic',
      data: meta,
      confidence: result.confidence,
    };
  }

  return {
    mode: 'standard',
    engine: null,
    data: null,
    boundary: result.boundary_result || meta.boundary_result || {},
    confidence: result.confidence,
  };
}

/**
 * Extract pipeline steps from result for ThinkingAnimation
 */
export function extractPipelineSteps(result) {
  if (!result) return [];

  const meta = result.omega_metadata;
  if (!meta) return [];

  // From v4 engines
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
 * Determine the default pipeline steps for a given mode (for pre-loading animation)
 */
export function getDefaultPipelineSteps(mode, subMode) {
  if (mode === 'standard') {
    return [
      { id: 0, step: 'retrieval', label: 'Retrieving Models', status: 'pending', icon: '🔎' },
      { id: 1, step: 'generation', label: 'Generating Independent Outputs', status: 'pending', icon: '🧠' },
      { id: 2, step: 'verification', label: 'Cross-Verifying', status: 'pending', icon: '🔄' },
      { id: 3, step: 'risk', label: 'Computing Risk', status: 'pending', icon: '📊' },
      { id: 4, step: 'synthesis', label: 'Synthesizing', status: 'pending', icon: '🧩' },
      { id: 5, step: 'rendering', label: 'Rendering', status: 'pending', icon: '🧾' },
    ];
  }

  if (subMode === 'debate') {
    return [
      { id: 0, step: 'setup', label: 'Assigning Model Roles', status: 'pending', icon: '🔎' },
      { id: 1, step: 'rounds', label: 'Running Debate Rounds', status: 'pending', icon: '🧠' },
      { id: 2, step: 'analysis', label: 'Analyzing Positions', status: 'pending', icon: '🔄' },
      { id: 3, step: 'consensus', label: 'Computing Consensus', status: 'pending', icon: '📊' },
      { id: 4, step: 'rendering', label: 'Rendering', status: 'pending', icon: '🧾' },
    ];
  }

  if (subMode === 'evidence') {
    return [
      { id: 0, step: 'retrieval', label: 'Independent Model Retrieval', status: 'pending', icon: '🔎' },
      { id: 1, step: 'extraction', label: 'Extracting Structured Claims', status: 'pending', icon: '🧠' },
      { id: 2, step: 'triangulation', label: 'Triangular Cross-Verification', status: 'pending', icon: '🔄' },
      { id: 3, step: 'contradiction', label: 'Contradiction Detection', status: 'pending', icon: '📊' },
      { id: 4, step: 'bayesian', label: 'Bayesian Confidence Update', status: 'pending', icon: '🧩' },
      { id: 5, step: 'rendering', label: 'Rendering', status: 'pending', icon: '🧾' },
    ];
  }

  if (subMode === 'glass') {
    return [
      { id: 0, step: 'generation', label: 'Independent Generation', status: 'pending', icon: '🔎' },
      { id: 1, step: 'audit', label: 'Blind Cross-Model Audit', status: 'pending', icon: '🧠' },
      { id: 2, step: 'tactical', label: 'Computing Tactical Map', status: 'pending', icon: '🔄' },
      { id: 3, step: 'trust', label: 'Computing Trust Score', status: 'pending', icon: '📊' },
      { id: 4, step: 'rendering', label: 'Rendering', status: 'pending', icon: '🧾' },
    ];
  }

  return [
    { id: 0, step: 'processing', label: 'Processing', status: 'pending', icon: '🔎' },
    { id: 1, step: 'rendering', label: 'Rendering', status: 'pending', icon: '🧾' },
  ];
}

const modeController = { resolveRenderMode, extractPipelineSteps, getDefaultPipelineSteps };
export default modeController;
