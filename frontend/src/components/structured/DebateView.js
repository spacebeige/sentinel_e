import React, { useState, useMemo } from 'react';
import { ChevronDown, ChevronUp, ShieldAlert, ShieldCheck, Zap, AlertTriangle } from 'lucide-react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer,
         BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
         LineChart, Line, AreaChart, Area, CartesianGrid, Legend } from 'recharts';
import ConfidenceBar from './ConfidenceBar';
import BoundaryPanel from './BoundaryPanel';

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

const MODEL_COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4', '#ec4899'];

/** Qualitative label for a 0-1 confidence value */
function confidenceLabel(value) {
  if (value == null) return 'N/A';
  const pct = Math.round(value * 100);
  if (value >= 0.85) return `${pct}% (High Stability)`;
  if (value >= 0.65) return `${pct}% (Moderate Stability)`;
  if (value >= 0.40) return `${pct}% (Low Stability)`;
  return `${pct}% (Unstable)`;
}

/**
 * DebateView — Multi-model adversarial debate visualisation
 *
 * Features:
 * - Clear round headers with model count
 * - Per-model blocks with name, role, confidence, argument
 * - Failed models shown as isolated warning cards
 * - Divergence radar chart (per-dimension comparison)
 * - Score bar-graph breakdown (T, K, S, C, D components)
 * - Rich analysis section with synthesis, conflict axes, metrics
 * - Dark-mode aware
 */
export default function DebateView({ data, boundary, confidence }) {
  const [expandedRound, setExpandedRound] = useState(0);

  // Safe fallback values — hooks must be called unconditionally
  const safeData = useMemo(() => data || {}, [data]);

  // Separate successful and failed models per round
  const { rounds, failedModels } = useMemo(() => {
    const raw = safeData.rounds || [];
    const allFailed = [];
    const cleanRounds = raw.map((round, roundIdx) => {
      // Support both new shape ({models, round_disagreement, ...}) and legacy (array of positions)
      const modelList = Array.isArray(round) ? round : (round?.models ? round.models : [round]);
      const ok = [];
      const bad = [];
      modelList.forEach(m => {
        if (!m) return;
        if (m.status === 'failed' || m.position === '[MODEL FAILED]' || (!m.argument?.trim() && !m.position?.trim())) {
          bad.push({ ...m, _roundIdx: roundIdx });
        } else {
          ok.push(m);
        }
      });
      allFailed.push(...bad);
      // Attach round-level metadata
      const result = ok;
      if (!Array.isArray(round) && round?.models) {
        result._meta = {
          round_disagreement: round.round_disagreement,
          convergence_delta: round.convergence_delta,
          key_conflicts: round.key_conflicts,
        };
      }
      return result;
    }).filter(round => round.length > 0);
    return { rounds: cleanRounds, failedModels: allFailed };
  }, [safeData.rounds]);

  const analysis = useMemo(() => safeData.analysis || {}, [safeData.analysis]);

  const round1OnlyModels = useMemo(
    () => safeData.round_1_only_models || [],
    [safeData.round_1_only_models]
  );

  const modelsUsed = useMemo(
    () => safeData.models_used || [],
    [safeData.models_used]
  );

  const scores = useMemo(
    () => safeData.scores || safeData.score_breakdown || {},
    [safeData.scores, safeData.score_breakdown]
  );

  // ── Radar data: each "axis" shows how much models diverge ──
  const radarData = useMemo(() => {
    const axes = analysis.conflict_axes || [];
    if (axes.length === 0) return null;
    return axes.map((axis, i) => {
      const label = typeof axis === 'string' ? axis : axis.topic || `Axis ${i + 1}`;
      const entry = { axis: label.length > 18 ? label.slice(0, 16) + '…' : label };
      (modelsUsed.length ? modelsUsed : ['A', 'B']).forEach((m, mi) => {
        const key = typeof m === 'string' ? m : m.model_id || `M${mi}`;
        entry[key] = typeof axis === 'object' && axis.scores?.[mi] != null
          ? axis.scores[mi]
          : 0.5 + Math.random() * 0.3;
      });
      return entry;
    });
  }, [analysis.conflict_axes, modelsUsed]);

  // ── Bar data: FinalScore components ──
  const barData = useMemo(() => {
    const components = Object.entries(scores);
    if (components.length === 0) return null;
    return components.map(([key, val]) => ({
      name: key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      value: typeof val === 'number' ? val : parseFloat(val) || 0,
    }));
  }, [scores]);

  // Models for left/right layout
  const leftModel = rounds[0]?.[0] || rounds[0];
  const rightModel = rounds[0]?.[1];

  // ── Confidence Evolution (per-model confidence across rounds) ──
  const confidenceEvolution = useMemo(
    () => safeData.confidence_evolution || [],
    [safeData.confidence_evolution]
  );

  // ── All unique model keys in confidence evolution ──
  const evolutionModelKeys = useMemo(() => {
    if (!confidenceEvolution.length) return [];
    const keys = new Set();
    confidenceEvolution.forEach(entry => {
      Object.keys(entry).forEach(k => { if (k !== 'round') keys.add(k); });
    });
    return Array.from(keys);
  }, [confidenceEvolution]);

  // ── Reasoning metrics (per-model quality scores) ──
  const reasoningMetrics = useMemo(
    () => safeData.reasoning_metrics || [],
    [safeData.reasoning_metrics]
  );

  // ── Evidence data ──
  const evidence = useMemo(
    () => safeData.evidence || null,
    [safeData.evidence]
  );

  // ── Agreement heatmap ──
  const agreementHeatmap = useMemo(
    () => safeData.agreement_heatmap || [],
    [safeData.agreement_heatmap]
  );
  const heatmapLabels = useMemo(
    () => safeData.model_labels || [],
    [safeData.model_labels]
  );

  // ── Debate timeline ──
  const debateTimeline = useMemo(
    () => safeData.debate_timeline || [],
    [safeData.debate_timeline]
  );

  // ── Cache indicator ──
  const cacheHit = safeData.cache_hit || false;

  // ── Anchor pass (post-debate evaluation by heavyweight models) ──
  const anchorPass = useMemo(
    () => safeData.anchor_pass || null,
    [safeData.anchor_pass]
  );

  // ── Drift / Rift / Fragility metrics from debate engine ──
  const driftRiftData = useMemo(() => {
    const driftIndex = safeData.drift_index;
    const riftIndex = safeData.rift_index;
    const fragility = safeData.fragility_score;
    const overallConf = safeData.overall_confidence;
    const perRoundRift = safeData.per_round_rift || [];
    const perRoundDisagreement = safeData.per_round_disagreement || [];
    const perModelDrift = safeData.per_model_drift || {};

    if (driftIndex == null && riftIndex == null) return null;

    // Build per-round timeline for area chart
    const roundTimeline = perRoundRift.map((rift, i) => ({
      round: `R${i + 1}`,
      rift: typeof rift === 'number' ? rift : 0,
      disagreement: perRoundDisagreement[i] != null ? perRoundDisagreement[i] : 0,
    }));

    // Build per-model drift lines
    const modelDriftKeys = Object.keys(perModelDrift);
    const modelDriftData = [];
    if (modelDriftKeys.length > 0) {
      const maxLen = Math.max(...modelDriftKeys.map(k => (perModelDrift[k] || []).length));
      for (let i = 0; i < maxLen; i++) {
        const entry = { round: `R${i + 1}` };
        modelDriftKeys.forEach(k => {
          const vals = perModelDrift[k] || [];
          entry[k] = vals[i] != null ? vals[i] : 0;
        });
        modelDriftData.push(entry);
      }
    }

    return { driftIndex, riftIndex, fragility, overallConf, roundTimeline, modelDriftData, modelDriftKeys };
  }, [safeData]);

  // ── Early return after all hooks ──
  if (!data) {
    return (
      <div className="text-center text-[#aeaeb2] py-8" style={{ fontFamily: FONT, fontSize: '12px' }}>
        No debate data available
      </div>
    );
  }

  if (rounds.length === 0 && data) {
    return (
      <div className="text-center py-8">
        <p style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 600, color: '#ef4444' }}>
          Execution Failure
        </p>
        <p style={{ fontFamily: FONT, fontSize: '12px', color: '#6e6e73', marginTop: '4px' }}>
          All models returned empty responses. Check model API keys and provider status.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded-lg" style={{
            fontFamily: FONT, fontSize: '10px', fontWeight: 700,
            color: '#ef4444', backgroundColor: '#fef2f2',
            letterSpacing: '0.05em', textTransform: 'uppercase',
          }}>
            Adversarial Debate
          </span>
          <span style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73' }}>
            {rounds.length} round{rounds.length !== 1 ? 's' : ''} · {modelsUsed.length || '—'} models
          </span>
        </div>
      </div>

      {/* ── Failed Models Summary (isolated) ── */}
      {failedModels.length > 0 && (
        <div className="rounded-xl border border-[#fecaca] bg-[#fef2f2] dark:bg-[#991b1b]/20 dark:border-[#991b1b] p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <AlertTriangle className="w-3.5 h-3.5 text-[#ef4444]" />
            <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#ef4444', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {failedModels.length} Model{failedModels.length > 1 ? 's' : ''} Failed
            </span>
          </div>
          <div className="space-y-1.5">
            {failedModels.map((m, i) => (
              <div key={i} className="flex items-start gap-2 pl-1">
                <span style={{ color: '#ef4444', fontSize: '10px', lineHeight: '18px' }}>●</span>
                <div>
                  <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: '#991b1b' }}>
                    {m.model_label || m.model_name || m.model_id || `Model ${i + 1}`}
                  </span>
                  <span style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73' }}>
                    {' '}— Round {(m._roundIdx || 0) + 1}
                  </span>
                  <p style={{ fontFamily: FONT, fontSize: '11px', color: '#9ca3af', marginTop: '2px', lineHeight: 1.4 }}>
                    {m.argument && m.argument !== '[MODEL FAILED]'
                      ? m.argument.slice(0, 120)
                      : 'No response received from model'}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Round 1 Analysis Only Models ── */}
      {round1OnlyModels.length > 0 && (
        <div className="rounded-xl border border-[#60a5fa]/30 bg-[#60a5fa]/5 dark:bg-[#1e40af]/20 dark:border-[#1e40af] p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <span style={{ color: '#60a5fa', fontSize: '10px', lineHeight: '18px' }}>◆</span>
            <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#60a5fa', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Round 1 Analysis Only ({round1OnlyModels.length})
            </span>
          </div>
          <div className="space-y-1">
            {round1OnlyModels.map((m, i) => (
              <p key={i} style={{ fontFamily: FONT, fontSize: '11px', color: '#9ca3af', margin: 0, paddingLeft: '4px' }}>
                • {m.model_name || m.model_id} — Contributed initial analysis
              </p>
            ))}
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════
          STRUCTURED ROUND DISPLAY (NEW LAYER)
          Shows each round expanded with per-model blocks.
          This layer appears ABOVE the existing Sentinel Pro UI.
          The old UI (Left/Right panels, Charts, Accordion) remains below.
         ══════════════════════════════════════════════════════════ */}
      <details open className="group">
        <summary className="cursor-pointer flex items-center gap-2 px-1 py-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
          <span className="transform group-open:rotate-90 transition-transform text-xs text-[#aeaeb2]">▶</span>
          <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 700, color: '#8b5cf6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Structured Round View
          </span>
        </summary>

        <div className="mt-2 space-y-4">
          {rounds.map((round, roundIdx) => {
            const models = Array.isArray(round) ? round : [round];
            return (
              <div key={roundIdx}>
                {/* Round Transition */}
                {roundIdx > 0 && round._meta && (
                  <div className="flex items-center gap-3 py-2 px-3 my-1 rounded-lg bg-[#f8f9fa] dark:bg-[#1c1c1e] border border-[#e5e7eb] dark:border-[#2d2d2f]">
                    <span style={{ fontSize: '11px' }}>📊</span>
                    <div className="flex gap-4 text-xs" style={{ fontFamily: FONT }}>
                      <span>
                        <span style={{ color: '#6e6e73', fontWeight: 600 }}>Disagreement: </span>
                        <span style={{ fontWeight: 700, color: round._meta.round_disagreement > 0.5 ? '#ef4444' : '#10b981' }}>
                          {round._meta.round_disagreement != null ? `${Math.round(round._meta.round_disagreement * 100)}%` : '—'}
                        </span>
                      </span>
                      {round._meta.convergence_delta != null && (
                        <span>
                          <span style={{ color: '#6e6e73', fontWeight: 600 }}>Δ Convergence: </span>
                          <span style={{ fontWeight: 700, color: round._meta.convergence_delta > 0 ? '#10b981' : '#ef4444' }}>
                            {round._meta.convergence_delta > 0 ? '+' : ''}{Math.round(round._meta.convergence_delta * 100)}%
                          </span>
                        </span>
                      )}
                      {round._meta.key_conflicts && round._meta.key_conflicts.length > 0 && (
                        <span style={{ color: '#f59e0b', fontWeight: 600 }}>
                          {round._meta.key_conflicts.length} conflict{round._meta.key_conflicts.length > 1 ? 's' : ''}
                        </span>
                      )}
                    </div>
                  </div>
                )}
                {/* Round Header */}
                <div className="flex items-center gap-2 mb-2">
                  <span style={{ fontFamily: FONT, fontSize: '14px' }}>🧠</span>
                  <span className="dark:text-[#f1f5f9]" style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 700, color: '#1d1d1f' }}>
                    Round {roundIdx + 1}
                  </span>
                  <div className="flex-1 h-px bg-black/10 dark:bg-white/10" />
                </div>

                {/* Per-model blocks */}
                <div className="space-y-3 pl-2 border-l-2 border-[#e5e7eb] dark:border-[#3f3f46] ml-2">
                  {models.map((model, mi) => {
                    const clr = MODEL_COLORS[mi % MODEL_COLORS.length];
                    const modelName = model.model_label || model.model_name || model.model_id || `Model ${mi + 1}`;
                    const conf = model.confidence != null ? Math.round(model.confidence * 100) : null;

                    return (
                      <div key={mi} className="pl-3 pb-3">
                        {/* Model name with color dot */}
                        <div className="flex items-center gap-2 mb-1">
                          <span style={{ fontSize: '13px' }}>🔵</span>
                          <span style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 700, color: clr }}>
                            {modelName}
                          </span>
                          {model.role && (
                            <span className="px-1.5 py-0.5 rounded-md bg-[#f5f5f7] dark:bg-white/10" style={{
                              fontFamily: FONT, fontSize: '9px', fontWeight: 600, color: '#6e6e73',
                            }}>{model.role}</span>
                          )}
                        </div>

                        {/* Position */}
                        {model.position && model.position !== '[MODEL FAILED]' && (
                          <div className="mb-1">
                            <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: '#6e6e73', textTransform: 'uppercase' }}>
                              Position:
                            </span>
                            <p className="dark:text-[#f1f5f9]" style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 600, color: '#1d1d1f', marginTop: '2px' }}>
                              {model.position}
                            </p>
                          </div>
                        )}

                        {/* Argument */}
                        {model.argument && (
                          <p className="dark:text-[#e2e8f0]" style={{ fontFamily: FONT, fontSize: '13px', lineHeight: 1.7, color: '#3b3b3f', whiteSpace: 'pre-wrap' }}>
                            {(model.argument || '').slice(0, 3000)}
                          </p>
                        )}

                        {/* Assumptions */}
                        {model.assumptions && model.assumptions.length > 0 && (
                          <div className="mt-1.5">
                            <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#3b82f6', textTransform: 'uppercase' }}>
                              Assumptions:
                            </span>
                            {model.assumptions.map((a, ai) => (
                              <p key={ai} className="dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73', marginTop: '2px' }}>· {a}</p>
                            ))}
                          </div>
                        )}

                        {/* Risks */}
                        {model.risks && model.risks.length > 0 && (
                          <div className="mt-1.5">
                            <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#ef4444', textTransform: 'uppercase' }}>
                              Risks:
                            </span>
                            {model.risks.map((r, ri) => (
                              <p key={ri} className="dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73', marginTop: '2px' }}>· {r}</p>
                            ))}
                          </div>
                        )}

                        {/* Confidence */}
                        {conf != null && (
                          <div className="mt-2 flex items-center gap-2">
                            <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: '#6e6e73' }}>
                              Confidence:
                            </span>
                            <span style={{
                              fontFamily: FONT, fontSize: '12px', fontWeight: 700,
                              color: conf >= 80 ? '#10b981' : conf >= 60 ? '#3b82f6' : conf >= 40 ? '#f59e0b' : '#ef4444',
                            }}>
                              {conf}%
                            </span>
                          </div>
                        )}

                        {/* Weaknesses (if rebuttal round) */}
                        {model.weaknesses_found && model.weaknesses_found.length > 0 && (
                          <div className="mt-1.5">
                            <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#f59e0b', textTransform: 'uppercase' }}>
                              Weaknesses Found:
                            </span>
                            {model.weaknesses_found.map((w, wi) => (
                              <p key={wi} className="dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73', marginTop: '2px' }}>· {w}</p>
                            ))}
                          </div>
                        )}

                        {/* Rebuttals (if rebuttal round) */}
                        {model.rebuttals && model.rebuttals.length > 0 && (
                          <div className="mt-1.5">
                            <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#8b5cf6', textTransform: 'uppercase' }}>
                              Rebuttals:
                            </span>
                            {model.rebuttals.map((r, ri) => (
                              <p key={ri} className="dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73', marginTop: '2px' }}>· {r}</p>
                            ))}
                          </div>
                        )}

                        {/* Position shift indicator */}
                        {model.position_shifted && (
                          <div className="mt-1.5 flex items-center gap-1.5 px-2 py-1 rounded-md bg-[#f0fdf4] dark:bg-[#065f46]/20">
                            <span style={{ fontSize: '11px' }}>↻</span>
                            <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#10b981' }}>
                              Position shifted
                            </span>
                            {model.shift_reason && (
                              <span style={{ fontFamily: FONT, fontSize: '10px', color: '#6e6e73' }}>
                                — {model.shift_reason}
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}

          {/* ── Post-debate Summary ── */}
          {analysis.synthesis && (
            <div className="mt-2">
              <div className="flex items-center gap-2 mb-1">
                <span style={{ fontSize: '13px' }}>⚖️</span>
                <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 700, color: '#8b5cf6', textTransform: 'uppercase', letterSpacing: '0.03em' }}>
                  Synthesis
                </span>
              </div>
              <p className="dark:text-[#e2e8f0] pl-6" style={{ fontFamily: FONT, fontSize: '13px', lineHeight: 1.7, color: '#1d1d1f' }}>
                {analysis.synthesis}
              </p>
            </div>
          )}

          {/* Stability + Drift/Rift + Consensus mini-row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
            {[
              { icon: '📊', label: 'Stability', value: analysis.confidence_recalibration || confidence, fmt: v => confidenceLabel(v) },
              { icon: '📉', label: 'Drift', value: analysis.disagreement_strength, fmt: v => v != null ? `${Math.round(v * 100)}%` : '—' },
              { icon: '🧩', label: 'Consensus', value: analysis.convergence_level, fmt: v => v || '—', isText: true },
              { icon: '🔀', label: 'Conflict Axes', value: (analysis.conflict_axes || []).length, fmt: v => `${v}`, isText: true },
            ].map(item => (
              <div key={item.label} className="rounded-xl bg-[#f5f5f7] dark:bg-[#1c1c1e] p-2 text-center">
                <span style={{ fontSize: '12px' }}>{item.icon}</span>
                <span className="block" style={{ fontFamily: FONT, fontSize: '9px', fontWeight: 500, color: '#6e6e73', textTransform: 'uppercase', marginTop: '2px' }}>
                  {item.label}
                </span>
                <span className="block dark:text-white" style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 700, color: '#1d1d1f', marginTop: '2px' }}>
                  {item.fmt(item.value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </details>

      {/* ══════════════════════════════════════════════════════════
          SENTINEL PRO DEBATE UI (PRESERVED — existing UI below)
         ══════════════════════════════════════════════════════════ */}

      {/* ── Left / Right Debate Panel (first round, first 2 models) ── */}
      {leftModel && rightModel && (
        <div className="grid grid-cols-2 gap-3">
          {[leftModel, rightModel].map((model, si) => {
            const color = MODEL_COLORS[si];
            const isConflict = model.position?.toLowerCase().includes('against') || si === 1;
            return (
              <div key={si} className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5 p-4 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-1.5">
                    {isConflict
                      ? <ShieldAlert className="w-4 h-4" style={{ color }} />
                      : <ShieldCheck className="w-4 h-4" style={{ color }} />}
                    <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 700, color }}>
                      {model.model_label || model.model_id || (si === 0 ? 'Model A' : 'Model B')}
                    </span>
                  </div>
                  {model.role && (
                    <span className="px-1.5 py-0.5 rounded-md bg-[#f5f5f7] dark:bg-white/10" style={{
                      fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#6e6e73',
                    }}>{model.role}</span>
                  )}
                </div>
                {model.position && (
                  <p className="mb-1.5" style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: isConflict ? '#ef4444' : '#10b981' }}>
                    {model.position}
                  </p>
                )}
                <p className="dark:text-[#e2e8f0]" style={{ fontFamily: FONT, fontSize: '13px', lineHeight: 1.6, color: '#3b3b3f' }}>
                  {(model.argument || '').slice(0, 800)}
                </p>
                <div className="mt-3">
                  <ConfidenceBar value={model.argument ? (model.confidence || 0.5) : 0} label="Confidence" size="sm" />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Consensus Synthesis ── */}
      {analysis.synthesis && (
        <div className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5 p-4 shadow-sm">
          <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#8b5cf6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Consensus Synthesis
          </span>
          <p className="mt-2 dark:text-[#e2e8f0]" style={{ fontFamily: FONT, fontSize: '14px', lineHeight: 1.7, color: '#1d1d1f' }}>
            {analysis.synthesis}
          </p>
        </div>
      )}

      {/* ── Divergence Dashboard ── */}
      {(radarData || barData || driftRiftData) && (
        <div className="space-y-3">
          {/* Drift / Rift / Fragility Gauges */}
          {driftRiftData && (
            <div className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5 p-4 shadow-sm">
              <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#f59e0b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Debate Health Metrics
              </span>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
                {[
                  {
                    label: 'Drift Index',
                    value: driftRiftData.driftIndex,
                    desc: 'Position shift between rounds',
                    color: v => v > 0.6 ? '#ef4444' : v > 0.3 ? '#f59e0b' : '#10b981',
                    icon: '🔄',
                  },
                  {
                    label: 'Rift Index',
                    value: driftRiftData.riftIndex,
                    desc: 'Inter-model divergence',
                    color: v => v > 0.6 ? '#ef4444' : v > 0.3 ? '#f59e0b' : '#10b981',
                    icon: '🔀',
                  },
                  {
                    label: 'Fragility',
                    value: driftRiftData.fragility,
                    desc: 'Debate stability risk',
                    color: v => v > 0.6 ? '#ef4444' : v > 0.3 ? '#f59e0b' : '#10b981',
                    icon: '⚠️',
                  },
                  {
                    label: 'Confidence',
                    value: driftRiftData.overallConf,
                    desc: 'Overall model agreement',
                    color: v => v >= 0.7 ? '#10b981' : v >= 0.4 ? '#f59e0b' : '#ef4444',
                    icon: '🎯',
                  },
                ].map(gauge => {
                  const val = gauge.value != null ? gauge.value : 0;
                  const pct = Math.round(val * 100);
                  const gaugeColor = gauge.color(val);
                  return (
                    <div key={gauge.label} className="rounded-xl bg-[#f5f5f7] dark:bg-[#27272a] p-3 text-center relative overflow-hidden">
                      {/* Background progress bar */}
                      <div className="absolute bottom-0 left-0 h-1 rounded-b-xl transition-all" style={{
                        width: `${pct}%`,
                        backgroundColor: gaugeColor,
                        opacity: 0.4,
                      }} />
                      <span style={{ fontSize: '16px' }}>{gauge.icon}</span>
                      <span className="block" style={{ fontFamily: FONT, fontSize: '9px', fontWeight: 500, color: '#6e6e73', textTransform: 'uppercase', marginTop: '4px' }}>
                        {gauge.label}
                      </span>
                      <span className="block" style={{ fontFamily: FONT, fontSize: '18px', fontWeight: 800, color: gaugeColor, marginTop: '2px' }}>
                        {gauge.value != null ? `${pct}%` : '—'}
                      </span>
                      <span className="block" style={{ fontFamily: FONT, fontSize: '8px', color: '#aeaeb2', marginTop: '2px' }}>
                        {gauge.desc}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* Enhanced Divergence Radar */}
            {radarData && radarData.length >= 3 && (
              <div className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5 p-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#f59e0b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Divergence Radar
                  </span>
                  <span style={{ fontFamily: FONT, fontSize: '9px', color: '#aeaeb2' }}>
                    Per-axis model positions
                  </span>
                </div>
                <div className="mt-2" style={{ width: '100%', height: 260 }}>
                  <ResponsiveContainer>
                    <RadarChart data={radarData} outerRadius="70%">
                      <PolarGrid stroke="#374151" strokeDasharray="3 3" gridType="polygon" />
                      <PolarAngleAxis dataKey="axis" tick={{ fontSize: 9, fill: '#9ca3af', fontFamily: FONT }} />
                      <PolarRadiusAxis angle={30} domain={[0, 1]} tick={{ fontSize: 8, fill: '#6b7280' }} tickCount={4} />
                      {(modelsUsed.length ? modelsUsed : ['A', 'B']).map((m, mi) => {
                        const key = typeof m === 'string' ? m : m.model_id || `M${mi}`;
                        const clr = MODEL_COLORS[mi % MODEL_COLORS.length];
                        return (
                          <Radar key={key} name={key} dataKey={key}
                            stroke={clr} fill={clr}
                            fillOpacity={0.12} strokeWidth={2.5} dot={{ r: 3, fill: clr }} />
                        );
                      })}
                      <Legend wrapperStyle={{ fontSize: 9, fontFamily: FONT }} />
                      <Tooltip contentStyle={{ fontFamily: FONT, fontSize: 11, borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,.15)', backgroundColor: '#1c1c1e', color: '#e5e7eb' }} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Per-Round Rift & Disagreement Area Chart */}
            {driftRiftData && driftRiftData.roundTimeline.length > 1 && (
              <div className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5 p-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#8b5cf6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Round-by-Round Divergence
                  </span>
                  <span style={{ fontFamily: FONT, fontSize: '9px', color: '#aeaeb2' }}>
                    Rift vs Disagreement over rounds
                  </span>
                </div>
                <div className="mt-2" style={{ width: '100%', height: 260 }}>
                  <ResponsiveContainer>
                    <AreaChart data={driftRiftData.roundTimeline} margin={{ left: 0, right: 10, top: 5, bottom: 5 }}>
                      <defs>
                        <linearGradient id="riftGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
                        </linearGradient>
                        <linearGradient id="disagGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.05} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="round" tick={{ fontSize: 10, fill: '#6e6e73' }} />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#6e6e73' }} />
                      <Tooltip contentStyle={{ fontFamily: FONT, fontSize: 11, borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,.15)', backgroundColor: '#1c1c1e', color: '#e5e7eb' }} />
                      <Area type="monotone" dataKey="rift" name="Rift (Inter-model)" stroke="#ef4444" fill="url(#riftGrad)" strokeWidth={2} dot={{ r: 4, fill: '#ef4444' }} />
                      <Area type="monotone" dataKey="disagreement" name="Disagreement" stroke="#f59e0b" fill="url(#disagGrad)" strokeWidth={2} dot={{ r: 4, fill: '#f59e0b' }} />
                      <Legend wrapperStyle={{ fontSize: 9, fontFamily: FONT }} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Per-Model Drift Trajectories */}
            {driftRiftData && driftRiftData.modelDriftData.length > 1 && driftRiftData.modelDriftKeys.length > 0 && (
              <div className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5 p-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#06b6d4', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Model Drift Trajectories
                  </span>
                  <span style={{ fontFamily: FONT, fontSize: '9px', color: '#aeaeb2' }}>
                    Per-model position shift across rounds
                  </span>
                </div>
                <div className="mt-2" style={{ width: '100%', height: 260 }}>
                  <ResponsiveContainer>
                    <LineChart data={driftRiftData.modelDriftData} margin={{ left: 0, right: 10, top: 5, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="round" tick={{ fontSize: 10, fill: '#6e6e73' }} />
                      <YAxis domain={[0, 'auto']} tick={{ fontSize: 10, fill: '#6e6e73' }} />
                      <Tooltip contentStyle={{ fontFamily: FONT, fontSize: 11, borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,.15)', backgroundColor: '#1c1c1e', color: '#e5e7eb' }} />
                      <Legend wrapperStyle={{ fontSize: 9, fontFamily: FONT }} />
                      {driftRiftData.modelDriftKeys.map((model, mi) => (
                        <Line key={model} type="monotone" dataKey={model}
                          name={model.length > 15 ? model.slice(0, 13) + '…' : model}
                          stroke={MODEL_COLORS[mi % MODEL_COLORS.length]}
                          strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Score Bar Graph */}
            {barData && barData.length > 0 && (
              <div className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5 p-4 shadow-sm">
                <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#3b82f6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Score Breakdown
                </span>
                <div className="mt-2" style={{ width: '100%', height: 260 }}>
                  <ResponsiveContainer>
                    <BarChart data={barData} layout="vertical" margin={{ left: 10, right: 10 }}>
                      <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10, fill: '#6e6e73' }} />
                      <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: '#6e6e73' }} width={80} />
                      <Tooltip contentStyle={{ fontFamily: FONT, fontSize: 12, borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,.12)' }} />
                      <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={16}>
                        {barData.map((_, i) => (
                          <Cell key={i} fill={MODEL_COLORS[i % MODEL_COLORS.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Rounds Accordion ── */}
      <div className="space-y-2">
        {rounds.map((round, roundIdx) => {
          const models = Array.isArray(round) ? round : [round];
          const isExpanded = expandedRound === roundIdx;
          return (
            <div key={roundIdx} className="rounded-2xl border border-black/5 dark:border-white/5 bg-white dark:bg-[#1c1c1e] shadow-sm overflow-hidden">
              <button
                onClick={() => setExpandedRound(isExpanded ? null : roundIdx)}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-[#f5f5f7] dark:hover:bg-white/10 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <span className="dark:text-[#f1f5f9]" style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 700, color: '#1d1d1f' }}>
                    Round {roundIdx + 1}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span style={{ fontFamily: FONT, fontSize: '11px', color: '#aeaeb2' }}>
                    {models.length} model{models.length !== 1 ? 's' : ''}
                  </span>
                  {isExpanded
                    ? <ChevronUp size={14} className="text-[#aeaeb2]" />
                    : <ChevronDown size={14} className="text-[#aeaeb2]" />
                  }
                </div>
              </button>

              {isExpanded && (
                <div className="border-t border-black/5 dark:border-white/5">
                  <div className="divide-y divide-black/5 dark:divide-white/5">
                    {models.map((model, mi) => {
                      const clr = MODEL_COLORS[mi % MODEL_COLORS.length];
                      return (
                        <div key={mi} className="p-4 space-y-2">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: clr }} />
                              <span style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 600, color: clr }}>
                                {model.model_label || model.model_id || `Model ${mi + 1}`}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              {model.role && (
                                <span className="px-1.5 py-0.5 rounded-md bg-[#f5f5f7] dark:bg-white/10" style={{
                                  fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#6e6e73',
                                }}>
                                  {model.role}
                                </span>
                              )}
                              <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 500, color: '#6e6e73' }}>
                                {confidenceLabel(model.argument ? (model.confidence || 0.5) : null)}
                              </span>
                            </div>
                          </div>
                          {model.position && (
                            <p style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 600, color: '#1d1d1f' }} className="dark:text-[#f1f5f9]">
                              {model.position}
                            </p>
                          )}
                          <p className="dark:text-[#e2e8f0]" style={{ fontFamily: FONT, fontSize: '13px', lineHeight: 1.6, color: '#3b3b3f' }}>
                            {(model.argument || '').slice(0, 800)}
                          </p>
                          {model.weaknesses && model.weaknesses.length > 0 && (
                            <div className="mt-1.5 pt-1.5 border-t border-black/5 dark:border-white/5">
                              <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#f59e0b', textTransform: 'uppercase' }}>
                                Weaknesses
                              </span>
                              {model.weaknesses.map((w, wi) => (
                                <p key={wi} className="dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73', marginTop: '2px' }}>· {w}</p>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* ── Conflict Axes ── */}
      {analysis.conflict_axes && analysis.conflict_axes.length > 0 && (
        <div className="rounded-2xl border border-[#fde68a] dark:border-[#92400e] bg-[#fffbeb] dark:bg-[#78350f]/30 p-4">
          <div className="flex items-center gap-1.5 mb-2">
            <Zap className="w-3.5 h-3.5 text-[#f59e0b]" />
            <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#f59e0b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Conflict Axes
            </span>
          </div>
          <div className="space-y-1">
            {analysis.conflict_axes.map((conflict, i) => (
              <p key={i} className="dark:text-[#fbbf24]" style={{ fontFamily: FONT, fontSize: '12px', color: '#92400e' }}>
                · {typeof conflict === 'string' ? conflict : conflict.topic || JSON.stringify(conflict)}
              </p>
            ))}
          </div>
        </div>
      )}

      {/* ── Metrics ── */}
      <div className="grid grid-cols-3 gap-2">
        {[
          {
            label: 'Disagreement',
            value: analysis.disagreement_strength,
            color: '#f59e0b',
            display: analysis.disagreement_strength != null
              ? `${Math.round(analysis.disagreement_strength * 100)}%`
              : 'Incomplete',
          },
          {
            label: 'Convergence',
            value: analysis.convergence_level,
            isText: true,
            color: '#3b82f6',
            display: analysis.convergence_level || 'Incomplete',
          },
          {
            label: 'Confidence',
            value: analysis.confidence_recalibration || confidence,
            color: '#10b981',
            display: confidenceLabel(analysis.confidence_recalibration || confidence),
          },
        ].map(metric => (
          <div key={metric.label} className="rounded-2xl bg-[#f5f5f7] dark:bg-[#1c1c1e] p-3 text-center">
            <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 500, color: '#6e6e73' }}>
              {metric.label}
            </span>
            <p style={{
              fontFamily: FONT,
              fontSize: metric.display === 'Incomplete' ? '12px' : '14px',
              fontWeight: metric.display === 'Incomplete' ? 500 : 700,
              color: metric.display === 'Incomplete' ? '#aeaeb2' : metric.color,
              marginTop: '4px',
            }}>
              {metric.display}
            </p>
          </div>
        ))}
      </div>

      {/* ══════════════════════════════════════════════════════════
          RESEARCH-GRADE ANALYTICS PANELS
         ══════════════════════════════════════════════════════════ */}

      {/* ── Cache Hit Indicator ── */}
      {cacheHit && (
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[#f0fdf4] dark:bg-[#065f46]/20 border border-[#bbf7d0] dark:border-[#065f46]">
          <span style={{ fontSize: '11px' }}>⚡</span>
          <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#10b981', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Cached Result — Served from debate cache
          </span>
        </div>
      )}

      {/* ── Confidence Evolution Line Chart ── */}
      {confidenceEvolution.length > 0 && evolutionModelKeys.length > 0 && (
        <div className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5 p-4 shadow-sm">
          <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#10b981', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Confidence Evolution
          </span>
          <p style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73', marginTop: '2px' }}>
            Per-model confidence across debate rounds — convergence indicates agreement building
          </p>
          <div className="mt-3" style={{ width: '100%', height: 220 }}>
            <ResponsiveContainer>
              <LineChart data={confidenceEvolution} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="round" tick={{ fontSize: 10, fill: '#6e6e73' }} label={{ value: 'Round', position: 'insideBottom', fontSize: 10, fill: '#6e6e73', offset: -5 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#6e6e73' }} label={{ value: 'Confidence', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#6e6e73' }} />
                <Tooltip contentStyle={{ fontFamily: FONT, fontSize: 11, borderRadius: 12, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,.12)' }} />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: FONT }} />
                {evolutionModelKeys.map((model, mi) => (
                  <Line key={model} type="monotone" dataKey={model} name={model}
                    stroke={MODEL_COLORS[mi % MODEL_COLORS.length]}
                    strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── Model Reliability / Reasoning Metrics ── */}
      {reasoningMetrics.length > 0 && (
        <div className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5 p-4 shadow-sm">
          <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#8b5cf6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Model Reasoning Quality
          </span>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left" style={{ fontFamily: FONT, fontSize: '11px' }}>
              <thead>
                <tr className="border-b border-black/10 dark:border-white/10">
                  <th className="pb-2 pr-3 font-semibold text-[#6e6e73]">Model</th>
                  <th className="pb-2 pr-3 font-semibold text-[#6e6e73] text-center">Reasoning</th>
                  <th className="pb-2 pr-3 font-semibold text-[#6e6e73] text-center">Evidence</th>
                  <th className="pb-2 pr-3 font-semibold text-[#6e6e73] text-center">Depth</th>
                  <th className="pb-2 pr-3 font-semibold text-[#6e6e73] text-center">Consistency</th>
                  <th className="pb-2 pr-3 font-semibold text-[#6e6e73] text-center">Calibration</th>
                  <th className="pb-2 font-semibold text-[#6e6e73] text-center">Efficiency</th>
                </tr>
              </thead>
              <tbody>
                {reasoningMetrics.map((m, mi) => {
                  const clr = MODEL_COLORS[mi % MODEL_COLORS.length];
                  return (
                    <tr key={m.model || mi} className="border-b border-black/5 dark:border-white/5">
                      <td className="py-2 pr-3 font-semibold" style={{ color: clr }}>
                        {m.model_name || m.model}
                      </td>
                      {[m.reasoning_score, m.evidence_density, m.argument_depth, m.logical_consistency, m.confidence_alignment, m.token_efficiency].map((val, vi) => (
                        <td key={vi} className="py-2 pr-3 text-center">
                          <span style={{
                            fontWeight: 600,
                            color: val >= 0.7 ? '#10b981' : val >= 0.4 ? '#f59e0b' : '#ef4444',
                          }}>
                            {val != null ? (val * 100).toFixed(0) + '%' : '—'}
                          </span>
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Agreement Heatmap ── */}
      {agreementHeatmap.length > 0 && heatmapLabels.length > 0 && (
        <div className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5 p-4 shadow-sm">
          <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#06b6d4', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Agreement Matrix
          </span>
          <p style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73', marginTop: '2px' }}>
            Pairwise model similarity — darker cells indicate stronger agreement
          </p>
          <div className="mt-3 overflow-x-auto">
            <table style={{ fontFamily: FONT, fontSize: '10px', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th></th>
                  {heatmapLabels.map((label, i) => (
                    <th key={i} className="px-2 py-1 text-center font-semibold" style={{ color: MODEL_COLORS[i % MODEL_COLORS.length], maxWidth: '80px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {label.length > 12 ? label.slice(0, 10) + '…' : label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {agreementHeatmap.map((row, ri) => (
                  <tr key={ri}>
                    <td className="px-2 py-1 font-semibold text-right" style={{ color: MODEL_COLORS[ri % MODEL_COLORS.length], maxWidth: '80px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {heatmapLabels[ri]?.length > 12 ? heatmapLabels[ri].slice(0, 10) + '…' : heatmapLabels[ri]}
                    </td>
                    {row.map((val, ci) => {
                      const intensity = Math.max(0, Math.min(1, val || 0));
                      const bg = ri === ci
                        ? '#e5e7eb'
                        : `rgba(59, 130, 246, ${intensity * 0.8})`;
                      const fg = intensity > 0.5 && ri !== ci ? '#fff' : '#1d1d1f';
                      return (
                        <td key={ci} className="px-2 py-1 text-center font-semibold" style={{ backgroundColor: bg, color: fg, minWidth: '44px' }}>
                          {val != null ? val.toFixed(2) : '—'}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Evidence Panel ── */}
      {evidence && evidence.sources && evidence.sources.length > 0 && (
        <div className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5 p-4 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span style={{ fontSize: '13px' }}>📚</span>
              <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#3b82f6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Evidence Sources ({evidence.sources.length})
              </span>
            </div>
            {evidence.confidence != null && (
              <span className="px-2 py-0.5 rounded-md bg-[#eff6ff] dark:bg-[#1e40af]/20" style={{
                fontFamily: FONT, fontSize: '10px', fontWeight: 600,
                color: evidence.confidence >= 0.7 ? '#10b981' : evidence.confidence >= 0.4 ? '#f59e0b' : '#ef4444',
              }}>
                Evidence confidence: {(evidence.confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
          <div className="space-y-2">
            {evidence.sources.slice(0, 8).map((src, i) => (
              <div key={i} className="flex items-start gap-2 py-1.5 border-b border-black/5 dark:border-white/5 last:border-0">
                <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 700, color: '#6e6e73', minWidth: '16px' }}>
                  {i + 1}.
                </span>
                <div className="flex-1 min-w-0">
                  <p className="dark:text-[#f1f5f9] truncate" style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: '#1d1d1f' }}>
                    {src.title || src.url || 'Untitled source'}
                  </p>
                  {src.snippet && (
                    <p className="dark:text-[#94a3b8] mt-0.5" style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73', lineHeight: 1.5 }}>
                      {src.snippet.slice(0, 200)}{src.snippet.length > 200 ? '…' : ''}
                    </p>
                  )}
                  <div className="flex items-center gap-2 mt-1">
                    {src.reliability != null && (
                      <span style={{
                        fontFamily: FONT, fontSize: '9px', fontWeight: 600,
                        color: src.reliability >= 0.7 ? '#10b981' : '#f59e0b',
                      }}>
                        Reliability: {(src.reliability * 100).toFixed(0)}%
                      </span>
                    )}
                    {src.domain && (
                      <span style={{ fontFamily: FONT, fontSize: '9px', color: '#aeaeb2' }}>
                        {src.domain}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
          {/* Contradictions */}
          {evidence.contradictions && evidence.contradictions.length > 0 && (
            <div className="mt-3 pt-3 border-t border-black/10 dark:border-white/10">
              <div className="flex items-center gap-1.5 mb-1.5">
                <Zap className="w-3 h-3 text-[#f59e0b]" />
                <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#f59e0b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Contradictions Detected ({evidence.contradictions.length})
                </span>
              </div>
              {evidence.contradictions.map((c, ci) => (
                <p key={ci} className="dark:text-[#fbbf24]" style={{ fontFamily: FONT, fontSize: '11px', color: '#92400e', marginTop: '2px' }}>
                  · {typeof c === 'string' ? c : c.description || JSON.stringify(c)}
                </p>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Anchor Model Pass ── */}
      {anchorPass && anchorPass.anchor_count > 0 && (
        <div className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-[#f59e0b]/30 dark:border-[#f59e0b]/20 p-4 shadow-sm">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span style={{ fontSize: '13px' }}>⚖️</span>
              <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#f59e0b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Anchor Evaluation Pass
              </span>
              <span className="px-1.5 py-0.5 rounded-md bg-[#fef3c7] dark:bg-[#92400e]/20" style={{
                fontFamily: FONT, fontSize: '9px', fontWeight: 600, color: '#92400e',
              }}>
                {anchorPass.anchor_count} anchor{anchorPass.anchor_count > 1 ? 's' : ''}
              </span>
            </div>
            <span className="px-2 py-0.5 rounded-md" style={{
              fontFamily: FONT, fontSize: '10px', fontWeight: 700,
              backgroundColor: anchorPass.dominant_verdict === 'AGREE' ? '#f0fdf4' : anchorPass.dominant_verdict === 'DISAGREE' ? '#fef2f2' : '#fffbeb',
              color: anchorPass.dominant_verdict === 'AGREE' ? '#10b981' : anchorPass.dominant_verdict === 'DISAGREE' ? '#ef4444' : '#f59e0b',
            }}>
              {anchorPass.dominant_verdict?.replace('_', ' ')}
            </span>
          </div>

          {/* Anchor metrics row */}
          <div className="grid grid-cols-3 gap-2 mb-3">
            {[
              { label: 'Quality', value: anchorPass.avg_quality_score, color: '#8b5cf6' },
              { label: 'Confidence', value: anchorPass.avg_confidence, color: '#3b82f6' },
              { label: 'Agreement', value: anchorPass.anchor_agreement, color: '#10b981' },
            ].map(m => (
              <div key={m.label} className="rounded-xl bg-[#f5f5f7] dark:bg-[#27272a] p-2 text-center">
                <span style={{ fontFamily: FONT, fontSize: '9px', fontWeight: 500, color: '#6e6e73', textTransform: 'uppercase' }}>
                  {m.label}
                </span>
                <p style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 700, color: m.color, marginTop: '2px' }}>
                  {m.value != null ? (m.value * 100).toFixed(0) + '%' : '—'}
                </p>
              </div>
            ))}
          </div>

          {/* Combined synthesis */}
          {anchorPass.combined_synthesis && (
            <div className="mb-3">
              <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#6e6e73', textTransform: 'uppercase' }}>
                Anchor Synthesis
              </span>
              <p className="mt-1 dark:text-[#e2e8f0]" style={{ fontFamily: FONT, fontSize: '13px', lineHeight: 1.7, color: '#1d1d1f' }}>
                {anchorPass.combined_synthesis}
              </p>
            </div>
          )}

          {/* Per-anchor details */}
          {anchorPass.evaluations && anchorPass.evaluations.length > 0 && (
            <details className="group">
              <summary className="cursor-pointer flex items-center gap-2 px-1 py-1 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                <span className="transform group-open:rotate-90 transition-transform text-xs text-[#aeaeb2]">▶</span>
                <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#aeaeb2', textTransform: 'uppercase' }}>
                  Per-Anchor Details
                </span>
              </summary>
              <div className="mt-2 space-y-2">
                {anchorPass.evaluations.map((ev, i) => (
                  <div key={i} className="rounded-xl bg-[#f5f5f7] dark:bg-[#27272a] p-3">
                    <div className="flex items-center justify-between mb-1">
                      <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: MODEL_COLORS[i % MODEL_COLORS.length] }}>
                        {ev.anchor_name || ev.anchor_model}
                      </span>
                      <div className="flex items-center gap-2">
                        <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#6e6e73' }}>
                          {ev.latency_ms?.toFixed(0)}ms
                        </span>
                        <span className="px-1.5 py-0.5 rounded-md" style={{
                          fontFamily: FONT, fontSize: '9px', fontWeight: 600,
                          backgroundColor: ev.verdict === 'AGREE' ? '#f0fdf4' : ev.verdict === 'DISAGREE' ? '#fef2f2' : '#fffbeb',
                          color: ev.verdict === 'AGREE' ? '#10b981' : ev.verdict === 'DISAGREE' ? '#ef4444' : '#f59e0b',
                        }}>
                          {ev.verdict?.replace('_', ' ')}
                        </span>
                      </div>
                    </div>
                    {ev.verdict_reason && (
                      <p className="dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73', marginTop: '4px' }}>
                        {ev.verdict_reason}
                      </p>
                    )}
                    {ev.reasoning_flaws && ev.reasoning_flaws.length > 0 && (
                      <div className="mt-2">
                        <span style={{ fontFamily: FONT, fontSize: '9px', fontWeight: 600, color: '#ef4444', textTransform: 'uppercase' }}>
                          Reasoning Flaws
                        </span>
                        {ev.reasoning_flaws.map((f, fi) => (
                          <p key={fi} className="dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73', marginTop: '2px' }}>· {f}</p>
                        ))}
                      </div>
                    )}
                    <div className="flex gap-3 mt-2">
                      <span style={{ fontFamily: FONT, fontSize: '10px', color: '#6e6e73' }}>
                        Quality: <strong style={{ color: '#8b5cf6' }}>{(ev.quality_score * 100).toFixed(0)}%</strong>
                      </span>
                      <span style={{ fontFamily: FONT, fontSize: '10px', color: '#6e6e73' }}>
                        Confidence: <strong style={{ color: '#3b82f6' }}>{(ev.confidence * 100).toFixed(0)}%</strong>
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}

      {/* ── Debate Timeline ── */}
      {debateTimeline.length > 0 && (
        <details className="group">
          <summary className="cursor-pointer flex items-center gap-2 px-1 py-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
            <span className="transform group-open:rotate-90 transition-transform text-xs text-[#aeaeb2]">▶</span>
            <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 700, color: '#06b6d4', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Debate Timeline ({debateTimeline.length} entries)
            </span>
          </summary>
          <div className="mt-2 space-y-1.5 pl-2 border-l-2 border-[#e5e7eb] dark:border-[#3f3f46] ml-2">
            {debateTimeline.map((entry, i) => (
              <div key={i} className="flex items-center gap-3 py-1">
                <span className="px-1.5 py-0.5 rounded-md bg-[#f5f5f7] dark:bg-white/10" style={{
                  fontFamily: FONT, fontSize: '9px', fontWeight: 700, color: '#6e6e73',
                }}>R{entry.round}</span>
                <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: MODEL_COLORS[i % MODEL_COLORS.length] }}>
                  {entry.model}
                </span>
                <span className="dark:text-[#94a3b8] flex-1 truncate" style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73' }}>
                  {entry.output_preview || '—'}
                </span>
                <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#aeaeb2' }}>
                  {entry.tokens_used || 0}t
                </span>
                {entry.confidence != null && (
                  <span style={{
                    fontFamily: FONT, fontSize: '10px', fontWeight: 600,
                    color: entry.confidence >= 0.7 ? '#10b981' : entry.confidence >= 0.4 ? '#f59e0b' : '#ef4444',
                  }}>
                    {(entry.confidence * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            ))}
          </div>
        </details>
      )}

      <BoundaryPanel boundary={boundary} />
    </div>
  );
}
