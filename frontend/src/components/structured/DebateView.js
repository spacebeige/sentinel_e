import React, { useState, useMemo } from 'react';
import { ChevronDown, ChevronUp, ShieldAlert, ShieldCheck, Zap, AlertTriangle } from 'lucide-react';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer,
         BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';
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
      const models = Array.isArray(round) ? round : [round];
      const ok = [];
      const bad = [];
      models.forEach(m => {
        if (!m) return;
        if (m.status === 'failed' || m.position === '[MODEL FAILED]' || (!m.argument?.trim() && !m.position?.trim())) {
          bad.push({ ...m, _roundIdx: roundIdx });
        } else {
          ok.push(m);
        }
      });
      allFailed.push(...bad);
      return ok;
    }).filter(round => round.length > 0);
    return { rounds: cleanRounds, failedModels: allFailed };
  }, [safeData.rounds]);

  const analysis = useMemo(() => safeData.analysis || {}, [safeData.analysis]);

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
          <div className="space-y-1">
            {failedModels.map((m, i) => (
              <div key={i} className="flex items-center gap-2">
                <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: '#991b1b' }}>
                  {m.model_label || m.model_name || m.model_id || `Model ${i + 1}`}
                </span>
                <span style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73' }}>
                  — Round {(m._roundIdx || 0) + 1} · {m.argument && m.argument !== '[MODEL FAILED]' ? m.argument.slice(0, 80) : 'No response received'}
                </span>
              </div>
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
                          <p className="dark:text-[#e2e8f0]" style={{ fontFamily: FONT, fontSize: '13px', lineHeight: 1.6, color: '#3b3b3f' }}>
                            {(model.argument || '').slice(0, 1000)}
                          </p>
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

      {/* ── Charts Row: Radar + Bar ── */}
      {(radarData || barData) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* Divergence Radar */}
          {radarData && radarData.length >= 3 && (
            <div className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5 p-4 shadow-sm">
              <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#f59e0b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Divergence Radar
              </span>
              <div className="mt-2" style={{ width: '100%', height: 220 }}>
                <ResponsiveContainer>
                  <RadarChart data={radarData} outerRadius="70%">
                    <PolarGrid stroke="#e5e7eb" />
                    <PolarAngleAxis dataKey="axis" tick={{ fontSize: 10, fill: '#6e6e73' }} />
                    {(modelsUsed.length ? modelsUsed : ['A', 'B']).map((m, mi) => {
                      const key = typeof m === 'string' ? m : m.model_id || `M${mi}`;
                      return (
                        <Radar key={key} name={key} dataKey={key}
                          stroke={MODEL_COLORS[mi % MODEL_COLORS.length]}
                          fill={MODEL_COLORS[mi % MODEL_COLORS.length]}
                          fillOpacity={0.15} strokeWidth={2} />
                      );
                    })}
                  </RadarChart>
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
              <div className="mt-2" style={{ width: '100%', height: 220 }}>
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

      <BoundaryPanel boundary={boundary} />
    </div>
  );
}
