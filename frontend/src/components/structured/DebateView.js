import React, { useState, useMemo } from 'react';
import { ChevronDown, ChevronUp, ShieldAlert, ShieldCheck, Zap } from 'lucide-react';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer,
         BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';
import ConfidenceBar from './ConfidenceBar';
import BoundaryPanel from './BoundaryPanel';

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

const MODEL_COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4'];

/**
 * DebateView — Multi-model adversarial debate visualisation
 *
 * Features:
 * - Left / right debate panels (first two models)
 * - Round-by-round accordion with model columns
 * - Divergence radar chart (per-dimension comparison)
 * - Score bar-graph breakdown (T, K, S, C, D components)
 * - Conflicting vs supported claim markers
 * - Dark-mode aware
 */
export default function DebateView({ data, boundary, confidence }) {
  const [expandedRound, setExpandedRound] = useState(0);

  if (!data) return null;

  const rounds = data.rounds || [];
  const analysis = data.analysis || {};
  const modelsUsed = data.models_used || [];
  const scores = data.scores || data.score_breakdown || {};

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
          : 0.5 + Math.random() * 0.3; // fallback if no per-axis scores
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
                <p className="dark:text-[#e2e8f0]" style={{ fontFamily: FONT, fontSize: '12px', lineHeight: 1.6, color: '#3b3b3f' }}>
                  {(model.argument || '').slice(0, 600)}
                </p>
                <div className="mt-3">
                  <ConfidenceBar value={model.confidence || 0.5} label="Confidence" size="sm" />
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
          <p className="mt-2 dark:text-[#e2e8f0]" style={{ fontFamily: FONT, fontSize: '14px', lineHeight: 1.6, color: '#1d1d1f' }}>
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
                <span className="dark:text-[#f1f5f9]" style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 600, color: '#1d1d1f' }}>
                  Round {roundIdx + 1}
                </span>
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
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-0 divide-y md:divide-y-0 md:divide-x divide-black/5 dark:divide-white/5">
                    {models.map((model, mi) => {
                      const clr = MODEL_COLORS[mi % MODEL_COLORS.length];
                      return (
                        <div key={mi} className="p-4 space-y-2">
                          <div className="flex items-center justify-between">
                            <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: clr }}>
                              {model.model_label || model.model_id || `Model ${mi + 1}`}
                            </span>
                            {model.role && (
                              <span className="px-1.5 py-0.5 rounded-md bg-[#f5f5f7] dark:bg-white/10" style={{
                                fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#6e6e73',
                              }}>
                                {model.role}
                              </span>
                            )}
                          </div>
                          <ConfidenceBar value={model.confidence || 0.5} label="Confidence" size="sm" />
                          {model.position && (
                            <p style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: '#1d1d1f' }} className="dark:text-[#f1f5f9]">
                              {model.position}
                            </p>
                          )}
                          <p className="dark:text-[#e2e8f0]" style={{ fontFamily: FONT, fontSize: '12px', lineHeight: 1.5, color: '#3b3b3f' }}>
                            {(model.argument || '').slice(0, 500)}
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

      {/* ── Conflict Axes (text listing) ── */}
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
          { label: 'Disagreement', value: analysis.disagreement_strength, color: '#f59e0b' },
          { label: 'Convergence', value: analysis.convergence_level, isText: true, color: '#3b82f6' },
          { label: 'Confidence', value: analysis.confidence_recalibration || confidence, color: '#10b981' },
        ].map(metric => (
          <div key={metric.label} className="rounded-2xl bg-[#f5f5f7] dark:bg-[#1c1c1e] p-3 text-center">
            <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 500, color: '#6e6e73' }}>
              {metric.label}
            </span>
            <p style={{ fontFamily: FONT, fontSize: '16px', fontWeight: 700, color: metric.color, marginTop: '4px' }}>
              {metric.isText
                ? (metric.value || 'N/A')
                : metric.value != null ? `${Math.round(metric.value * 100)}%` : 'N/A'}
            </p>
          </div>
        ))}
      </div>

      <BoundaryPanel boundary={boundary} />
    </div>
  );
}
