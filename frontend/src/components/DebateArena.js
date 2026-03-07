/**
 * ============================================================
 * DebateArena — Battle Platform v2 Three-Panel Debate UI
 * ============================================================
 *
 * Layout:
 *   ┌──────────────────────────────────────────────────────────┐
 *   │  PROMPT HEADER (prompt text + stability score + winner)  │
 *   ├────────────────┬──────────────────┬─────────────────────┤
 *   │  LEFT PANEL    │  CENTER PANEL    │  RIGHT PANEL        │
 *   │  Debate        │  Model Columns   │  Metrics + Charts   │
 *   │  Timeline      │  (dynamic count) │  + Consensus Scores │
 *   └────────────────┴──────────────────┴─────────────────────┘
 *
 * Props:
 *   data  {Object}  — BattleVisualizationPayload from POST /battle/debate
 *                     Shape: { prompt, models, round_outputs, reasoning_metrics,
 *                              consensus_scores, consensus_stability_score,
 *                              debate_timeline, winner, winner_score,
 *                              agreement_heatmap, model_labels, conflict_edges,
 *                              charts? }
 *
 * The component renders model columns dynamically from payload.models —
 * no model IDs are hardcoded.  Supports 3–6 models.
 * ============================================================
 */

import React, { useState, useMemo } from 'react';
import {
  Swords, Trophy, Activity, BarChart2,
  ChevronDown, ChevronRight, Layers,
} from 'lucide-react';

// ── Colour palette ────────────────────────────────────────────
const TIER_COLORS = {
  1: '#f59e0b',   // amber  — anchor
  2: '#3b82f6',   // blue   — debate
  3: '#10b981',   // emerald — specialist
};

const PROVIDER_COLORS = {
  groq: '#6366f1',
  openrouter: '#06b6d4',
};

const STATUS_COLORS = {
  HIGH:    '#ef4444',
  MEDIUM:  '#f59e0b',
  LOW:     '#10b981',
};

// ── Small helpers ─────────────────────────────────────────────
const pct   = (v) => v != null ? `${(v * 100).toFixed(0)}%`  : '—';
const fixed2 = (v) => v != null ? Number(v).toFixed(2)         : '—';
const clamp  = (v) => Math.max(0, Math.min(1, v ?? 0));

// Derive a per-model colour from tier, provider, or position index
function modelColor(model, idx) {
  if (model.tier && TIER_COLORS[model.tier])  return TIER_COLORS[model.tier];
  if (model.provider && PROVIDER_COLORS[model.provider]) return PROVIDER_COLORS[model.provider];
  const palette = ['#6366f1', '#f59e0b', '#10b981', '#ef4444', '#06b6d4', '#8b5cf6'];
  return palette[idx % palette.length];
}

// ── Sub-components ────────────────────────────────────────────

/** Compact metric pill */
const MetricPill = ({ label, value, color }) => (
  <div className="flex flex-col items-center px-2 py-1.5 rounded"
       style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)', minWidth: 60 }}>
    <span className="text-[9px] uppercase tracking-wider font-semibold"
          style={{ color: 'var(--text-tertiary)' }}>{label}</span>
    <span className="text-sm font-bold mt-0.5"
          style={{ color: color || 'var(--text-primary)' }}>{value}</span>
  </div>
);

/** Horizontal progress bar */
const ProgressBar = ({ value, color }) => (
  <div className="h-1.5 rounded-full overflow-hidden"
       style={{ backgroundColor: 'var(--bg-tertiary)' }}>
    <div className="h-full rounded-full transition-all duration-500"
         style={{ width: `${clamp(value) * 100}%`, backgroundColor: color || 'var(--accent-blue)' }} />
  </div>
);

/**
 * Single model column — renders one model's outputs for each debate round.
 */
const ModelColumn = ({ model, roundOutputs, roundKeys, metrics, consensusScore, color }) => {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex flex-col rounded-lg overflow-hidden flex-1 min-w-0"
         style={{ border: `1px solid ${color}44`, backgroundColor: 'var(--bg-secondary)' }}>
      {/* Column header */}
      <button
        onClick={() => setCollapsed(c => !c)}
        className="flex items-center justify-between px-3 py-2 text-left"
        style={{ backgroundColor: `${color}22`, borderBottom: `1px solid ${color}44` }}
      >
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
          <span className="text-xs font-bold truncate" style={{ color: 'var(--text-primary)' }}>
            {model.name}
          </span>
          {model.tier && (
            <span className="text-[8px] px-1.5 py-0.5 rounded font-semibold flex-shrink-0"
                  style={{ backgroundColor: `${color}33`, color }}>
              T{model.tier}
            </span>
          )}
        </div>
        {collapsed ? <ChevronRight className="w-3 h-3 flex-shrink-0" style={{ color: 'var(--text-tertiary)' }} />
                   : <ChevronDown  className="w-3 h-3 flex-shrink-0" style={{ color: 'var(--text-tertiary)' }} />}
      </button>

      {!collapsed && (
        <div className="flex flex-col gap-2 p-2 flex-1">
          {/* Debate rounds */}
          {roundKeys.map((rk) => {
            const roundEntries = roundOutputs?.[rk] || [];
            const entry = roundEntries.find(
              (e) => e.model === model.id || e.model_id === model.id
            );
            return (
              <div key={rk}
                   className="p-2 rounded"
                   style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
                <div className="text-[8px] uppercase font-semibold mb-1"
                     style={{ color: 'var(--text-tertiary)' }}>
                  Round {rk}
                  {entry?.tokens_used ? (
                    <span className="ml-2 normal-case font-normal">{entry.tokens_used} tok</span>
                  ) : null}
                </div>
                <p className="text-[11px] leading-relaxed"
                   style={{ color: entry ? 'var(--text-secondary)' : 'var(--text-tertiary)' }}>
                  {entry?.output || <em>— skipped —</em>}
                </p>
              </div>
            );
          })}

          {/* Per-model metrics */}
          {metrics && (
            <div className="mt-1 space-y-1">
              <div className="text-[9px] uppercase font-semibold"
                   style={{ color: 'var(--text-tertiary)' }}>Metrics</div>
              {[
                { key: 'reasoning_score',    label: 'Reasoning' },
                { key: 'evidence_density',   label: 'Evidence'  },
                { key: 'logical_consistency',label: 'Logic'     },
                { key: 'argument_depth',     label: 'Depth'     },
              ].map(({ key, label }) => (
                <div key={key}>
                  <div className="flex justify-between text-[9px] mb-0.5">
                    <span style={{ color: 'var(--text-tertiary)' }}>{label}</span>
                    <span style={{ color: 'var(--text-secondary)' }}>{pct(metrics[key])}</span>
                  </div>
                  <ProgressBar value={metrics[key]} color={color} />
                </div>
              ))}
            </div>
          )}

          {/* Consensus score */}
          {consensusScore != null && (
            <div className="flex items-center justify-between text-xs mt-1 px-1">
              <span style={{ color: 'var(--text-tertiary)' }}>Consensus</span>
              <span className="font-bold"
                    style={{ color: consensusScore >= 0.7 ? '#10b981' : consensusScore >= 0.5 ? '#f59e0b' : '#ef4444' }}>
                {pct(consensusScore)}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

/**
 * Left panel — Debate Timeline showing round-by-round progression.
 */
const TimelinePanel = ({ timeline, stability }) => {
  if (!timeline || timeline.length === 0) return null;

  return (
    <div className="rounded-lg p-3 space-y-3"
         style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-secondary)' }}>
      <div className="flex items-center gap-2">
        <Activity className="w-3.5 h-3.5" style={{ color: 'var(--accent-blue)' }} />
        <span className="text-[10px] font-bold uppercase tracking-wider"
              style={{ color: 'var(--text-secondary)' }}>Debate Timeline</span>
        {stability != null && (
          <span className="ml-auto text-[10px] font-bold"
                style={{ color: stability >= 0.7 ? '#10b981' : stability >= 0.5 ? '#f59e0b' : '#ef4444' }}>
            Stability {pct(stability)}
          </span>
        )}
      </div>

      {timeline.map((entry, i) => (
        <div key={i} className="flex gap-2">
          <div className="flex flex-col items-center">
            <div className="w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold flex-shrink-0"
                 style={{ backgroundColor: 'var(--accent-blue)', color: '#fff' }}>
              {entry.round || i + 1}
            </div>
            {i < timeline.length - 1 && (
              <div className="w-px flex-1 mt-1"
                   style={{ backgroundColor: 'var(--border-secondary)' }} />
            )}
          </div>
          <div className="flex-1 pb-3">
            <div className="text-[10px] font-medium mb-1"
                 style={{ color: 'var(--text-secondary)' }}>
              {entry.label || `Round ${entry.round || i + 1}`}
            </div>
            {entry.avg_confidence != null && (
              <div>
                <div className="flex justify-between text-[9px] mb-0.5">
                  <span style={{ color: 'var(--text-tertiary)' }}>Avg Confidence</span>
                  <span style={{ color: 'var(--text-primary)' }}>{pct(entry.avg_confidence)}</span>
                </div>
                <ProgressBar value={entry.avg_confidence} color="var(--accent-blue)" />
              </div>
            )}
            {entry.consensus != null && (
              <div className="mt-1">
                <div className="flex justify-between text-[9px] mb-0.5">
                  <span style={{ color: 'var(--text-tertiary)' }}>Consensus</span>
                  <span style={{ color: 'var(--text-primary)' }}>{pct(entry.consensus)}</span>
                </div>
                <ProgressBar value={entry.consensus} color="#10b981" />
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

/**
 * Right panel — Reasoning metrics table + consensus scores + optional charts.
 */
const MetricsPanel = ({ reasoningMetrics, consensusScores, charts }) => (
  <div className="space-y-3">
    {/* Consensus Scores ranking */}
    {consensusScores && consensusScores.length > 0 && (
      <div className="rounded-lg p-3 space-y-2"
           style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-secondary)' }}>
        <div className="flex items-center gap-2">
          <Trophy className="w-3.5 h-3.5" style={{ color: '#f59e0b' }} />
          <span className="text-[10px] font-bold uppercase tracking-wider"
                style={{ color: 'var(--text-secondary)' }}>Rankings</span>
        </div>
        {consensusScores.map((s, i) => (
          <div key={s.model} className="flex items-center gap-2">
            <span className="w-4 text-[10px] font-bold text-right flex-shrink-0"
                  style={{ color: i === 0 ? '#f59e0b' : 'var(--text-tertiary)' }}>
              #{s.rank ?? i + 1}
            </span>
            <span className="flex-1 text-[11px] truncate"
                  style={{ color: 'var(--text-primary)' }}>
              {s.model_name || s.model}
            </span>
            <span className="text-[11px] font-bold flex-shrink-0"
                  style={{ color: s.composite_score >= 0.7 ? '#10b981' : s.composite_score >= 0.5 ? '#f59e0b' : '#ef4444' }}>
              {fixed2(s.composite_score)}
            </span>
          </div>
        ))}
      </div>
    )}

    {/* Reasoning Metrics table */}
    {reasoningMetrics && reasoningMetrics.length > 0 && (
      <div className="rounded-lg p-3 space-y-2"
           style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-secondary)' }}>
        <div className="flex items-center gap-2">
          <BarChart2 className="w-3.5 h-3.5" style={{ color: 'var(--accent-purple)' }} />
          <span className="text-[10px] font-bold uppercase tracking-wider"
                style={{ color: 'var(--text-secondary)' }}>Reasoning Metrics</span>
        </div>
        {reasoningMetrics.map((m) => (
          <div key={m.model} className="space-y-1">
            <div className="text-[10px] font-semibold" style={{ color: 'var(--text-secondary)' }}>
              {m.model_name || m.model}
            </div>
            <div className="grid grid-cols-2 gap-1">
              {[
                { k: 'reasoning_score',    l: 'Reasoning'   },
                { k: 'evidence_density',   l: 'Evidence'    },
                { k: 'logical_consistency',l: 'Logic'       },
                { k: 'argument_depth',     l: 'Depth'       },
                { k: 'token_efficiency',   l: 'Efficiency'  },
              ].map(({ k, l }) => (
                <div key={k} className="text-[9px]" style={{ color: 'var(--text-tertiary)' }}>
                  {l}: <span style={{ color: 'var(--text-primary)' }}>{pct(m[k])}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    )}

    {/* Base64 chart images (optional) */}
    {charts && Object.entries(charts).map(([key, b64]) =>
      b64 ? (
        <div key={key} className="rounded-lg overflow-hidden"
             style={{ border: '1px solid var(--border-secondary)' }}>
          <div className="text-[9px] uppercase font-semibold px-2 py-1"
               style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-tertiary)' }}>
            {key.replace(/_/g, ' ')}
          </div>
          <img src={`data:image/png;base64,${b64}`} alt={key}
               className="w-full" style={{ display: 'block' }} />
        </div>
      ) : null
    )}
  </div>
);

// ── Main component ────────────────────────────────────────────

/**
 * DebateArena
 *
 * Accepts either:
 *   1. The new BattleVisualizationPayload (from POST /battle/debate)
 *      → Three-panel layout with dynamic model columns
 *   2. Legacy omega_metadata payload (backward-compat)
 *      → Simple metrics card (unchanged behaviour)
 */
const DebateArena = ({ data }) => {
  if (!data) return null;

  // ── Detect payload type ────────────────────────────────────
  const isBattlePayload = !!(data.models_selected || data.models || data.round_outputs);

  // ── Legacy fallback ────────────────────────────────────────
  if (!isBattlePayload) {
    const omega = data.omega_metadata || {};
    const confidence = data.confidence ?? omega.confidence;
    const boundary  = data.boundary_result || omega.boundary_result || {};
    const session   = data.session_state   || omega.session_state   || {};
    const reasoning = data.reasoning_trace || omega.reasoning_trace || {};
    const fragility = omega.fragility_index ?? session.fragility_index;

    return (
      <div className="space-y-4 p-4">
        <div className="flex items-center gap-2 pb-2"
             style={{ borderBottom: '1px solid var(--border-secondary)' }}>
          <Swords className="w-4 h-4" style={{ color: 'var(--accent-orange)' }} />
          <h3 className="text-xs font-bold uppercase tracking-widest"
              style={{ color: 'var(--accent-orange)' }}>Debate Arena</h3>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <MetricPill label="Confidence" value={pct(confidence)}
            color={confidence >= 0.8 ? '#10b981' : confidence >= 0.5 ? '#f59e0b' : '#ef4444'} />
          <MetricPill label="Risk"       value={boundary.risk_level || 'LOW'}
            color={STATUS_COLORS[boundary.risk_level] || '#10b981'} />
          <MetricPill label="Passes"     value={reasoning.total_passes || '—'}
            color="var(--text-primary)" />
          <MetricPill label="Fragility"  value={fixed2(fragility)}
            color={fragility >= 0.5 ? '#ef4444' : '#10b981'} />
        </div>
      </div>
    );
  }

  // ── New BattleVisualizationPayload ─────────────────────────
  const {
    prompt,
    prompt_type,
    models,                     // [{id, name, provider, role, tier}, ...]
    models_selected,            // fallback when models metadata not present
    round_outputs,              // {1: [...], 2: [...], 3: [...]}
    reasoning_metrics,
    consensus_scores,
    consensus_stability_score,
    debate_timeline,
    winner,
    winner_score,
    charts,                     // optional base64 chart images
  } = data;

  // Build models list: prefer enriched models array; fall back to selected keys
  const modelList = useMemo(() => {
    if (models && models.length > 0) return models;
    if (models_selected && models_selected.length > 0) {
      return models_selected.map((id) => ({ id, name: id }));
    }
    return [];
  }, [models, models_selected]);

  const roundKeys = useMemo(
    () => Object.keys(round_outputs || {}).sort(),
    [round_outputs]
  );

  // Build lookup maps
  const metricsMap = useMemo(
    () => Object.fromEntries((reasoning_metrics || []).map((m) => [m.model, m])),
    [reasoning_metrics]
  );
  const consensusMap = useMemo(
    () => Object.fromEntries((consensus_scores || []).map((s) => [s.model, s.composite_score])),
    [consensus_scores]
  );

  return (
    <div className="flex flex-col gap-3 p-3">
      {/* ── Header ─────────────────────────────────────────── */}
      <div className="rounded-lg px-4 py-3 flex flex-wrap items-start gap-3"
           style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-secondary)' }}>
        <div className="flex items-center gap-2 flex-shrink-0">
          <Swords className="w-4 h-4" style={{ color: 'var(--accent-orange)' }} />
          <span className="text-xs font-bold uppercase tracking-widest"
                style={{ color: 'var(--accent-orange)' }}>Debate Arena</span>
        </div>

        {prompt && (
          <p className="flex-1 text-xs leading-snug"
             style={{ color: 'var(--text-secondary)', minWidth: 120 }}>
            {prompt}
          </p>
        )}

        <div className="flex items-center gap-3 flex-shrink-0 ml-auto">
          {consensus_stability_score != null && (
            <MetricPill
              label="Stability"
              value={pct(consensus_stability_score)}
              color={consensus_stability_score >= 0.7 ? '#10b981' : consensus_stability_score >= 0.5 ? '#f59e0b' : '#ef4444'}
            />
          )}
          {winner && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded"
                 style={{ backgroundColor: '#f59e0b22', border: '1px solid #f59e0b44' }}>
              <Trophy className="w-3 h-3" style={{ color: '#f59e0b' }} />
              <span className="text-[10px] font-bold" style={{ color: '#f59e0b' }}>{winner}</span>
              {winner_score != null && (
                <span className="text-[9px]" style={{ color: '#f59e0b99' }}>
                  {fixed2(winner_score)}
                </span>
              )}
            </div>
          )}
          {prompt_type && (
            <span className="text-[9px] px-2 py-1 rounded font-semibold uppercase"
                  style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-tertiary)' }}>
              {prompt_type}
            </span>
          )}
          <div className="flex items-center gap-1.5 text-[9px]"
               style={{ color: 'var(--text-tertiary)' }}>
            <Layers className="w-3 h-3" />
            {modelList.length} models
          </div>
        </div>
      </div>

      {/* ── Three-panel body ───────────────────────────────── */}
      <div className="flex gap-3 items-start"
           style={{ minHeight: 300 }}>

        {/* LEFT: Timeline */}
        <div style={{ width: 200, flexShrink: 0 }}>
          <TimelinePanel
            timeline={debate_timeline}
            stability={consensus_stability_score}
          />
        </div>

        {/* CENTER: Dynamic model columns */}
        <div className="flex gap-2 flex-1 min-w-0 overflow-x-auto">
          {modelList.map((model, idx) => {
            const color = modelColor(model, idx);
            return (
              <ModelColumn
                key={model.id}
                model={model}
                roundOutputs={round_outputs}
                roundKeys={roundKeys}
                metrics={metricsMap[model.id]}
                consensusScore={consensusMap[model.id]}
                color={color}
              />
            );
          })}
          {modelList.length === 0 && (
            <div className="flex-1 flex items-center justify-center text-xs"
                 style={{ color: 'var(--text-tertiary)' }}>
              No model data available.
            </div>
          )}
        </div>

        {/* RIGHT: Metrics + Charts */}
        <div style={{ width: 220, flexShrink: 0 }}>
          <MetricsPanel
            reasoningMetrics={reasoning_metrics}
            consensusScores={consensus_scores}
            charts={charts}
          />
        </div>
      </div>
    </div>
  );
};

export default DebateArena;

