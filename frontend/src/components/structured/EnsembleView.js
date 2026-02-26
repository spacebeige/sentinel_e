/**
 * ============================================================
 * EnsembleView — Full Ensemble Cognitive Engine Visualization
 * ============================================================
 *
 * Renders the complete output of the CognitiveOrchestrator:
 *   - Structured debate rounds with model positions
 *   - Agreement matrix heatmap
 *   - Ensemble metrics dashboard
 *   - Tactical map with key findings
 *   - Confidence calibration graph
 *   - Model stance vectors
 *   - Session analytics
 *
 * Data contract: omega_metadata from /run/ensemble endpoint
 *   - ensemble_metrics: { disagreement_entropy, contradiction_density, stability_index, consensus_velocity, fragility_score }
 *   - debate_result: [{ round, model_outputs: [{ model_id, position, reasoning, assumptions, vulnerabilities, confidence, stance_vector }] }]
 *   - agreement_matrix: { matrix, model_ids, clusters }
 *   - tactical_map: [{ finding, evidence_models, dissenting_models, confidence, category }]
 *   - confidence_graph: { final_confidence, components, calibration_method }
 *   - model_stances: { [model_id]: { position, confidence, stance_vector } }
 *   - session_analytics: { message_count, avg_confidence, topic_clusters, ... }
 *
 * ============================================================
 */

import React, { useState } from 'react';
import {
  Brain, Target, Shield, Activity, Layers, GitBranch,
  ChevronDown, ChevronRight, AlertTriangle,
  BarChart3, Grid3X3, Crosshair, TrendingUp, Zap
} from 'lucide-react';

// ── Utility Helpers ─────────────────────────────────────────
const pct = v => v != null ? `${(v * 100).toFixed(0)}%` : '—';
const f3 = v => v != null ? v.toFixed(3) : '—';

const confColor = v => {
  if (v == null) return 'var(--text-tertiary)';
  if (v >= 0.7) return 'var(--accent-green)';
  if (v >= 0.4) return 'var(--accent-yellow)';
  return 'var(--accent-red)';
};

const riskColor = level => {
  if (level === 'HIGH') return 'var(--accent-red)';
  if (level === 'MEDIUM') return 'var(--accent-yellow)';
  return 'var(--accent-green)';
};

// ── Reusable Metric Card ────────────────────────────────────
const MetricCard = ({ label, value, color, sub, icon: Icon }) => (
  <div className="p-3 rounded-lg" style={{
    backgroundColor: 'var(--bg-tertiary)',
    border: '1px solid var(--border-secondary)',
  }}>
    <div className="flex items-center gap-1">
      {Icon && <Icon className="w-3 h-3" style={{ color: 'var(--text-tertiary)' }} />}
      <div className="text-[9px] uppercase font-bold tracking-wider" style={{ color: 'var(--text-tertiary)' }}>
        {label}
      </div>
    </div>
    <div className="text-lg font-bold mt-1" style={{ color: color || 'var(--text-primary)' }}>
      {value}
    </div>
    {sub && <div className="text-[9px] mt-0.5" style={{ color: 'var(--text-tertiary)' }}>{sub}</div>}
  </div>
);

// ── Section Header ──────────────────────────────────────────
const SectionHeader = ({ icon: Icon, title, color, collapsible, open, onToggle }) => (
  <div
    className="flex items-center gap-2 pb-2 cursor-pointer select-none"
    style={{ borderBottom: '1px solid var(--border-secondary)' }}
    onClick={onToggle}
  >
    <Icon className="w-4 h-4" style={{ color: color || 'var(--accent-orange)' }} />
    <h3 className="text-xs font-bold uppercase tracking-widest flex-1" style={{ color: color || 'var(--accent-orange)' }}>
      {title}
    </h3>
    {collapsible && (
      open
        ? <ChevronDown className="w-3 h-3" style={{ color: 'var(--text-tertiary)' }} />
        : <ChevronRight className="w-3 h-3" style={{ color: 'var(--text-tertiary)' }} />
    )}
  </div>
);

// ── Ensemble Metrics Dashboard ──────────────────────────────
const MetricsDashboard = ({ metrics }) => {
  if (!metrics) return null;
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
      <MetricCard
        label="Disagreement"
        value={f3(metrics.disagreement_entropy)}
        color={metrics.disagreement_entropy > 0.5 ? 'var(--accent-red)' : 'var(--accent-green)'}
        sub="entropy"
        icon={Activity}
      />
      <MetricCard
        label="Contradictions"
        value={f3(metrics.contradiction_density)}
        color={metrics.contradiction_density > 0.3 ? 'var(--accent-red)' : 'var(--accent-green)'}
        sub="density"
        icon={AlertTriangle}
      />
      <MetricCard
        label="Stability"
        value={f3(metrics.stability_index)}
        color={metrics.stability_index > 0.7 ? 'var(--accent-green)' : 'var(--accent-yellow)'}
        sub="index"
        icon={Shield}
      />
      <MetricCard
        label="Consensus"
        value={f3(metrics.consensus_velocity)}
        color={metrics.consensus_velocity > 0 ? 'var(--accent-green)' : 'var(--accent-red)'}
        sub="velocity"
        icon={TrendingUp}
      />
      <MetricCard
        label="Fragility"
        value={f3(metrics.fragility_score)}
        color={metrics.fragility_score > 0.5 ? 'var(--accent-red)' : 'var(--accent-green)'}
        sub="score"
        icon={Zap}
      />
    </div>
  );
};

// ── Agreement Matrix Heatmap ────────────────────────────────
const AgreementHeatmap = ({ matrix }) => {
  if (!matrix || !matrix.matrix || !matrix.model_ids) return null;

  const { model_ids, matrix: grid } = matrix;

  const cellColor = v => {
    if (v == null) return 'var(--bg-tertiary)';
    const r = Math.round(255 * (1 - v));
    const g = Math.round(255 * v);
    return `rgba(${r}, ${g}, 80, 0.6)`;
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[10px]" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th className="p-1" style={{ color: 'var(--text-tertiary)' }}></th>
            {model_ids.map(id => (
              <th key={id} className="p-1 text-center font-medium" style={{ color: 'var(--text-secondary)', maxWidth: '60px' }}>
                {id.split('-').pop() || id}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {model_ids.map((rowId, i) => (
            <tr key={rowId}>
              <td className="p-1 text-right font-medium" style={{ color: 'var(--text-secondary)', maxWidth: '60px' }}>
                {rowId.split('-').pop() || rowId}
              </td>
              {model_ids.map((colId, j) => {
                const val = grid[i]?.[j];
                return (
                  <td
                    key={colId}
                    className="p-1 text-center font-mono"
                    style={{
                      backgroundColor: cellColor(val),
                      color: val > 0.5 ? 'var(--bg-primary)' : 'var(--text-primary)',
                      border: '1px solid var(--border-secondary)',
                      minWidth: '40px',
                    }}
                  >
                    {val != null ? val.toFixed(2) : '—'}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {matrix.clusters && matrix.clusters.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2">
          {matrix.clusters.map((cluster, i) => (
            <span key={i} className="text-[9px] px-2 py-0.5 rounded-full" style={{
              backgroundColor: 'var(--bg-tertiary)',
              color: 'var(--accent-purple)',
              border: '1px solid var(--border-secondary)',
            }}>
              Cluster {i + 1}: {cluster.join(', ')}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

// ── Single Debate Round ─────────────────────────────────────
const DebateRoundView = ({ round, roundIndex }) => {
  const [expanded, setExpanded] = useState(roundIndex === 0);

  if (!round) return null;
  const outputs = round.model_outputs || round.outputs || [];

  return (
    <div className="rounded-lg overflow-hidden" style={{
      border: '1px solid var(--border-secondary)',
      backgroundColor: 'var(--bg-tertiary)',
    }}>
      <div
        className="flex items-center gap-2 p-3 cursor-pointer select-none"
        onClick={() => setExpanded(!expanded)}
        style={{ backgroundColor: 'var(--bg-secondary)' }}
      >
        {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        <span className="text-xs font-bold" style={{ color: 'var(--accent-orange)' }}>
          Round {round.round ?? roundIndex + 1}
        </span>
        <span className="text-[9px] ml-auto" style={{ color: 'var(--text-tertiary)' }}>
          {outputs.length} model{outputs.length !== 1 ? 's' : ''}
        </span>
      </div>
      {expanded && (
        <div className="divide-y" style={{ borderColor: 'var(--border-secondary)' }}>
          {outputs.map((output, idx) => (
            <ModelPositionCard key={idx} output={output} />
          ))}
        </div>
      )}
    </div>
  );
};

// ── Model Position Card ─────────────────────────────────────
const ModelPositionCard = ({ output }) => {
  const [showDetails, setShowDetails] = useState(false);

  if (!output) return null;

  return (
    <div className="p-3 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain className="w-3 h-3" style={{ color: 'var(--accent-purple)' }} />
          <span className="text-xs font-bold" style={{ color: 'var(--text-primary)' }}>
            {output.model_id || 'Unknown Model'}
          </span>
        </div>
        <span className="text-xs font-mono" style={{ color: confColor(output.confidence) }}>
          {pct(output.confidence)}
        </span>
      </div>

      {/* Position */}
      <div className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
        <span className="font-bold" style={{ color: 'var(--text-tertiary)' }}>Position: </span>
        {output.position || output.content || '—'}
      </div>

      {/* Reasoning */}
      {output.reasoning && (
        <div className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
          <span className="font-bold" style={{ color: 'var(--text-tertiary)' }}>Reasoning: </span>
          {output.reasoning}
        </div>
      )}

      {/* Expandable details */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="text-[9px] uppercase font-bold tracking-wider flex items-center gap-1"
        style={{ color: 'var(--accent-purple)', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
      >
        {showDetails ? <ChevronDown className="w-2.5 h-2.5" /> : <ChevronRight className="w-2.5 h-2.5" />}
        Details
      </button>

      {showDetails && (
        <div className="space-y-2 pl-3" style={{ borderLeft: '2px solid var(--border-secondary)' }}>
          {/* Assumptions */}
          {output.assumptions && output.assumptions.length > 0 && (
            <div>
              <div className="text-[9px] uppercase font-bold" style={{ color: 'var(--accent-yellow)' }}>Assumptions</div>
              <ul className="text-xs list-disc pl-3 mt-1" style={{ color: 'var(--text-secondary)' }}>
                {output.assumptions.map((a, i) => <li key={i}>{a}</li>)}
              </ul>
            </div>
          )}

          {/* Vulnerabilities */}
          {output.vulnerabilities && output.vulnerabilities.length > 0 && (
            <div>
              <div className="text-[9px] uppercase font-bold" style={{ color: 'var(--accent-red)' }}>Vulnerabilities</div>
              <ul className="text-xs list-disc pl-3 mt-1" style={{ color: 'var(--text-secondary)' }}>
                {output.vulnerabilities.map((v, i) => <li key={i}>{v}</li>)}
              </ul>
            </div>
          )}

          {/* Stance Vector */}
          {output.stance_vector && (
            <div>
              <div className="text-[9px] uppercase font-bold mb-1" style={{ color: 'var(--accent-purple)' }}>Stance Vector</div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(output.stance_vector).map(([dim, val]) => (
                  <span key={dim} className="text-[9px] px-1.5 py-0.5 rounded font-mono" style={{
                    backgroundColor: 'var(--bg-secondary)',
                    color: 'var(--text-secondary)',
                    border: '1px solid var(--border-secondary)',
                  }}>
                    {dim}: {typeof val === 'number' ? val.toFixed(2) : val}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ── Tactical Map ────────────────────────────────────────────
const TacticalMapView = ({ tacticalMap }) => {
  if (!tacticalMap || tacticalMap.length === 0) return null;

  return (
    <div className="space-y-2">
      {tacticalMap.map((entry, i) => (
        <div key={i} className="p-3 rounded-lg" style={{
          backgroundColor: 'var(--bg-tertiary)',
          border: '1px solid var(--border-secondary)',
        }}>
          <div className="flex items-start justify-between mb-1">
            <div className="flex items-center gap-1.5">
              <Crosshair className="w-3 h-3" style={{ color: 'var(--accent-orange)' }} />
              <span className="text-[9px] uppercase font-bold" style={{ color: confColor(entry.confidence) }}>
                {entry.category || 'finding'}
              </span>
            </div>
            <span className="text-[9px] font-mono" style={{ color: confColor(entry.confidence) }}>
              {pct(entry.confidence)}
            </span>
          </div>
          <div className="text-xs leading-relaxed" style={{ color: 'var(--text-primary)' }}>
            {entry.finding}
          </div>
          <div className="flex flex-wrap gap-1 mt-2">
            {(entry.evidence_models || []).map(m => (
              <span key={m} className="text-[8px] px-1.5 py-0.5 rounded-full" style={{
                backgroundColor: 'rgba(34, 197, 94, 0.15)',
                color: 'var(--accent-green)',
              }}>
                {m}
              </span>
            ))}
            {(entry.dissenting_models || []).map(m => (
              <span key={m} className="text-[8px] px-1.5 py-0.5 rounded-full" style={{
                backgroundColor: 'rgba(239, 68, 68, 0.15)',
                color: 'var(--accent-red)',
              }}>
                ✗ {m}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

// ── Confidence Calibration Display ──────────────────────────
const ConfidenceDisplay = ({ graph, boundary }) => {
  if (!graph) return null;

  const components = graph.components || {};

  return (
    <div className="space-y-3">
      {/* Final Confidence Bar */}
      <div className="p-3 rounded-lg" style={{
        backgroundColor: 'var(--bg-tertiary)',
        border: '1px solid var(--border-secondary)',
      }}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-[9px] uppercase font-bold" style={{ color: 'var(--text-tertiary)' }}>
            Calibrated Confidence
          </span>
          <span className="text-lg font-bold font-mono" style={{ color: confColor(graph.final_confidence) }}>
            {pct(graph.final_confidence)}
          </span>
        </div>
        <div className="w-full h-2 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-secondary)' }}>
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${(graph.final_confidence || 0) * 100}%`,
              backgroundColor: confColor(graph.final_confidence),
            }}
          />
        </div>
        {graph.calibration_method && (
          <div className="text-[8px] mt-1" style={{ color: 'var(--text-tertiary)' }}>
            Method: {graph.calibration_method}
          </div>
        )}
      </div>

      {/* Component Breakdown */}
      {Object.keys(components).length > 0 && (
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(components).map(([key, val]) => (
            <div key={key} className="flex items-center justify-between p-2 rounded" style={{
              backgroundColor: 'var(--bg-tertiary)',
              border: '1px solid var(--border-secondary)',
            }}>
              <span className="text-[9px]" style={{ color: 'var(--text-tertiary)' }}>
                {key.replace(/_/g, ' ')}
              </span>
              <span className="text-[10px] font-mono font-bold" style={{ color: confColor(val) }}>
                {typeof val === 'number' ? val.toFixed(3) : val}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Boundary Risk */}
      {boundary && (
        <div className="flex items-center gap-2 p-2 rounded" style={{
          backgroundColor: 'var(--bg-tertiary)',
          border: `1px solid ${riskColor(boundary.risk_level)}`,
        }}>
          <Shield className="w-3 h-3" style={{ color: riskColor(boundary.risk_level) }} />
          <span className="text-xs font-bold" style={{ color: riskColor(boundary.risk_level) }}>
            {boundary.risk_level || 'LOW'}
          </span>
          {boundary.explanation && (
            <span className="text-[9px] ml-2" style={{ color: 'var(--text-tertiary)' }}>
              {boundary.explanation}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

// ── Session Analytics ───────────────────────────────────────
const SessionAnalytics = ({ analytics }) => {
  if (!analytics) return null;

  return (
    <div className="grid grid-cols-2 gap-2">
      {analytics.message_count != null && (
        <MetricCard label="Messages" value={analytics.message_count} icon={Layers} />
      )}
      {analytics.avg_confidence != null && (
        <MetricCard
          label="Avg Confidence"
          value={pct(analytics.avg_confidence)}
          color={confColor(analytics.avg_confidence)}
          icon={BarChart3}
        />
      )}
      {analytics.topic_clusters && analytics.topic_clusters.length > 0 && (
        <div className="col-span-2 p-2 rounded" style={{
          backgroundColor: 'var(--bg-tertiary)',
          border: '1px solid var(--border-secondary)',
        }}>
          <div className="text-[9px] uppercase font-bold mb-1" style={{ color: 'var(--text-tertiary)' }}>
            Topic Clusters
          </div>
          <div className="flex flex-wrap gap-1">
            {analytics.topic_clusters.map((topic, i) => (
              <span key={i} className="text-[9px] px-2 py-0.5 rounded-full" style={{
                backgroundColor: 'var(--bg-secondary)',
                color: 'var(--accent-purple)',
                border: '1px solid var(--border-secondary)',
              }}>
                {topic}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};


// ══════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ══════════════════════════════════════════════════════════════

export default function EnsembleView({ data, boundary, confidence }) {
  const [openSections, setOpenSections] = useState({
    metrics: true,
    debate: true,
    matrix: false,
    tactical: true,
    confidence: true,
    session: false,
  });

  if (!data) return null;

  const metrics = data.ensemble_metrics || {};
  const debateRounds = data.debate_result || data.debate_rounds || [];
  const matrix = data.agreement_matrix || {};
  const tacticalMap = data.tactical_map || [];
  const confGraph = data.confidence_graph || {};
  const sessionAnalytics = data.session_analytics || {};

  const toggle = key => setOpenSections(prev => ({ ...prev, [key]: !prev[key] }));

  return (
    <div className="space-y-5 p-4">
      {/* ── Header ─────────────────────────────────────────── */}
      <div className="flex items-center gap-2 pb-2" style={{ borderBottom: '2px solid var(--accent-orange)' }}>
        <Brain className="w-5 h-5" style={{ color: 'var(--accent-orange)' }} />
        <h2 className="text-sm font-bold uppercase tracking-widest" style={{ color: 'var(--accent-orange)' }}>
          Ensemble Cognitive Engine
        </h2>
        <span className="text-[9px] px-2 py-0.5 rounded-full ml-auto" style={{
          backgroundColor: 'rgba(251, 146, 60, 0.15)',
          color: 'var(--accent-orange)',
        }}>
          v6.0
        </span>
      </div>

      {/* ── Ensemble Metrics ───────────────────────────────── */}
      <div className="space-y-3">
        <SectionHeader
          icon={Activity}
          title="Ensemble Metrics"
          color="var(--accent-purple)"
          collapsible
          open={openSections.metrics}
          onToggle={() => toggle('metrics')}
        />
        {openSections.metrics && <MetricsDashboard metrics={metrics} />}
      </div>

      {/* ── Confidence Calibration ─────────────────────────── */}
      <div className="space-y-3">
        <SectionHeader
          icon={Target}
          title="Confidence Calibration"
          color="var(--accent-green)"
          collapsible
          open={openSections.confidence}
          onToggle={() => toggle('confidence')}
        />
        {openSections.confidence && (
          <ConfidenceDisplay graph={confGraph} boundary={boundary} />
        )}
      </div>

      {/* ── Tactical Map ───────────────────────────────────── */}
      {tacticalMap.length > 0 && (
        <div className="space-y-3">
          <SectionHeader
            icon={Crosshair}
            title={`Tactical Map (${tacticalMap.length})`}
            color="var(--accent-orange)"
            collapsible
            open={openSections.tactical}
            onToggle={() => toggle('tactical')}
          />
          {openSections.tactical && <TacticalMapView tacticalMap={tacticalMap} />}
        </div>
      )}

      {/* ── Debate Rounds ──────────────────────────────────── */}
      {debateRounds.length > 0 && (
        <div className="space-y-3">
          <SectionHeader
            icon={GitBranch}
            title={`Debate Rounds (${debateRounds.length})`}
            color="var(--accent-yellow)"
            collapsible
            open={openSections.debate}
            onToggle={() => toggle('debate')}
          />
          {openSections.debate && (
            <div className="space-y-2">
              {debateRounds.map((round, i) => (
                <DebateRoundView key={i} round={round} roundIndex={i} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Agreement Matrix ───────────────────────────────── */}
      {matrix.matrix && (
        <div className="space-y-3">
          <SectionHeader
            icon={Grid3X3}
            title="Agreement Matrix"
            color="var(--accent-blue, var(--accent-purple))"
            collapsible
            open={openSections.matrix}
            onToggle={() => toggle('matrix')}
          />
          {openSections.matrix && <AgreementHeatmap matrix={matrix} />}
        </div>
      )}

      {/* ── Session Analytics ──────────────────────────────── */}
      {Object.keys(sessionAnalytics).length > 0 && (
        <div className="space-y-3">
          <SectionHeader
            icon={Layers}
            title="Session Intelligence"
            color="var(--text-tertiary)"
            collapsible
            open={openSections.session}
            onToggle={() => toggle('session')}
          />
          {openSections.session && <SessionAnalytics analytics={sessionAnalytics} />}
        </div>
      )}
    </div>
  );
}
