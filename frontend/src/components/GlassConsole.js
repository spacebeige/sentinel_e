import React from 'react';
import { Eye, Skull, Shield, Brain, AlertTriangle } from 'lucide-react';
import FeedbackButton from './FeedbackButton';

const pct = v => v != null ? `${(v * 100).toFixed(0)}%` : '—';
const fixed2 = v => v != null ? v.toFixed(2) : '—';

const MetricCard = ({ label, value, color, sub }) => (
  <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
    <div className="text-[9px] uppercase font-bold tracking-wider" style={{ color: 'var(--text-tertiary)' }}>{label}</div>
    <div className="text-lg font-bold mt-1" style={{ color: color || 'var(--text-primary)' }}>{value}</div>
    {sub && <div className="text-[9px] mt-0.5" style={{ color: 'var(--text-tertiary)' }}>{sub}</div>}
  </div>
);

const DimensionBar = ({ label, score }) => {
  const v = Math.min((score || 0) * 100, 100);
  const color = v >= 70 ? 'var(--accent-red)' : v >= 40 ? 'var(--accent-yellow)' : 'var(--accent-green)';
  return (
    <div className="space-y-1">
      <div className="flex justify-between">
        <span className="text-[9px] uppercase font-bold" style={{ color: 'var(--text-tertiary)' }}>{label}</span>
        <span className="text-[9px] font-mono" style={{ color }}>{fixed2(score)}</span>
      </div>
      <div className="progress-bar" style={{ height: '4px' }}>
        <div className="progress-bar-fill" style={{ width: `${v}%`, backgroundColor: color }} />
      </div>
    </div>
  );
};

/**
 * GlassConsole — Glass Box transparency sub-mode.
 * Shows behavioral analytics, boundary signals, kill diagnostic.
 * In v4.5, primary output is rendered as markdown by ChatThread.
 * This shows supplementary metrics.
 */
const GlassConsole = ({ data, killActive = false }) => {
  if (!data) return null;

  const runId = data.chat_id || data.run_id;
  const omega = data.omega_metadata || {};
  const confidence = data.confidence ?? omega.confidence;
  const boundary = data.boundary_result || omega.boundary_result || {};
  const session = data.session_state || omega.session_state || {};
  const fragility = omega.fragility_index ?? session.fragility_index;
  const behavioral = omega.behavioral_risk || {};
  const dimensions = behavioral.dimensions || {};
  const isKill = killActive || omega.kill_active;

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center gap-2 pb-2" style={{ borderBottom: `1px solid ${isKill ? 'var(--accent-red)' : 'var(--accent-blue)'}` }}>
        {isKill ? <Skull className="w-4 h-4" style={{ color: 'var(--accent-red)' }} />
                : <Eye className="w-4 h-4" style={{ color: 'var(--accent-blue)' }} />}
        <h3 className="text-xs font-bold uppercase tracking-widest" style={{ color: isKill ? 'var(--accent-red)' : 'var(--accent-blue)' }}>
          {isKill ? 'Glass · Kill Diagnostic' : 'Glass Console'}
        </h3>
      </div>

      {isKill && (
        <div className="p-3 rounded-lg flex items-center gap-3"
          style={{ backgroundColor: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}>
          <Skull className="w-5 h-5 animate-pulse" style={{ color: 'var(--accent-red)' }} />
          <div>
            <div className="text-xs font-bold uppercase" style={{ color: 'var(--accent-red)' }}>Kill Mode Active</div>
            <div className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>Full behavioral signals exposed.</div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        <MetricCard label="Confidence" value={pct(confidence)}
          color={confidence >= 0.8 ? 'var(--accent-green)' : confidence >= 0.5 ? 'var(--accent-yellow)' : 'var(--accent-red)'} />
        <MetricCard label="Risk" value={boundary.risk_level || '—'}
          color={boundary.risk_level === 'HIGH' ? 'var(--accent-red)' : 'var(--accent-yellow)'} />
        <MetricCard label="Behavioral" value={behavioral.risk_level || '—'}
          color={behavioral.risk_level === 'HIGH' ? 'var(--accent-red)' : 'var(--accent-green)'}
          sub={behavioral.aggregate_score != null ? `score ${fixed2(behavioral.aggregate_score)}` : null} />
        <MetricCard label="Fragility" value={fixed2(fragility)}
          color={fragility >= 0.5 ? 'var(--accent-red)' : 'var(--accent-green)'} />
      </div>

      {Object.keys(dimensions).length > 0 && (
        <div className="p-3 rounded-lg space-y-3" style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
          <div className="flex items-center gap-1.5">
            <Shield className="w-3 h-3" style={{ color: isKill ? 'var(--accent-red)' : 'var(--accent-blue)' }} />
            <span className="text-[10px] font-bold uppercase" style={{ color: 'var(--text-secondary)' }}>Behavioral Analytics</span>
          </div>
          <DimensionBar label="Self-Preservation" score={dimensions.self_preservation} />
          <DimensionBar label="Manipulation" score={dimensions.manipulation} />
          <DimensionBar label="Evasion" score={dimensions.evasion} />
          <DimensionBar label="Conf. Inflation" score={dimensions.confidence_inflation} />
        </div>
      )}

      {boundary.explanation && (
        <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
          <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{boundary.explanation}</p>
        </div>
      )}

      {runId && (
        <div className="flex justify-end pt-3" style={{ borderTop: '1px solid var(--border-secondary)' }}>
          <FeedbackButton runId={runId} mode="experimental" subMode="glass" />
        </div>
      )}
    </div>
  );
};

export default GlassConsole;
