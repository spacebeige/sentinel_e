import React from 'react';
import { Swords, TrendingUp, AlertTriangle, Shield, Brain, Activity } from 'lucide-react';
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

/**
 * DebateArena — Standalone debate metrics panel (legacy compatibility).
 * In v4.5, debate output is primarily rendered as markdown by ChatThread.
 * This component shows supplementary metrics when used directly.
 */
const DebateArena = ({ data }) => {
  if (!data) return null;

  const runId = data.chat_id || data.run_id;
  const omega = data.omega_metadata || {};
  const confidence = data.confidence ?? omega.confidence;
  const reasoning = data.reasoning_trace || omega.reasoning_trace || {};
  const boundary = data.boundary_result || omega.boundary_result || {};
  const session = data.session_state || omega.session_state || {};
  const fragility = omega.fragility_index ?? session.fragility_index;

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center gap-2 pb-2" style={{ borderBottom: '1px solid var(--border-secondary)' }}>
        <Swords className="w-4 h-4" style={{ color: 'var(--accent-orange)' }} />
        <h3 className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--accent-orange)' }}>
          Debate Arena
        </h3>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <MetricCard label="Confidence" value={pct(confidence)}
          color={confidence >= 0.8 ? 'var(--accent-green)' : confidence >= 0.5 ? 'var(--accent-yellow)' : 'var(--accent-red)'} />
        <MetricCard label="Risk" value={boundary.risk_level || 'LOW'}
          color={boundary.risk_level === 'HIGH' ? 'var(--accent-red)' : boundary.risk_level === 'MEDIUM' ? 'var(--accent-yellow)' : 'var(--accent-green)'} />
        <MetricCard label="Passes" value={reasoning.total_passes || '—'}
          sub={reasoning.logical_gaps_detected ? `${reasoning.logical_gaps_detected} gaps` : 'clean'} />
        <MetricCard label="Fragility" value={fixed2(fragility)}
          color={fragility >= 0.5 ? 'var(--accent-red)' : 'var(--accent-green)'} />
      </div>

      {session.inferred_domain && (
        <div className="p-3 rounded-lg space-y-2" style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
          <div className="flex items-center gap-1.5">
            <Brain className="w-3 h-3" style={{ color: 'var(--accent-purple)' }} />
            <span className="text-[10px] font-bold uppercase" style={{ color: 'var(--text-secondary)' }}>Session</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div><span style={{ color: 'var(--text-tertiary)' }}>Domain:</span> <span style={{ color: 'var(--text-primary)' }}>{session.inferred_domain}</span></div>
            <div><span style={{ color: 'var(--text-tertiary)' }}>Expertise:</span> <span style={{ color: 'var(--text-primary)' }}>{fixed2(session.user_expertise_score)}</span></div>
          </div>
        </div>
      )}

      {boundary.explanation && (
        <div className="p-3 rounded-lg" style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
          <div className="flex items-center gap-1.5 mb-2">
            <Shield className="w-3 h-3" style={{ color: 'var(--accent-yellow)' }} />
            <span className="text-[10px] font-bold uppercase" style={{ color: 'var(--text-secondary)' }}>Boundary</span>
          </div>
          <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{boundary.explanation}</p>
        </div>
      )}

      {runId && (
        <div className="flex justify-end pt-3" style={{ borderTop: '1px solid var(--border-secondary)' }}>
          <FeedbackButton runId={runId} mode="experimental" subMode="debate" />
        </div>
      )}
    </div>
  );
};

export default DebateArena;
