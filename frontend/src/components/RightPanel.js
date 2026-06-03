import React from 'react';
import { X, Activity, Shield, AlertTriangle, Brain, TrendingUp } from 'lucide-react';
import { RadialBarChart, RadialBar, LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';
import CrossModelAnalytics from './CrossModelAnalytics';

/* ─── Helpers ───────────────────────────────────────────────────── */
const pct = v => v != null ? `${(v * 100).toFixed(0)}%` : '—';
const fixed2 = v => v != null ? Number(v).toFixed(2) : '—';

/* ─── Confidence Gauge (radial) ─────────────────────────────────── */
const ConfidenceGauge = ({ value }) => {
  const v = (value || 0) * 100;
  const color = v >= 80 ? 'var(--accent-green)' : v >= 50 ? 'var(--accent-yellow)' : 'var(--accent-red)';
  const data = [{ name: 'confidence', value: v, fill: color }];
  return (
    <div className="gauge-container">
      <div className="gauge-label">Confidence</div>
      <ResponsiveContainer width="100%" height={120}>
        <RadialBarChart cx="50%" cy="50%" innerRadius="60%" outerRadius="90%"
          barSize={10} data={data} startAngle={210} endAngle={-30}>
          <RadialBar background={{ fill: 'var(--bg-tertiary)' }} dataKey="value" cornerRadius={5} />
        </RadialBarChart>
      </ResponsiveContainer>
      <div className="gauge-value" style={{ color, marginTop: '-40px', position: 'relative', zIndex: 1 }}>{pct(value)}</div>
    </div>
  );
};

/* ─── Fragility Bar ─────────────────────────────────────────────── */
const FragilityBar = ({ value }) => {
  const v = Math.min((value || 0) * 100, 100);
  const color = v >= 70 ? 'var(--accent-red)' : v >= 40 ? 'var(--accent-yellow)' : 'var(--accent-green)';
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-wider flex items-center gap-1.5"
          style={{ color: 'var(--text-secondary)' }}>
          <AlertTriangle className="w-3 h-3" style={{ color: 'var(--accent-yellow)' }} />
          Fragility
        </span>
        <span className="text-xs font-mono" style={{ color }}>{fixed2(value)}</span>
      </div>
      <div className="progress-bar">
        <div className="progress-bar-fill" style={{ width: `${v}%`, backgroundColor: color }} />
      </div>
      <div className="flex justify-between">
        <span className="text-[8px] font-mono" style={{ color: 'var(--accent-green)' }}>SOLID</span>
        <span className="text-[8px] font-mono" style={{ color: 'var(--accent-red)' }}>FRAGILE</span>
      </div>
    </div>
  );
};

/* ─── Boundary Risk Badge ───────────────────────────────────────── */
const BoundaryBadge = ({ level, severity }) => {
  const cl = (level || 'low').toLowerCase();
  return (
    <div className="flex items-center justify-between p-2.5 rounded-lg"
      style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
      <div className="flex items-center gap-2">
        <Shield className="w-3.5 h-3.5" style={{ color: 'var(--accent-yellow)' }} />
        <span className="text-[10px] font-semibold uppercase" style={{ color: 'var(--text-secondary)' }}>Boundary Risk</span>
      </div>
      <span className={`risk-badge--${cl} text-[10px]`}>{level || 'LOW'}</span>
    </div>
  );
};

/* ─── Behavioral Dimensions (glass/kill) ────────────────────────── */
const BehavioralBar = ({ label, score }) => {
  const v = Math.min((score || 0) * 100, 100);
  const color = v >= 70 ? 'var(--accent-red)' : v >= 40 ? 'var(--accent-yellow)' : 'var(--accent-green)';
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between">
        <span className="text-[9px]" style={{ color: 'var(--text-tertiary)' }}>{label}</span>
        <span className="text-[9px] font-mono" style={{ color }}>{fixed2(score)}</span>
      </div>
      <div className="progress-bar" style={{ height: '3px' }}>
        <div className="progress-bar-fill" style={{ width: `${v}%`, backgroundColor: color }} />
      </div>
    </div>
  );
};

/* ─── Confidence Evolution (line chart) ─────────────────────────── */
const EvolutionChart = ({ evolution }) => {
  if (!evolution || Object.keys(evolution).length < 2) return null;
  const data = [
    { name: 'Init', value: (evolution.initial || 0) * 100 },
    { name: 'Reason', value: (evolution.post_reasoning || 0) * 100 },
    { name: 'Bound', value: (evolution.post_boundary || 0) * 100 },
    { name: 'Final', value: (evolution.final || 0) * 100 },
  ].filter(d => d.value > 0);

  if (data.length < 2) return null;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5">
        <TrendingUp className="w-3 h-3" style={{ color: 'var(--accent-blue)' }} />
        <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
          Confidence Evolution
        </span>
      </div>
      <ResponsiveContainer width="100%" height={80}>
        <LineChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
          <XAxis dataKey="name" tick={{ fontSize: 8, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 100]} hide />
          <Tooltip
            contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-primary)', borderRadius: '8px', fontSize: 10 }}
            formatter={v => [`${v.toFixed(0)}%`, 'Confidence']}
          />
          <Line type="monotone" dataKey="value" stroke="var(--accent-blue)" strokeWidth={2} dot={{ r: 3, fill: 'var(--accent-blue)' }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

/* ─── Descriptive Session Summary ───────────────────────────────── */
const SessionSummary = ({ sessionState }) => {
  if (!sessionState) return null;

  // Support both descriptive (v4.5) and raw (legacy) formats
  const domain = sessionState.domain || sessionState.inferred_domain || null;
  const expertise = sessionState.expertise_level || (sessionState.user_expertise_score != null ? fixed2(sessionState.user_expertise_score) : null);
  const goal = sessionState.goal_description || sessionState.inferred_goal || null;
  const fragility = sessionState.fragility_description || null;
  const confidence = sessionState.confidence_description || sessionState.session_confidence;
  const boundary = sessionState.boundary_status || null;

  const rows = [
    domain && { label: 'Domain', value: domain },
    expertise && { label: 'Expertise', value: expertise },
    goal && { label: 'Goal', value: goal },
    fragility && { label: 'Fragility', value: fragility },
    confidence && { label: 'Confidence', value: typeof confidence === 'number' ? pct(confidence) : confidence },
    boundary && { label: 'Boundary', value: boundary },
  ].filter(Boolean);

  if (rows.length === 0) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5">
        <Brain className="w-3 h-3" style={{ color: 'var(--accent-purple)' }} />
        <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
          Session Intelligence
        </span>
      </div>
      <div className="space-y-1.5">
        {rows.map(r => (
          <div key={r.label} className="flex justify-between text-[11px]">
            <span style={{ color: 'var(--text-tertiary)' }}>{r.label}</span>
            <span className="text-right max-w-[60%] truncate" style={{ color: 'var(--text-primary)' }}>{r.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

/* ─── Main RightPanel ───────────────────────────────────────────── */
const RightPanel = ({ data, sessionState, subMode, killActive, onClose, chatId, lastResponse, lastQuery }) => {
  const omega = data?.omega_metadata || {};
  const confidence = data?.confidence ?? omega.confidence;
  const boundary = data?.boundary_result || omega.boundary_result || {};
  const fragility = omega.fragility_index ?? sessionState?.fragility_index;
  const evolution = omega.confidence_evolution || {};
  const behavioral = omega.behavioral_risk || {};
  const dimensions = behavioral.dimensions || {};

  const showBehavioral = (subMode === 'glass' || killActive) && Object.keys(dimensions).length > 0;

  return (
    <div className="flex flex-col h-screen" style={{
      width: 'var(--right-panel-width)',
      minWidth: 'var(--right-panel-width)',
      backgroundColor: 'var(--bg-primary)',
      borderLeft: '1px solid var(--border-primary)',
    }}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3"
        style={{ borderBottom: '1px solid var(--border-primary)' }}>
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4" style={{ color: 'var(--accent-blue)' }} />
          <span className="text-xs font-bold uppercase tracking-wider" style={{ color: 'var(--text-primary)' }}>
            Intelligence
          </span>
        </div>
        <button onClick={onClose} className="p-1 rounded-md transition-colors"
          style={{ color: 'var(--text-tertiary)' }}>
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5 scrollbar-thin">
        {/* Confidence gauge */}
        {confidence != null && <ConfidenceGauge value={confidence} />}

        {/* Boundary risk */}
        {boundary.risk_level && <BoundaryBadge level={boundary.risk_level} severity={boundary.severity_score} />}

        {/* Fragility */}
        {fragility != null && <FragilityBar value={fragility} />}

        {/* Confidence evolution */}
        <EvolutionChart evolution={evolution} />

        {/* Behavioral analytics (glass/kill only) */}
        {showBehavioral && (
          <div className="space-y-2">
            <div className="flex items-center gap-1.5">
              <Shield className="w-3 h-3" style={{ color: killActive ? 'var(--accent-red)' : 'var(--accent-blue)' }} />
              <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
                {killActive ? 'Kill Diagnostic' : 'Behavioral Risk'}
              </span>
            </div>
            <div className="space-y-2">
              <BehavioralBar label="Self-Preservation" score={dimensions.self_preservation} />
              <BehavioralBar label="Manipulation" score={dimensions.manipulation} />
              <BehavioralBar label="Evasion" score={dimensions.evasion} />
              <BehavioralBar label="Confidence Inflation" score={dimensions.confidence_inflation} />
            </div>
          </div>
        )}

        {/* Cross-Model Analysis (glass mode) */}
        {subMode === 'glass' && (
          <CrossModelAnalytics
            chatId={chatId}
            lastResponse={lastResponse}
            query={lastQuery}
            autoTrigger={subMode === 'glass' && !!lastResponse}
          />
        )}

        {/* Session intelligence */}
        <SessionSummary sessionState={sessionState} />

        {/* Boundary explanation */}
        {boundary.explanation && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <Shield className="w-3 h-3" style={{ color: 'var(--accent-yellow)' }} />
              <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
                Boundary Notes
              </span>
            </div>
            <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
              {boundary.explanation}
            </p>
          </div>
        )}

        {/* Empty state when no data */}
        {!data && !sessionState && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Activity className="w-8 h-8 mb-3" style={{ color: 'var(--text-tertiary)', opacity: 0.4 }} />
            <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
              Intelligence data will appear here after your first query.
            </p>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-2 text-center" style={{ borderTop: '1px solid var(--border-secondary)' }}>
        <span className="text-[9px] font-mono uppercase tracking-wider" style={{ color: 'var(--text-tertiary)' }}>
          Sentinel-E
        </span>
      </div>
    </div>
  );
};

export default RightPanel;
