import React from 'react';
import { BookOpen, Globe, AlertTriangle, CheckCircle, XCircle, Shield } from 'lucide-react';
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

const SourceCard = ({ source, index }) => {
  const reliability = source.reliability_score || source.reliability || 0;
  const v = Math.min(reliability * 100, 100);
  const color = reliability >= 0.7 ? 'var(--accent-green)' : reliability >= 0.4 ? 'var(--accent-yellow)' : 'var(--accent-red)';
  return (
    <div className="p-3 rounded-lg space-y-2 transition-colors"
      style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)' }}>
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded"
            style={{ backgroundColor: 'var(--bg-primary)', color: 'var(--text-tertiary)' }}>#{index + 1}</span>
          <Globe className="w-3 h-3" style={{ color: 'var(--accent-purple)' }} />
          <span className="text-xs font-medium truncate max-w-[180px]" style={{ color: 'var(--text-primary)' }}>
            {source.title || source.name || 'Source'}
          </span>
        </div>
        <span className="text-[10px] font-mono" style={{ color }}>{fixed2(reliability)}</span>
      </div>
      {source.snippet && (
        <p className="text-[10px] leading-relaxed line-clamp-3" style={{ color: 'var(--text-secondary)' }}>{source.snippet}</p>
      )}
      <div className="flex items-center gap-2">
        <span className="text-[8px] uppercase font-bold" style={{ color: 'var(--text-tertiary)' }}>Reliability</span>
        <div className="flex-1 progress-bar" style={{ height: '3px' }}>
          <div className="progress-bar-fill" style={{ width: `${v}%`, backgroundColor: color }} />
        </div>
      </div>
    </div>
  );
};

const ContradictionCard = ({ contradiction }) => (
  <div className="p-3 rounded-lg" style={{ backgroundColor: 'rgba(239,68,68,0.05)', border: '1px solid rgba(239,68,68,0.15)' }}>
    <div className="flex items-center gap-2 mb-2">
      <XCircle className="w-3 h-3" style={{ color: 'var(--accent-red)' }} />
      <span className="text-[10px] uppercase font-bold" style={{ color: 'var(--accent-red)' }}>Contradiction</span>
    </div>
    <p className="text-[10px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
      {typeof contradiction === 'string' ? contradiction : (contradiction.description || contradiction.detail || JSON.stringify(contradiction))}
    </p>
  </div>
);

/**
 * EvidenceConsole — Evidence-backed reasoning sub-mode.
 * In v4.5, primary output is rendered as markdown by ChatThread.
 * This shows supplementary source/contradiction metrics.
 */
const EvidenceConsole = ({ data }) => {
  if (!data) return null;

  const runId = data.chat_id || data.run_id;
  const omega = data.omega_metadata || {};
  const confidence = data.confidence ?? omega.confidence;
  const boundary = data.boundary_result || omega.boundary_result || {};
  const evidence = omega.evidence_result || {};
  const sources = evidence.sources || [];
  const contradictions = evidence.contradictions || [];

  const avgReliability = sources.length > 0
    ? sources.reduce((sum, s) => sum + (s.reliability_score || s.reliability || 0), 0) / sources.length
    : null;

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center gap-2 pb-2" style={{ borderBottom: '1px solid var(--accent-purple)' }}>
        <BookOpen className="w-4 h-4" style={{ color: 'var(--accent-purple)' }} />
        <h3 className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--accent-purple)' }}>
          Evidence Console
        </h3>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <MetricCard label="Sources" value={sources.length} color="var(--accent-purple)" />
        <MetricCard label="Avg Reliability" value={avgReliability != null ? fixed2(avgReliability) : '—'}
          color={avgReliability >= 0.7 ? 'var(--accent-green)' : 'var(--accent-yellow)'} />
        <MetricCard label="Contradictions" value={contradictions.length}
          color={contradictions.length > 0 ? 'var(--accent-red)' : 'var(--accent-green)'} />
        <MetricCard label="Confidence" value={pct(confidence)}
          color={confidence >= 0.8 ? 'var(--accent-green)' : 'var(--accent-yellow)'} />
      </div>

      {sources.length > 0 && (
        <div className="space-y-2">
          {sources.map((source, i) => <SourceCard key={i} source={source} index={i} />)}
        </div>
      )}

      {contradictions.length > 0 && (
        <div className="space-y-2">
          {contradictions.map((c, i) => <ContradictionCard key={i} contradiction={c} />)}
        </div>
      )}

      {contradictions.length === 0 && sources.length > 0 && (
        <div className="p-3 rounded-lg flex items-center gap-2"
          style={{ backgroundColor: 'rgba(34,197,94,0.05)', border: '1px solid rgba(34,197,94,0.15)' }}>
          <CheckCircle className="w-4 h-4" style={{ color: 'var(--accent-green)' }} />
          <span className="text-[10px]" style={{ color: 'var(--accent-green)' }}>No contradictions detected across {sources.length} sources</span>
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
          <FeedbackButton runId={runId} mode="experimental" subMode="evidence" />
        </div>
      )}
    </div>
  );
};

export default EvidenceConsole;
