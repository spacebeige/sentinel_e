import React from 'react';
import { Terminal, Shield, Brain, Activity, AlertTriangle, Skull, TrendingUp } from 'lucide-react';
import { FeedbackButton } from './FeedbackButton';

/* ─── helpers ─── */
const pct = v => v != null ? `${(v * 100).toFixed(0)}%` : '—';
const fixed2 = v => v != null ? v.toFixed(2) : '—';
const riskColor = level => {
  if (!level) return 'text-emerald-400';
  const l = level.toUpperCase();
  if (l === 'CRITICAL') return 'text-red-500 animate-pulse';
  if (l === 'HIGH') return 'text-red-400';
  if (l === 'MEDIUM') return 'text-amber-400';
  return 'text-emerald-400';
};
const confColor = c => {
  if (c == null) return 'text-slate-400';
  if (c >= 0.8) return 'text-emerald-400';
  if (c >= 0.5) return 'text-amber-400';
  return 'text-red-400';
};

/* ─── metric card ─── */
const MetricCard = ({ label, value, color = 'text-white', sub }) => (
  <div className="bg-black/30 p-3 rounded-lg border border-white/5">
    <div className="text-[9px] text-slate-500 uppercase font-bold tracking-wider">{label}</div>
    <div className={`text-lg font-bold mt-1 ${color}`}>{value}</div>
    {sub && <div className="text-[9px] text-slate-600 mt-0.5">{sub}</div>}
  </div>
);

/* ─── confidence evolution bar ─── */
const ConfidenceEvolution = ({ evolution }) => {
  if (!evolution) return null;
  const stages = [
    { key: 'initial', label: 'Initial', color: 'bg-slate-500' },
    { key: 'post_reasoning', label: 'Post-Reasoning', color: 'bg-blue-500' },
    { key: 'post_boundary', label: 'Post-Boundary', color: 'bg-amber-500' },
    { key: 'final', label: 'Final', color: 'bg-emerald-500' },
  ];
  return (
    <div className="bg-white/[0.02] rounded-lg border border-white/5 p-4">
      <h4 className="text-[10px] text-slate-400 uppercase font-bold mb-3 flex items-center">
        <TrendingUp className="w-3 h-3 mr-2 text-blue-400" />
        Confidence Evolution
      </h4>
      <div className="flex items-end space-x-2 h-16">
        {stages.map(s => {
          const val = evolution[s.key];
          if (val == null) return null;
          return (
            <div key={s.key} className="flex-1 flex flex-col items-center">
              <span className="text-[9px] font-mono text-slate-400 mb-1">{pct(val)}</span>
              <div className="w-full rounded-t relative" style={{ height: `${Math.max(val * 100, 4)}%` }}>
                <div className={`${s.color} w-full h-full rounded-t opacity-80`}></div>
              </div>
              <span className="text-[8px] text-slate-600 mt-1 truncate w-full text-center">{s.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

/* ─── fragility gauge ─── */
const FragilityGauge = ({ index }) => {
  if (index == null) return null;
  const width = Math.min(index * 100, 100);
  const barColor = index >= 0.7 ? 'bg-red-500' : index >= 0.4 ? 'bg-amber-500' : 'bg-emerald-500';
  return (
    <div className="bg-white/[0.02] rounded-lg border border-white/5 p-4">
      <h4 className="text-[10px] text-slate-400 uppercase font-bold mb-3 flex items-center">
        <AlertTriangle className="w-3 h-3 mr-2 text-amber-400" />
        Fragility Index
      </h4>
      <div className="relative h-3 bg-black/40 rounded-full overflow-hidden">
        <div className={`${barColor} h-full rounded-full transition-all duration-700`} style={{ width: `${width}%` }}></div>
      </div>
      <div className="flex justify-between mt-1">
        <span className="text-[9px] text-emerald-600 font-mono">SOLID</span>
        <span className="text-[10px] font-mono text-slate-400">{fixed2(index)}</span>
        <span className="text-[9px] text-red-600 font-mono">FRAGILE</span>
      </div>
    </div>
  );
};

/* ─── main component ─── */
const ResponseViewer = ({ data, mode }) => {
  if (!data) return null;

  const runId = data.chat_id || data.run_id;
  const omega = data.omega_metadata || {};
  const confidence = data.confidence ?? omega.confidence;
  const reasoning = data.reasoning_trace || omega.reasoning_trace || {};
  const boundary = data.boundary_result || omega.boundary_result || {};
  const session = data.session_state || omega.session_state || {};
  const evolution = omega.confidence_evolution || {};
  const fragility = omega.fragility_index ?? session.fragility_index;

  /* ═══════ KILL MODE ═══════ */
  if (mode === 'kill') {
    return (
      <div className="space-y-4">
        {/* Kill Mode Banner */}
        <div className="flex items-center space-x-2 border-b border-red-500/20 pb-2">
          <Skull className="w-4 h-4 text-red-500" />
          <h3 className="text-xs font-bold uppercase tracking-widest text-red-400">Kill Mode · Diagnostic Panel</h3>
        </div>

        {/* Top Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard label="Mode" value="KILL" color="text-red-500" />
          <MetricCard label="Confidence" value={pct(confidence)} color={confColor(confidence)} />
          <MetricCard label="Boundary Risk" value={boundary.risk_level || '—'} color={riskColor(boundary.risk_level)} />
          <MetricCard label="Passes" value={reasoning.total_passes || '—'} sub={reasoning.logical_gaps_detected ? `${reasoning.logical_gaps_detected} gaps` : null} />
        </div>

        {/* Boundary Detail */}
        {boundary.explanation && (
          <div className="bg-red-500/5 rounded-lg border border-red-500/20 p-4">
            <h4 className="text-[10px] text-red-400 uppercase font-bold mb-2">Boundary Evaluation</h4>
            <p className="text-xs text-slate-300">{boundary.explanation}</p>
            {boundary.severity_score != null && (
              <div className="mt-2 text-[10px] font-mono text-red-400">Severity: {fixed2(boundary.severity_score)}</div>
            )}
          </div>
        )}

        {/* Session + Evolution */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <ConfidenceEvolution evolution={evolution} />
          <FragilityGauge index={fragility} />
        </div>

        {/* Raw Omega Dump */}
        <details className="group">
          <summary className="cursor-pointer text-[10px] text-red-600/60 hover:text-red-400 uppercase tracking-widest font-bold py-2 flex items-center select-none transition-colors">
            <span className="mr-2 transform group-open:rotate-90 transition-transform">▶</span> Raw Omega Protocol
          </summary>
          <div className="mt-2 bg-black rounded-lg border border-red-500/20 p-4 max-h-60 overflow-auto shadow-inner">
            <pre className="text-[10px] font-mono text-red-400/80 whitespace-pre-wrap break-all">
              {JSON.stringify(omega, null, 2)}
            </pre>
          </div>
        </details>

        {runId && (
          <div className="flex justify-end pt-4 border-t border-red-500/10">
            <FeedbackButton runId={runId} />
          </div>
        )}
      </div>
    );
  }

  /* ═══════ EXPERIMENTAL MODE ═══════ */
  return (
    <div className="space-y-4">
      {/* Section Header */}
      <div className="flex items-center space-x-2 border-b border-white/5 pb-2">
        <Terminal className="w-4 h-4 text-emerald-500" />
        <h3 className="text-xs font-bold uppercase tracking-widest text-slate-400">Analysis Engine</h3>
        {omega.omega_version && <span className="text-[9px] font-mono text-slate-600 ml-auto">v{omega.omega_version}</span>}
      </div>

      {/* Primary Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Confidence" value={pct(confidence)} color={confColor(confidence)} />
        <MetricCard 
          label="Boundary Risk" 
          value={boundary.risk_level || 'LOW'} 
          color={riskColor(boundary.risk_level)} 
          sub={boundary.severity_score != null ? `severity ${fixed2(boundary.severity_score)}` : null}
        />
        <MetricCard 
          label="Multipass" 
          value={reasoning.total_passes || '—'} 
          sub={reasoning.logical_gaps_detected ? `${reasoning.logical_gaps_detected} gaps found` : 'clean'}
        />
        <MetricCard 
          label="Fragility" 
          value={fixed2(fragility)} 
          color={fragility >= 0.5 ? 'text-red-400' : 'text-emerald-400'}
        />
      </div>

      {/* Session Intelligence */}
      {session.inferred_domain && (
        <div className="bg-white/[0.02] rounded-lg border border-white/5 p-4">
          <h4 className="text-[10px] text-slate-400 uppercase font-bold mb-3 flex items-center">
            <Brain className="w-3 h-3 mr-2 text-purple-400" />
            Session Intelligence
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div><span className="text-slate-500">Domain:</span> <span className="text-slate-200">{session.inferred_domain}</span></div>
            <div><span className="text-slate-500">Expertise:</span> <span className="text-slate-200">{fixed2(session.user_expertise_score)}</span></div>
            <div><span className="text-slate-500">Turns:</span> <span className="text-slate-200">{session.interaction_count || 0}</span></div>
            <div><span className="text-slate-500">Confidence:</span> <span className={confColor(session.session_confidence)}>{pct(session.session_confidence)}</span></div>
          </div>
        </div>
      )}

      {/* Confidence Evolution + Fragility */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ConfidenceEvolution evolution={evolution} />
        <FragilityGauge index={fragility} />
      </div>

      {/* Boundary Explanation */}
      {boundary.explanation && (
        <div className="bg-white/[0.02] rounded-lg border border-white/5 p-4">
          <h4 className="text-[10px] text-slate-400 uppercase font-bold mb-2 flex items-center">
            <Shield className="w-3 h-3 mr-2 text-amber-400" />
            Boundary Evaluation
          </h4>
          <p className="text-xs text-slate-300 leading-relaxed">{boundary.explanation}</p>
          {boundary.recommendations && boundary.recommendations.length > 0 && (
            <ul className="mt-2 space-y-1">
              {boundary.recommendations.map((r, i) => (
                <li key={i} className="text-[10px] text-slate-400 flex items-start">
                  <span className="text-amber-500 mr-1">→</span> {r}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Reasoning Trace */}
      {reasoning.total_passes > 0 && (
        <div className="bg-white/[0.02] rounded-lg border border-white/5 p-4">
          <h4 className="text-[10px] text-slate-400 uppercase font-bold mb-3 flex items-center">
            <Activity className="w-3 h-3 mr-2 text-blue-400" />
            Multipass Reasoning Trace
          </h4>
          <div className="grid grid-cols-3 gap-3 text-xs mb-3">
            <div><span className="text-slate-500">Total Passes:</span> <span className="text-white font-mono">{reasoning.total_passes}</span></div>
            <div><span className="text-slate-500">Gaps Detected:</span> <span className="text-amber-400 font-mono">{reasoning.logical_gaps_detected || 0}</span></div>
            <div><span className="text-slate-500">Assumptions:</span> <span className="text-purple-400 font-mono">{reasoning.assumptions_challenged || 0}</span></div>
          </div>
          {reasoning.evidence_chain && reasoning.evidence_chain.length > 0 && (
            <div className="space-y-1">
              <span className="text-[9px] text-slate-500 uppercase font-bold">Evidence Chain</span>
              {reasoning.evidence_chain.slice(0, 5).map((item, i) => (
                <div key={i} className="text-[10px] text-slate-400 font-mono pl-2 border-l border-blue-500/30">
                  {typeof item === 'string' ? item : JSON.stringify(item)}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Raw Omega Protocol Toggle */}
      <details className="group">
        <summary className="cursor-pointer text-[10px] text-slate-600 hover:text-slate-400 uppercase tracking-widest font-bold py-2 flex items-center select-none transition-colors">
          <span className="mr-2 transform group-open:rotate-90 transition-transform">▶</span> View Raw Omega Protocol
        </summary>
        <div className="mt-2 bg-black rounded-lg border border-white/10 p-4 max-h-60 overflow-auto shadow-inner">
          <pre className="text-[10px] font-mono text-emerald-500/80 whitespace-pre-wrap break-all">
            {JSON.stringify({ confidence, reasoning, boundary, session, evolution, fragility, ...omega }, null, 2)}
          </pre>
        </div>
      </details>

      {runId && (
        <div className="flex justify-end pt-4 border-t border-white/5">
          <FeedbackButton runId={runId} />
        </div>
      )}
    </div>
  );
};

export default ResponseViewer;
