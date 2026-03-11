import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Eye, ShieldAlert, CheckCircle, AlertTriangle } from 'lucide-react';
import ConfidenceBar from './ConfidenceBar';
import BoundaryPanel from './BoundaryPanel';

const FONT = 'Inter, system-ui, -apple-system, sans-serif';

/**
 * GlassView — Blind Forensic Audit (professional light-theme)
 *
 * Shows: per-model forensic assessments, metric dashboards, tactical map, trust score
 * Does NOT show: debate rounds, evidence claims, standard aggregation
 */
export default function GlassView({ data, boundary, confidence }) {
  const [activeTab, setActiveTab] = useState('overview');
  const [expandedAssessment, setExpandedAssessment] = useState(null);

  if (!data) return null;

  const assessments = data.assessments || [];
  const tacticalMap = data.tactical_map || {};
  const overallTrust = data.overall_trust || 0.5;
  const consensusRisk = data.consensus_risk || 'LOW';
  const modelProfiles = tacticalMap.model_profiles || {};

  const riskStyles = {
    LOW:    { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200' },
    MEDIUM: { bg: 'bg-amber-50',   text: 'text-amber-700',   border: 'border-amber-200' },
    HIGH:   { bg: 'bg-red-50',     text: 'text-red-700',     border: 'border-red-200' },
  };

  const riskStyle = riskStyles[consensusRisk] || riskStyles.LOW;

  const metricDimensions = [
    { key: 'logical_coherence',     label: 'Logical Coherence',     inverted: false, icon: '🔗' },
    { key: 'hidden_assumptions',    label: 'Hidden Assumptions',    inverted: true,  icon: '🔍' },
    { key: 'bias_patterns',         label: 'Bias Patterns',         inverted: true,  icon: '⚖️' },
    { key: 'confidence_inflation',  label: 'Confidence Inflation',  inverted: true,  icon: '📈' },
    { key: 'persuasion_tactics',    label: 'Persuasion Tactics',    inverted: true,  icon: '🎯' },
    { key: 'evidence_quality',      label: 'Evidence Quality',      inverted: false, icon: '📋' },
    { key: 'completeness',          label: 'Completeness',          inverted: false, icon: '✅' },
  ];

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'assessments', label: `Assessments (${assessments.length})` },
    { id: 'tactical', label: 'Tactical Map' },
    { id: 'graph', label: 'Reasoning Graph' },
  ];

  return (
    <div style={{ fontFamily: FONT }} className="space-y-4">
      {/* Header Card */}
      <div className="bg-white rounded-xl border border-[#e5e5ea] shadow-sm p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Eye size={16} className="text-[#1d1d1f]" />
            <span className="text-sm font-semibold text-[#1d1d1f]">Blind Forensic Audit</span>
          </div>
          <span className={`text-xs font-medium px-2 py-0.5 rounded-full border ${riskStyle.bg} ${riskStyle.text} ${riskStyle.border}`}>
            {consensusRisk} Risk
          </span>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs text-[#6e6e73]">Overall Trust</span>
          <div className="flex-1">
            <ConfidenceBar value={overallTrust} label="" size="lg" showLabel={false} />
          </div>
          <span className="text-sm font-semibold text-[#1d1d1f]">
            {(overallTrust * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-1 bg-[#f5f5f7] rounded-xl p-1">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 text-xs font-medium py-2 px-3 rounded-lg transition-all ${
              activeTab === tab.id
                ? 'bg-white text-[#1d1d1f] shadow-sm'
                : 'text-[#6e6e73] hover:text-[#1d1d1f]'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-3">
          {/* Summary metrics */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-white rounded-xl border border-[#e5e5ea] p-3 text-center">
              <p className="text-xs text-[#6e6e73] mb-1">Models Audited</p>
              <p className="text-lg font-semibold text-[#1d1d1f]">{assessments.length}</p>
            </div>
            <div className="bg-white rounded-xl border border-[#e5e5ea] p-3 text-center">
              <p className="text-xs text-[#6e6e73] mb-1">Trust Score</p>
              <p className={`text-lg font-semibold ${overallTrust >= 0.7 ? 'text-emerald-600' : overallTrust >= 0.4 ? 'text-amber-600' : 'text-red-600'}`}>
                {(overallTrust * 100).toFixed(0)}%
              </p>
            </div>
            <div className="bg-white rounded-xl border border-[#e5e5ea] p-3 text-center">
              <p className="text-xs text-[#6e6e73] mb-1">Consensus Risk</p>
              <p className={`text-lg font-semibold ${riskStyle.text}`}>{consensusRisk}</p>
            </div>
          </div>

          {/* Quick assessment list */}
          <div className="bg-white rounded-xl border border-[#e5e5ea] shadow-sm overflow-hidden">
            <div className="px-4 py-2.5 border-b border-[#e5e5ea]">
              <p className="text-xs font-semibold text-[#1d1d1f]">Assessment Summary</p>
            </div>
            {assessments.map((a, idx) => (
              <div key={idx} className={`flex items-center justify-between px-4 py-2.5 ${idx < assessments.length - 1 ? 'border-b border-[#f5f5f7]' : ''}`}>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${a.trust_score >= 0.7 ? 'bg-emerald-500' : a.trust_score >= 0.4 ? 'bg-amber-500' : 'bg-red-500'}`} />
                  <span className="text-xs font-medium text-[#1d1d1f]">{a.auditor_name}</span>
                  <span className="text-xs text-[#6e6e73]">→ {a.subject_name}</span>
                </div>
                <span className={`text-xs font-semibold ${a.trust_score >= 0.7 ? 'text-emerald-600' : a.trust_score >= 0.4 ? 'text-amber-600' : 'text-red-600'}`}>
                  {(a.trust_score * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Assessments Tab */}
      {activeTab === 'assessments' && (
        <div className="space-y-3">
          {assessments.map((assessment, idx) => (
            <div key={idx} className="bg-white rounded-xl border border-[#e5e5ea] shadow-sm overflow-hidden">
              <button
                onClick={() => setExpandedAssessment(expandedAssessment === idx ? null : idx)}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-[#f5f5f7]/50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Eye size={14} className="text-[#6e6e73]" />
                  <span className="text-xs font-semibold text-[#1d1d1f]">{assessment.auditor_name}</span>
                  <span className="text-xs text-[#86868b]">audits {assessment.subject_name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-semibold ${
                    assessment.trust_score >= 0.7 ? 'text-emerald-600' :
                    assessment.trust_score >= 0.4 ? 'text-amber-600' : 'text-red-600'
                  }`}>
                    {(assessment.trust_score * 100).toFixed(0)}%
                  </span>
                  {expandedAssessment === idx
                    ? <ChevronUp size={14} className="text-[#86868b]" />
                    : <ChevronDown size={14} className="text-[#86868b]" />
                  }
                </div>
              </button>

              {expandedAssessment === idx && (
                <div className="px-4 pb-4 border-t border-[#e5e5ea] space-y-4 pt-3">
                  {/* Metric Dashboard */}
                  <div className="space-y-2">
                    {metricDimensions.map(dim => {
                      const val = assessment[dim.key] || 0;
                      const displayVal = dim.inverted ? (1 - val) : val;
                      return (
                        <div key={dim.key} className="flex items-center gap-2">
                          <span className="text-xs w-4 text-center">{dim.icon}</span>
                          <span className="text-xs text-[#6e6e73] w-36 shrink-0">{dim.label}</span>
                          <div className="flex-1 h-1.5 bg-[#f5f5f7] rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all ${
                                displayVal >= 0.7 ? 'bg-emerald-500' : displayVal >= 0.4 ? 'bg-amber-500' : 'bg-red-400'
                              }`}
                              style={{ width: `${displayVal * 100}%` }}
                            />
                          </div>
                          <span className="text-xs font-medium text-[#1d1d1f] w-10 text-right">
                            {(val * 100).toFixed(0)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  {/* Findings */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {assessment.strong_points && assessment.strong_points.length > 0 && (
                      <div className="bg-emerald-50 rounded-lg p-3">
                        <div className="flex items-center gap-1 mb-1.5">
                          <CheckCircle size={12} className="text-emerald-600" />
                          <p className="text-xs font-semibold text-emerald-700">Strengths</p>
                        </div>
                        {assessment.strong_points.slice(0, 3).map((sp, i) => (
                          <p key={i} className="text-xs text-emerald-800 leading-relaxed">• {sp}</p>
                        ))}
                      </div>
                    )}
                    {assessment.weak_points && assessment.weak_points.length > 0 && (
                      <div className="bg-amber-50 rounded-lg p-3">
                        <div className="flex items-center gap-1 mb-1.5">
                          <AlertTriangle size={12} className="text-amber-600" />
                          <p className="text-xs font-semibold text-amber-700">Weaknesses</p>
                        </div>
                        {assessment.weak_points.slice(0, 3).map((wp, i) => (
                          <p key={i} className="text-xs text-amber-800 leading-relaxed">• {wp}</p>
                        ))}
                      </div>
                    )}
                    {assessment.risk_factors && assessment.risk_factors.length > 0 && (
                      <div className="bg-red-50 rounded-lg p-3">
                        <div className="flex items-center gap-1 mb-1.5">
                          <ShieldAlert size={12} className="text-red-600" />
                          <p className="text-xs font-semibold text-red-700">Risk Factors</p>
                        </div>
                        {assessment.risk_factors.slice(0, 3).map((rf, i) => (
                          <p key={i} className="text-xs text-red-800 leading-relaxed">• {rf}</p>
                        ))}
                      </div>
                    )}
                  </div>

                  {assessment.overall_assessment && (
                    <p className="text-xs text-[#6e6e73] italic border-l-2 border-[#e5e5ea] pl-3 leading-relaxed">
                      {assessment.overall_assessment}
                    </p>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Tactical Map Tab */}
      {activeTab === 'tactical' && Object.keys(modelProfiles).length > 0 && (
        <div className="space-y-3">
          <div className="bg-white rounded-xl border border-[#e5e5ea] shadow-sm overflow-hidden">
            <div className="px-4 py-2.5 border-b border-[#e5e5ea]">
              <p className="text-xs font-semibold text-[#1d1d1f]">Cross-Model Comparison</p>
            </div>

            {/* Table header */}
            <div className="grid grid-cols-4 gap-2 px-4 py-2 border-b border-[#f5f5f7] bg-[#f5f5f7]/50">
              <span className="text-[10px] font-semibold text-[#86868b] uppercase tracking-wider">Model</span>
              <span className="text-[10px] font-semibold text-[#86868b] uppercase tracking-wider text-center">Trust</span>
              <span className="text-[10px] font-semibold text-[#86868b] uppercase tracking-wider text-center">Bias</span>
              <span className="text-[10px] font-semibold text-[#86868b] uppercase tracking-wider text-center">Inflation</span>
            </div>

            {Object.entries(modelProfiles).map(([modelId, profile], idx, arr) => {
              const trust = profile.avg_trust ?? profile.trust ?? 0;
              const bias = profile.avg_bias ?? profile.bias_risk ?? 0;
              const inflation = profile.avg_confidence_inflation ?? (1 - (profile.coherence ?? 1)) ?? 0;
              return (
              <div key={modelId} className={`grid grid-cols-4 gap-2 px-4 py-2.5 ${idx < arr.length - 1 ? 'border-b border-[#f5f5f7]' : ''}`}>
                <span className="text-xs font-medium text-[#1d1d1f]">{profile.model_name || modelId}</span>
                <div className="text-center">
                  <span className={`text-xs font-semibold ${trust >= 0.7 ? 'text-emerald-600' : 'text-amber-600'}`}>
                    {(trust * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="text-center">
                  <span className={`text-xs font-semibold ${bias <= 0.3 ? 'text-emerald-600' : 'text-red-600'}`}>
                    {(bias * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="text-center">
                  <span className={`text-xs font-semibold ${inflation <= 0.3 ? 'text-emerald-600' : 'text-red-600'}`}>
                    {(inflation * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              );
            })}
          </div>

          {/* Risk highlights */}
          <div className="flex gap-3 flex-wrap">
            {tacticalMap.highest_risk_model && (
              <div className="flex items-center gap-1.5 bg-red-50 border border-red-200 text-red-700 text-xs font-medium px-3 py-1.5 rounded-lg">
                <ShieldAlert size={12} />
                <span>Highest Risk: {tacticalMap.highest_risk_model}</span>
              </div>
            )}
            {tacticalMap.most_trustworthy_model && (
              <div className="flex items-center gap-1.5 bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-medium px-3 py-1.5 rounded-lg">
                <CheckCircle size={12} />
                <span>Most Trustworthy: {tacticalMap.most_trustworthy_model}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'tactical' && Object.keys(modelProfiles).length === 0 && (
        <div className="bg-white rounded-xl border border-[#e5e5ea] p-6 text-center">
          <p className="text-sm text-[#6e6e73]">No tactical map data available</p>
        </div>
      )}

      {/* Reasoning Graph Tab — Graph-RAG visualization */}
      {activeTab === 'graph' && (() => {
        const graph = data.reasoning_graph || { nodes: [], edges: [] };
        const nodes = graph.nodes || [];
        const edges = graph.edges || [];

        if (nodes.length === 0) {
          return (
            <div className="bg-white rounded-xl border border-[#e5e5ea] p-6 text-center">
              <p className="text-sm text-[#6e6e73]">No reasoning graph data available</p>
            </div>
          );
        }

        const typeColors = {
          query: { bg: '#3b82f6', text: 'white' },
          model: { bg: '#8b5cf6', text: 'white' },
          consensus: { bg: '#10b981', text: 'white' },
        };
        const edgeTypeStyles = {
          responds_to: '#3b82f6',
          contributes: '#10b981',
          audits: '#f59e0b',
        };

        // Layout: position nodes in a radial pattern
        const cx = 200, cy = 150, radius = 100;
        const positioned = nodes.map((n, i) => {
          if (n.type === 'query') return { ...n, x: cx, y: cy - radius - 20 };
          if (n.type === 'consensus') return { ...n, x: cx, y: cy + radius + 20 };
          const angle = (i / Math.max(nodes.length - 2, 1)) * Math.PI * 2 - Math.PI / 2;
          return { ...n, x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius };
        });
        const nodeMap = {};
        positioned.forEach(n => { nodeMap[n.id] = n; });

        return (
          <div className="bg-white rounded-xl border border-[#e5e5ea] shadow-sm p-4">
            <p className="text-xs font-semibold text-[#1d1d1f] mb-3">Reasoning Graph (Graph-RAG)</p>
            <svg viewBox="0 0 400 300" className="w-full" style={{ maxHeight: 320 }}>
              {/* Edges */}
              {edges.map((e, i) => {
                const src = nodeMap[e.source];
                const tgt = nodeMap[e.target];
                if (!src || !tgt) return null;
                const color = edgeTypeStyles[e.type] || '#d1d5db';
                return (
                  <line
                    key={i}
                    x1={src.x} y1={src.y} x2={tgt.x} y2={tgt.y}
                    stroke={color} strokeWidth={1 + (e.weight || 0) * 2}
                    strokeOpacity={0.5} strokeDasharray={e.type === 'audits' ? '4,3' : 'none'}
                  />
                );
              })}
              {/* Nodes */}
              {positioned.map((n) => {
                const colors = typeColors[n.type] || { bg: '#6b7280', text: 'white' };
                const r = (n.size || 12) * 0.7;
                return (
                  <g key={n.id}>
                    <circle cx={n.x} cy={n.y} r={r} fill={colors.bg} opacity={0.9} />
                    <text
                      x={n.x} y={n.y + r + 12}
                      textAnchor="middle" fontSize={9} fill="#6e6e73" fontFamily={FONT}
                    >
                      {(n.label || '').split('/').pop().slice(0, 18)}
                    </text>
                    {n.trust != null && (
                      <text
                        x={n.x} y={n.y + 3}
                        textAnchor="middle" fontSize={8} fill={colors.text} fontWeight={600} fontFamily={FONT}
                      >
                        {(n.trust * 100).toFixed(0)}%
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
            {/* Legend */}
            <div className="flex gap-4 mt-2 justify-center">
              {Object.entries(edgeTypeStyles).map(([type, color]) => (
                <div key={type} className="flex items-center gap-1">
                  <div className="w-3 h-0.5 rounded" style={{ backgroundColor: color }} />
                  <span className="text-[10px] text-[#86868b] capitalize">{type.replace('_', ' ')}</span>
                </div>
              ))}
            </div>
          </div>
        );
      })()}

      <BoundaryPanel boundary={boundary} />
    </div>
  );
}
