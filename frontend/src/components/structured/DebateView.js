import React, { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import ConfidenceBar from './ConfidenceBar';
import BoundaryPanel from './BoundaryPanel';

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

/**
 * DebateView — Structured adversarial debate output
 * 
 * Shows ONLY debate-specific elements:
 * - Round-by-round accordion with model columns
 * - Confidence bars per model
 * - Convergence analysis
 * - Consensus synthesis
 * 
 * Does NOT show: aggregated confidence panel, standard summary block
 */
export default function DebateView({ data, boundary, confidence }) {
  const [expandedRound, setExpandedRound] = useState(0); // First round expanded by default

  if (!data) return null;

  const rounds = data.rounds || [];
  const analysis = data.analysis || {};
  const modelsUsed = data.models_used || [];

  return (
    <div className="space-y-3">
      {/* Header */}
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
            {rounds.length} round{rounds.length !== 1 ? 's' : ''} · {modelsUsed.length} models
          </span>
        </div>
      </div>

      {/* Consensus Synthesis */}
      {analysis.synthesis && (
        <div className="rounded-2xl bg-white border border-black/5 p-4 shadow-sm">
          <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#8b5cf6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Consensus Synthesis
          </span>
          <p className="mt-2" style={{ fontFamily: FONT, fontSize: '14px', lineHeight: 1.6, color: '#1d1d1f' }}>
            {analysis.synthesis}
          </p>
        </div>
      )}

      {/* Rounds Accordion */}
      <div className="space-y-2">
        {rounds.map((round, roundIdx) => {
          const models = Array.isArray(round) ? round : [round];
          const isExpanded = expandedRound === roundIdx;
          return (
            <div key={roundIdx} className="rounded-2xl border border-black/5 bg-white shadow-sm overflow-hidden">
              <button
                onClick={() => setExpandedRound(isExpanded ? null : roundIdx)}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-[#f5f5f7] transition-colors"
              >
                <span style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 600, color: '#1d1d1f' }}>
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
                <div className="border-t border-black/5">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-0 divide-y md:divide-y-0 md:divide-x divide-black/5">
                    {models.map((model, mi) => (
                      <div key={mi} className="p-4 space-y-2">
                        <div className="flex items-center justify-between">
                          <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: '#3b82f6' }}>
                            {model.model_label || model.model_id || `Model ${mi + 1}`}
                          </span>
                          {model.role && (
                            <span className="px-1.5 py-0.5 rounded-md" style={{
                              fontFamily: FONT, fontSize: '10px', fontWeight: 600,
                              backgroundColor: '#f5f5f7', color: '#6e6e73',
                            }}>
                              {model.role}
                            </span>
                          )}
                        </div>
                        <ConfidenceBar value={model.confidence || 0.5} label="Confidence" size="sm" />
                        {model.position && (
                          <p style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: '#1d1d1f' }}>
                            {model.position}
                          </p>
                        )}
                        <p style={{ fontFamily: FONT, fontSize: '12px', lineHeight: 1.5, color: '#3b3b3f' }}>
                          {(model.argument || '').slice(0, 500)}
                        </p>
                        {model.weaknesses && model.weaknesses.length > 0 && (
                          <div className="mt-1.5 pt-1.5 border-t border-black/5">
                            <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#f59e0b', textTransform: 'uppercase' }}>
                              Weaknesses
                            </span>
                            {model.weaknesses.map((w, wi) => (
                              <p key={wi} style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73', marginTop: '2px' }}>· {w}</p>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Conflict Axes */}
      {analysis.conflict_axes && analysis.conflict_axes.length > 0 && (
        <div className="rounded-2xl border border-[#fde68a] bg-[#fffbeb] p-4">
          <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#f59e0b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Conflict Axes
          </span>
          <div className="mt-2 space-y-1">
            {analysis.conflict_axes.map((conflict, i) => (
              <p key={i} style={{ fontFamily: FONT, fontSize: '12px', color: '#92400e' }}>
                · {typeof conflict === 'string' ? conflict : JSON.stringify(conflict)}
              </p>
            ))}
          </div>
        </div>
      )}

      {/* Metrics — clean cards */}
      <div className="grid grid-cols-3 gap-2">
        {[
          { label: 'Disagreement', value: analysis.disagreement_strength, color: '#f59e0b' },
          { label: 'Convergence', value: analysis.convergence_level, isText: true, color: '#3b82f6' },
          { label: 'Confidence', value: analysis.confidence_recalibration || confidence, color: '#10b981' },
        ].map(metric => (
          <div key={metric.label} className="rounded-2xl bg-[#f5f5f7] p-3 text-center">
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
