import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Users, ArrowRight, MessageSquare, Sparkles } from 'lucide-react';
import ConfidenceBar from './ConfidenceBar';
import BoundaryPanel from './BoundaryPanel';

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

/**
 * SynthesisView — Collaborative Reasoning Display
 *
 * Shows: draft → peer revisions → final synthesis with consensus metrics
 * Opposite of DebateView: models build together instead of arguing
 */
export default function SynthesisView({ data, boundary, confidence }) {
  const [expandedRevision, setExpandedRevision] = useState(null);
  const [showDraft, setShowDraft] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  if (!data) return null;

  const draft = data.draft || '';
  const draftModel = data.draft_model || 'Unknown';
  const draftScore = data.draft_score || 0;
  const revisions = data.revisions || [];
  const consensusScore = data.consensus_score || 0;
  const improvementDelta = data.improvement_delta || 0;
  const modelsParticipated = data.models_participated || 0;

  const revTypeColors = {
    endorsement: { bg: 'bg-emerald-50 dark:bg-emerald-900/20', text: 'text-emerald-700 dark:text-emerald-400', border: 'border-emerald-200 dark:border-emerald-800' },
    refinement: { bg: 'bg-blue-50 dark:bg-blue-900/20', text: 'text-blue-700 dark:text-blue-400', border: 'border-blue-200 dark:border-blue-800' },
    alternative: { bg: 'bg-amber-50 dark:bg-amber-900/20', text: 'text-amber-700 dark:text-amber-400', border: 'border-amber-200 dark:border-amber-800' },
  };

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'revisions', label: `Reviews (${revisions.length})` },
    { id: 'flow', label: 'Flow' },
  ];

  return (
    <div style={{ fontFamily: FONT }} className="space-y-4">
      {/* Header */}
      <div className="bg-white dark:bg-[#1c1c1e] rounded-xl border border-[#e5e5ea] dark:border-white/10 shadow-sm p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Users size={16} className="text-emerald-600" />
            <span className="text-sm font-semibold text-[#1d1d1f] dark:text-white">Collaborative Synthesis</span>
          </div>
          <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800">
            {modelsParticipated} models
          </span>
        </div>

        {/* Metrics row */}
        <div className="grid grid-cols-3 gap-3">
          <div className="text-center">
            <p className="text-xs text-[#6e6e73] mb-1">Consensus</p>
            <p className={`text-lg font-semibold ${consensusScore >= 0.7 ? 'text-emerald-600' : consensusScore >= 0.4 ? 'text-amber-600' : 'text-red-600'}`}>
              {(consensusScore * 100).toFixed(0)}%
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-[#6e6e73] mb-1">Improvement</p>
            <p className="text-lg font-semibold text-blue-600">
              +{(improvementDelta * 100).toFixed(0)}%
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-[#6e6e73] mb-1">Draft Score</p>
            <p className="text-lg font-semibold text-[#1d1d1f] dark:text-white">
              {(draftScore * 100).toFixed(0)}%
            </p>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-1 bg-[#f5f5f7] dark:bg-[#1c1c1e] rounded-xl p-1">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 text-xs font-medium py-2 px-3 rounded-lg transition-all ${
              activeTab === tab.id
                ? 'bg-white dark:bg-white/10 text-[#1d1d1f] dark:text-white shadow-sm'
                : 'text-[#6e6e73] hover:text-[#1d1d1f] dark:hover:text-white'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-3">
          {/* Draft toggle */}
          <div className="bg-white dark:bg-[#1c1c1e] rounded-xl border border-[#e5e5ea] dark:border-white/10 shadow-sm overflow-hidden">
            <button
              onClick={() => setShowDraft(v => !v)}
              className="w-full flex items-center justify-between px-4 py-3 hover:bg-[#f5f5f7] dark:hover:bg-white/5 transition-colors"
            >
              <div className="flex items-center gap-2">
                <MessageSquare size={14} className="text-[#6e6e73]" />
                <span className="text-xs font-semibold text-[#1d1d1f] dark:text-white">
                  Initial Draft by {draftModel}
                </span>
              </div>
              {showDraft ? <ChevronUp size={14} className="text-[#86868b]" /> : <ChevronDown size={14} className="text-[#86868b]" />}
            </button>
            {showDraft && (
              <div className="px-4 pb-4 border-t border-[#e5e5ea] dark:border-white/10">
                <p className="text-sm text-[#1d1d1f] dark:text-[#e2e8f0] leading-relaxed mt-3 whitespace-pre-wrap">
                  {draft}
                </p>
              </div>
            )}
          </div>

          {/* Revision summary list */}
          {revisions.map((rev, idx) => {
            const style = revTypeColors[rev.type] || revTypeColors.refinement;
            return (
              <div key={idx} className="flex items-center gap-3 px-4 py-2.5 bg-white dark:bg-[#1c1c1e] rounded-xl border border-[#e5e5ea] dark:border-white/10">
                <div className={`w-2 h-2 rounded-full ${rev.type === 'endorsement' ? 'bg-emerald-500' : rev.type === 'alternative' ? 'bg-amber-500' : 'bg-blue-500'}`} />
                <span className="text-xs font-medium text-[#1d1d1f] dark:text-white flex-1">{rev.model}</span>
                <span className={`text-xs font-medium px-2 py-0.5 rounded-full border ${style.bg} ${style.text} ${style.border}`}>
                  {rev.type}
                </span>
                <span className="text-xs text-[#6e6e73]">{(rev.agreement * 100).toFixed(0)}% agree</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Revisions Tab */}
      {activeTab === 'revisions' && (
        <div className="space-y-3">
          {revisions.map((rev, idx) => {
            const style = revTypeColors[rev.type] || revTypeColors.refinement;
            return (
              <div key={idx} className="bg-white dark:bg-[#1c1c1e] rounded-xl border border-[#e5e5ea] dark:border-white/10 shadow-sm overflow-hidden">
                <button
                  onClick={() => setExpandedRevision(expandedRevision === idx ? null : idx)}
                  className="w-full flex items-center justify-between px-4 py-3 hover:bg-[#f5f5f7] dark:hover:bg-white/5 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <Users size={14} className="text-[#6e6e73]" />
                    <span className="text-xs font-semibold text-[#1d1d1f] dark:text-white">{rev.model}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full border ${style.bg} ${style.text} ${style.border}`}>
                      {rev.type}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-[#1d1d1f] dark:text-white">
                      {(rev.agreement * 100).toFixed(0)}%
                    </span>
                    {expandedRevision === idx
                      ? <ChevronUp size={14} className="text-[#86868b]" />
                      : <ChevronDown size={14} className="text-[#86868b]" />
                    }
                  </div>
                </button>

                {expandedRevision === idx && (
                  <div className="px-4 pb-4 border-t border-[#e5e5ea] dark:border-white/10 pt-3 space-y-3">
                    <p className="text-xs text-[#6e6e73]">{rev.comment}</p>

                    {/* Agreement bar */}
                    <div>
                      <span className="text-xs text-[#6e6e73]">Agreement with draft</span>
                      <ConfidenceBar value={rev.agreement} size="sm" showLabel={false} />
                    </div>

                    {/* Key additions */}
                    {rev.key_additions && rev.key_additions.length > 0 && (
                      <div>
                        <span className="text-xs font-semibold text-[#1d1d1f] dark:text-white mb-1 block">Unique Contributions</span>
                        {rev.key_additions.map((add, i) => (
                          <div key={i} className="flex items-start gap-1.5 mt-1">
                            <Sparkles size={10} className="text-blue-500 mt-0.5 flex-shrink-0" />
                            <p className="text-xs text-[#6e6e73] leading-relaxed">{add}</p>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Output preview */}
                    {rev.output_preview && (
                      <div className="bg-[#f5f5f7] dark:bg-white/5 rounded-lg p-3">
                        <p className="text-xs text-[#6e6e73] leading-relaxed">{rev.output_preview}</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Flow Tab — Visual synthesis pipeline */}
      {activeTab === 'flow' && (
        <div className="bg-white dark:bg-[#1c1c1e] rounded-xl border border-[#e5e5ea] dark:border-white/10 shadow-sm p-4">
          <div className="flex flex-col items-center gap-3">
            {/* Draft node */}
            <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
              <MessageSquare size={14} className="text-blue-600" />
              <span className="text-xs font-semibold text-blue-700 dark:text-blue-400">Draft — {draftModel}</span>
              <span className="text-xs text-blue-500">{(draftScore * 100).toFixed(0)}%</span>
            </div>

            <ArrowRight size={16} className="text-[#aeaeb2] rotate-90" />

            {/* Reviewer nodes */}
            <div className="flex flex-wrap justify-center gap-2">
              {revisions.map((rev, idx) => {
                const style = revTypeColors[rev.type] || revTypeColors.refinement;
                return (
                  <div key={idx} className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border ${style.bg} ${style.border}`}>
                    <Users size={12} className={style.text} />
                    <span className={`text-xs font-medium ${style.text}`}>{rev.model}</span>
                  </div>
                );
              })}
            </div>

            <ArrowRight size={16} className="text-[#aeaeb2] rotate-90" />

            {/* Final synthesis node */}
            <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800">
              <Sparkles size={14} className="text-emerald-600" />
              <span className="text-xs font-semibold text-emerald-700 dark:text-emerald-400">Final Synthesis</span>
              <span className="text-xs text-emerald-500">{(consensusScore * 100).toFixed(0)}% consensus</span>
            </div>
          </div>
        </div>
      )}

      {/* Boundary panel */}
      {boundary && Object.keys(boundary).length > 0 && (
        <BoundaryPanel boundary={boundary} />
      )}
    </div>
  );
}
