import React, { useState } from 'react';
import { ChevronDown, ChevronUp, CheckCircle, AlertTriangle } from 'lucide-react';
import ConfidenceBar from './ConfidenceBar';
import BoundaryPanel from './BoundaryPanel';

const FONT = 'Inter, system-ui, -apple-system, sans-serif';

/**
 * AggregationView — Parallel aggregation output (professional light-theme)
 *
 * Shows: synthesized result, per-model outputs, divergence metrics, agreement/disagreement
 * Does NOT show: debate rounds, forensic claims, audit metrics
 */
export default function AggregationView({ data, boundary, confidence }) {
  const [activeTab, setActiveTab] = useState('synthesis');
  const [expandedModel, setExpandedModel] = useState(null);

  if (!data) return null;

  const models = data.model_outputs || [];
  const synthesis = data.synthesis || '';
  const divergence = data.divergence_score || 0;
  const agreements = data.agreement_points || [];
  const disagreements = data.disagreement_details || [];
  const confPerModel = data.confidence_per_model || {};
  const succeeded = data.models_succeeded || models.length;
  const failed = data.models_failed || 0;

  const tabs = [
    { id: 'synthesis', label: 'Synthesis' },
    { id: 'models', label: `Models (${models.length})` },
    { id: 'analysis', label: 'Analysis' },
  ];

  return (
    <div style={{ fontFamily: FONT }} className="space-y-4">
      {/* Header Card */}
      <div className="bg-white rounded-xl border border-[#e5e5ea] shadow-sm p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-[#1d1d1f]">Aggregated Analysis</span>
            <span className="text-xs text-[#86868b]">
              {succeeded}/{succeeded + failed} models
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#6e6e73]">Confidence</span>
            <span className="text-sm font-semibold text-[#1d1d1f]">
              {((confidence || data.confidence_aggregation || 0.5) * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        <ConfidenceBar value={confidence || data.confidence_aggregation || 0.5} label="" size="sm" showLabel={false} />
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

      {/* Synthesis Tab */}
      {activeTab === 'synthesis' && (
        <div className="bg-white rounded-xl border border-[#e5e5ea] shadow-sm p-4 space-y-3">
          <div className="text-sm text-[#1d1d1f] leading-relaxed whitespace-pre-wrap">
            {synthesis}
          </div>
          <ConfidenceBar value={confidence || 0.5} label="Aggregated Confidence" />
        </div>
      )}

      {/* Models Tab */}
      {activeTab === 'models' && (
        <div className="space-y-2">
          {models.map((model, idx) => (
            <div key={idx} className="bg-white rounded-xl border border-[#e5e5ea] shadow-sm overflow-hidden">
              <button
                onClick={() => setExpandedModel(expandedModel === idx ? null : idx)}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-[#f5f5f7]/50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${model.error ? 'bg-red-500' : 'bg-emerald-500'}`} />
                  <span className="text-xs font-semibold text-[#1d1d1f]">{model.model_name}</span>
                  {model.latency_ms > 0 && (
                    <span className="text-xs text-[#86868b]">{model.latency_ms.toFixed(0)}ms</span>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <span className={`text-xs font-semibold ${
                    (confPerModel[model.model_id] || model.confidence || 0.5) >= 0.7 ? 'text-emerald-600' :
                    (confPerModel[model.model_id] || model.confidence || 0.5) >= 0.4 ? 'text-amber-600' : 'text-red-600'
                  }`}>
                    {((confPerModel[model.model_id] || model.confidence || 0.5) * 100).toFixed(0)}%
                  </span>
                  {expandedModel === idx
                    ? <ChevronUp size={14} className="text-[#86868b]" />
                    : <ChevronDown size={14} className="text-[#86868b]" />
                  }
                </div>
              </button>

              {expandedModel === idx && (
                <div className="px-4 pb-4 border-t border-[#e5e5ea]">
                  {model.error ? (
                    <p className="text-xs text-red-600 mt-3">{model.error}</p>
                  ) : (
                    <>
                      <p className="text-xs text-[#1d1d1f] mt-3 leading-relaxed whitespace-pre-wrap">
                        {model.output}
                      </p>
                      {model.claims && model.claims.length > 0 && (
                        <div className="mt-3 pt-2 border-t border-[#f5f5f7]">
                          <p className="text-xs text-[#6e6e73] font-semibold mb-1.5">
                            Claims ({model.claims.length})
                          </p>
                          {model.claims.slice(0, 5).map((claim, ci) => (
                            <div key={ci} className="flex items-start gap-2 text-xs text-[#6e6e73] mb-1">
                              <span className="text-[#86868b] shrink-0">•</span>
                              <span>{claim.text}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Analysis Tab */}
      {activeTab === 'analysis' && (
        <div className="space-y-3">
          {/* Divergence */}
          <div className="bg-white rounded-xl border border-[#e5e5ea] shadow-sm p-4">
            <p className="text-xs font-semibold text-[#1d1d1f] mb-2">Cross-Model Divergence</p>
            <ConfidenceBar value={divergence} label="" showLabel={false} />
            <p className="text-xs text-[#6e6e73] mt-1">
              {divergence < 0.3 ? 'Models are largely in agreement' :
               divergence < 0.6 ? 'Moderate disagreement between models' :
               'Significant divergence — review individual model outputs'}
            </p>
          </div>

          {/* Agreements */}
          {agreements.length > 0 && (
            <div className="bg-white rounded-xl border border-[#e5e5ea] shadow-sm p-4">
              <div className="flex items-center gap-1.5 mb-2">
                <CheckCircle size={14} className="text-emerald-600" />
                <p className="text-xs font-semibold text-[#1d1d1f]">Agreement Points</p>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {agreements.slice(0, 15).map((word, i) => (
                  <span key={i} className="text-xs bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 rounded-full">
                    {word}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Disagreements */}
          {disagreements.length > 0 && (
            <div className="bg-white rounded-xl border border-[#e5e5ea] shadow-sm p-4">
              <div className="flex items-center gap-1.5 mb-2">
                <AlertTriangle size={14} className="text-amber-600" />
                <p className="text-xs font-semibold text-[#1d1d1f]">Disagreements</p>
              </div>
              {disagreements.map((d, i) => (
                <div key={i} className="text-xs text-[#6e6e73] mb-1.5 flex items-start gap-1.5">
                  <span className="text-[#86868b] shrink-0">•</span>
                  <span>{d.detail || `${d.model_a} vs ${d.model_b}: ${d.type}`}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <BoundaryPanel boundary={boundary} />
    </div>
  );
}
