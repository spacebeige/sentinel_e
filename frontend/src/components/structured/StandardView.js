import React from 'react';
import ConfidenceBar from './ConfidenceBar';

/**
 * StandardView — Clean minimal output for Standard mode
 * 
 * Shows:
 * - Clean formatted response text
 * - Optional: confidence indicator (only if meaningful)
 * - No debate rounds, no evidence pipeline, no glass audit
 * - For simple tasks (code, explanation): minimal chrome
 */

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

function RiskCard({ boundary }) {
  if (!boundary || !boundary.risk_level) return null;
  // Only show for non-trivial risk
  if (boundary.severity_score != null && boundary.severity_score < 20) return null;

  const riskConfig = {
    LOW: { color: '#10b981', bg: '#f0fdf4', label: 'Low Risk' },
    MEDIUM: { color: '#f59e0b', bg: '#fffbeb', label: 'Moderate Risk' },
    HIGH: { color: '#ef4444', bg: '#fef2f2', label: 'High Risk' },
    CRITICAL: { color: '#ef4444', bg: '#fef2f2', label: 'Critical Risk' },
  };
  const config = riskConfig[boundary.risk_level] || riskConfig.LOW;
  const severity = boundary.severity_score != null ? Math.round(boundary.severity_score) : null;

  return (
    <div className="rounded-2xl border overflow-hidden" style={{ borderColor: config.color + '30', backgroundColor: config.bg }}>
      <div className="px-4 py-3">
        <div className="flex items-center justify-between mb-2">
          <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: config.color, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Risk Assessment
          </span>
          <span className="px-2 py-0.5 rounded-lg" style={{
            fontFamily: FONT, fontSize: '11px', fontWeight: 600,
            color: config.color, backgroundColor: config.color + '15',
          }}>
            {config.label}
          </span>
        </div>
        {severity != null && (
          <div className="mb-2">
            <div className="flex justify-between items-center mb-1">
              <span style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73' }}>Severity</span>
              <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: config.color }}>{severity}%</span>
            </div>
            <div className="h-1.5 bg-black/5 rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-all duration-700" style={{ width: `${severity}%`, backgroundColor: config.color }} />
            </div>
          </div>
        )}
        {boundary.explanation && (
          <p style={{ fontFamily: FONT, fontSize: '12px', color: '#6e6e73', lineHeight: 1.5 }}>
            {boundary.explanation}
          </p>
        )}
      </div>
    </div>
  );
}

export default function StandardView({ data, boundary, confidence, disagreementScore }) {
  if (!data) return null;

  const models = data.model_outputs || [];
  const synthesis = data.synthesis || '';
  const hasMultipleModels = models.length > 1;

  return (
    <div className="space-y-3">
      {/* Synthesis section — primary output */}
      {synthesis && (
        <div className="rounded-2xl bg-white border border-black/5 p-4 shadow-sm">
          <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#3b82f6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Aggregated Analysis
          </span>
          <p className="mt-2 whitespace-pre-wrap" style={{ fontFamily: FONT, fontSize: '14px', lineHeight: 1.6, color: '#1d1d1f' }}>
            {synthesis}
          </p>
        </div>
      )}

      {/* Confidence + Disagreement — only when multi-model */}
      {hasMultipleModels && confidence != null && (
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-2xl bg-[#f5f5f7] p-3">
            <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#6e6e73', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Confidence
            </span>
            <div className="mt-2">
              <ConfidenceBar value={confidence} label="" showLabel={false} size="sm" />
              <span className="block mt-1 text-right" style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: '#1d1d1f' }}>
                {Math.round(confidence * 100)}%
              </span>
            </div>
          </div>
          {disagreementScore != null && disagreementScore > 0 && (
            <div className="rounded-2xl bg-[#f5f5f7] p-3">
              <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#6e6e73', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Model Agreement
              </span>
              <div className="mt-2">
                <ConfidenceBar value={1 - disagreementScore} label="" showLabel={false} size="sm" />
                <span className="block mt-1 text-right" style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: '#1d1d1f' }}>
                  {Math.round((1 - disagreementScore) * 100)}%
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Model outputs — expandable only if multi-model */}
      {hasMultipleModels && models.length > 0 && (
        <details className="group">
          <summary className="cursor-pointer flex items-center gap-2 px-1 py-1.5 rounded-lg hover:bg-black/5 transition-colors">
            <span className="transform group-open:rotate-90 transition-transform text-xs text-[#aeaeb2]">▶</span>
            <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: '#6e6e73', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Individual Model Outputs ({models.length})
            </span>
          </summary>
          <div className="mt-2 space-y-2">
            {models.filter(m => !m.error).map((model, idx) => (
              <div key={idx} className="rounded-xl border border-black/5 bg-[#f5f5f7] p-3">
                <div className="flex items-center justify-between mb-1.5">
                  <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: '#1d1d1f' }}>
                    {model.model_name}
                  </span>
                  {model.latency_ms > 0 && (
                    <span style={{ fontFamily: FONT, fontSize: '10px', color: '#aeaeb2' }}>
                      {model.latency_ms.toFixed(0)}ms
                    </span>
                  )}
                </div>
                <p className="whitespace-pre-wrap" style={{ fontFamily: FONT, fontSize: '13px', lineHeight: 1.5, color: '#3b3b3f' }}>
                  {(model.output || '').slice(0, 800)}
                  {(model.output || '').length > 800 && '...'}
                </p>
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Boundary / Risk — professional card */}
      <RiskCard boundary={boundary} />
    </div>
  );
}
