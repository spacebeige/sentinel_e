import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ConfidenceBar from './ConfidenceBar';

/**
 * StandardView — Clean conversational output for Standard mode
 * 
 * Shows:
 * - Markdown-rendered synthesis (primary output)
 * - Qualitative confidence + agreement indicators
 * - Expandable individual model outputs
 * - Risk assessment card (when applicable)
 */

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

/** Qualitative label for a 0-1 value */
function qualityLabel(value) {
  if (value == null) return null;
  const pct = Math.round(value * 100);
  if (value >= 0.85) return { pct, text: 'High', color: '#10b981' };
  if (value >= 0.65) return { pct, text: 'Moderate', color: '#3b82f6' };
  if (value >= 0.40) return { pct, text: 'Low', color: '#f59e0b' };
  return { pct, text: 'Very Low', color: '#ef4444' };
}

/** Markdown component config — shared inline rendering */
const markdownComponents = {
  p: ({ children }) => (
    <p style={{ fontFamily: FONT, fontSize: '14px', lineHeight: 1.7, marginTop: '6px', marginBottom: '6px', color: '#1d1d1f' }} className="dark:text-[#e2e8f0]">{children}</p>
  ),
  h1: ({ children }) => <h1 style={{ fontFamily: FONT, fontSize: '18px', fontWeight: 700, marginTop: '14px', marginBottom: '6px', color: '#1d1d1f' }} className="dark:text-white">{children}</h1>,
  h2: ({ children }) => <h2 style={{ fontFamily: FONT, fontSize: '16px', fontWeight: 700, marginTop: '12px', marginBottom: '4px', color: '#1d1d1f' }} className="dark:text-white">{children}</h2>,
  h3: ({ children }) => <h3 style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 700, marginTop: '10px', marginBottom: '4px', color: '#1d1d1f' }} className="dark:text-white">{children}</h3>,
  ul: ({ children }) => <ul style={{ paddingLeft: '18px', marginTop: '4px', marginBottom: '4px', listStyleType: 'disc' }}>{children}</ul>,
  ol: ({ children }) => <ol style={{ paddingLeft: '18px', marginTop: '4px', marginBottom: '4px', listStyleType: 'decimal' }}>{children}</ol>,
  li: ({ children }) => <li style={{ fontFamily: FONT, fontSize: '14px', lineHeight: 1.6, marginBottom: '2px', color: '#1d1d1f' }} className="dark:text-[#e2e8f0]">{children}</li>,
  strong: ({ children }) => <strong style={{ fontWeight: 700, color: 'inherit' }}>{children}</strong>,
  em: ({ children }) => <em style={{ fontStyle: 'italic', color: 'inherit' }}>{children}</em>,
  blockquote: ({ children }) => (
    <blockquote style={{ borderLeft: '3px solid #d1d5db', paddingLeft: '12px', margin: '8px 0', color: '#6e6e73', fontStyle: 'italic' }}>{children}</blockquote>
  ),
  code: ({ inline, className, children }) => {
    if (inline) {
      return <code style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', backgroundColor: 'rgba(0,0,0,0.05)', padding: '1px 4px', borderRadius: '3px' }}>{children}</code>;
    }
    return (
      <pre className="my-2 p-3 rounded-lg bg-[#f5f5f7] dark:bg-[#1a1a1e] overflow-x-auto" style={{ margin: 0 }}>
        <code style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', lineHeight: 1.5 }} className="text-[#1d1d1f] dark:text-[#e2e8f0]">{children}</code>
      </pre>
    );
  },
  pre: ({ children }) => <>{children}</>,
  a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer" style={{ color: '#3b82f6', textDecoration: 'underline' }}>{children}</a>,
};

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
  const confQ = qualityLabel(confidence);
  const agrQ = disagreementScore != null ? qualityLabel(1 - disagreementScore) : null;

  return (
    <div className="space-y-3">
      {/* Synthesis section — primary output, rendered as markdown */}
      {synthesis && (
        <div className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5 p-4 shadow-sm">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
            {synthesis}
          </ReactMarkdown>
        </div>
      )}

      {/* Confidence + Agreement — qualitative labels */}
      {hasMultipleModels && confQ && (
        <div className={`grid ${agrQ ? 'grid-cols-2' : 'grid-cols-1'} gap-3`}>
          <div className="rounded-2xl bg-[#f5f5f7] dark:bg-[#1c1c1e] p-3">
            <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#6e6e73', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Confidence
            </span>
            <div className="mt-2">
              <ConfidenceBar value={confidence} label="" showLabel={false} size="sm" />
              <div className="flex justify-between mt-1">
                <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 500, color: confQ.color }}>
                  {confQ.text}
                </span>
                <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: '#1d1d1f' }} className="dark:text-white">
                  {confQ.pct}%
                </span>
              </div>
            </div>
          </div>
          {agrQ && (
            <div className="rounded-2xl bg-[#f5f5f7] dark:bg-[#1c1c1e] p-3">
              <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#6e6e73', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Model Agreement
              </span>
              <div className="mt-2">
                <ConfidenceBar value={1 - disagreementScore} label="" showLabel={false} size="sm" />
                <div className="flex justify-between mt-1">
                  <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 500, color: agrQ.color }}>
                    {agrQ.text}
                  </span>
                  <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: '#1d1d1f' }} className="dark:text-white">
                    {agrQ.pct}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Model outputs — expandable only if multi-model */}
      {hasMultipleModels && models.length > 0 && (
        <details className="group">
          <summary className="cursor-pointer flex items-center gap-2 px-1 py-1.5 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
            <span className="transform group-open:rotate-90 transition-transform text-xs text-[#aeaeb2]">▶</span>
            <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: '#6e6e73', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Individual Model Outputs ({models.length})
            </span>
          </summary>
          <div className="mt-2 space-y-2">
            {models.filter(m => !m.error).map((model, idx) => (
              <div key={idx} className="rounded-xl border border-black/5 dark:border-white/5 bg-[#f5f5f7] dark:bg-[#1c1c1e] p-3">
                <div className="flex items-center justify-between mb-1.5">
                  <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: '#1d1d1f' }} className="dark:text-white">
                    {model.model_name}
                  </span>
                  {model.latency_ms > 0 && (
                    <span style={{ fontFamily: FONT, fontSize: '10px', color: '#aeaeb2' }}>
                      {model.latency_ms.toFixed(0)}ms
                    </span>
                  )}
                </div>
                <p className="whitespace-pre-wrap" style={{ fontFamily: FONT, fontSize: '13px', lineHeight: 1.6, color: '#3b3b3f' }} >
                  {(model.output || '').slice(0, 800)}
                  {(model.output || '').length > 800 && '...'}
                </p>
              </div>
            ))}
            {/* Show failed models separately */}
            {models.filter(m => m.error).length > 0 && (
              <div className="rounded-xl border border-[#fecaca] bg-[#fef2f2] dark:bg-[#991b1b]/20 dark:border-[#991b1b] p-3">
                <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#ef4444', textTransform: 'uppercase' }}>
                  Failed Models
                </span>
                {models.filter(m => m.error).map((model, idx) => (
                  <p key={idx} style={{ fontFamily: FONT, fontSize: '11px', color: '#991b1b', marginTop: '4px' }}>
                    {model.model_name} — {model.error || 'No response received'}
                  </p>
                ))}
              </div>
            )}
          </div>
        </details>
      )}

      {/* Boundary / Risk — professional card */}
      <RiskCard boundary={boundary} />
    </div>
  );
}
