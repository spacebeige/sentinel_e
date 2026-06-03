import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Shield, AlertTriangle, ShieldAlert } from 'lucide-react';
import ConfidenceBar from './ConfidenceBar';

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

/**
 * BoundaryPanel — Professional risk assessment card
 * Clean design matching Figma design system. No debug-style output.
 */
export default function BoundaryPanel({ boundary = {} }) {
  const [expanded, setExpanded] = useState(false);

  const severity = boundary.severity_score ?? 0;
  const risk = boundary.risk_level || 'LOW';
  const explanation = boundary.explanation || '';

  // Don't render for trivial risk
  if (severity < 15 && risk === 'LOW') return null;

  const riskConfig = {
    LOW: { icon: Shield, color: '#10b981', bg: '#f0fdf4', border: '#bbf7d0', label: 'Low Risk' },
    MEDIUM: { icon: AlertTriangle, color: '#f59e0b', bg: '#fffbeb', border: '#fde68a', label: 'Moderate Risk' },
    HIGH: { icon: ShieldAlert, color: '#ef4444', bg: '#fef2f2', border: '#fecaca', label: 'High Risk' },
  };
  const config = riskConfig[risk] || riskConfig.LOW;
  const Icon = config.icon;

  return (
    <div className="rounded-2xl border overflow-hidden mt-3" style={{ borderColor: config.border, backgroundColor: config.bg }}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2.5 transition-colors"
        style={{ fontFamily: FONT }}
      >
        <div className="flex items-center gap-2">
          <Icon size={14} style={{ color: config.color }} />
          <span style={{ fontSize: '12px', fontWeight: 600, color: config.color }}>
            Risk Assessment
          </span>
          <span className="px-2 py-0.5 rounded-lg" style={{
            fontSize: '10px', fontWeight: 600,
            color: config.color, backgroundColor: config.color + '15',
          }}>
            {config.label}
          </span>
        </div>
        {expanded ? <ChevronUp size={14} style={{ color: '#aeaeb2' }} /> : <ChevronDown size={14} style={{ color: '#aeaeb2' }} />}
      </button>

      {expanded && (
        <div className="px-4 pb-3 space-y-2.5">
          <div>
            <div className="flex justify-between items-center mb-1">
              <span style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73' }}>Severity</span>
              <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: config.color }}>{Math.round(severity)}%</span>
            </div>
            <div className="h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: config.color + '15' }}>
              <div className="h-full rounded-full transition-all duration-700" style={{ width: `${severity}%`, backgroundColor: config.color }} />
            </div>
          </div>

          {explanation && (
            <p style={{ fontFamily: FONT, fontSize: '12px', color: '#6e6e73', lineHeight: 1.5 }}>
              {explanation}
            </p>
          )}

          {boundary.epistemic_risk !== undefined && (
            <div className="grid grid-cols-2 gap-3 mt-2">
              <div>
                <ConfidenceBar value={boundary.epistemic_risk} label="Epistemic Risk" size="sm" />
              </div>
              <div>
                <ConfidenceBar value={boundary.disagreement_score || 0} label="Disagreement" size="sm" />
              </div>
            </div>
          )}

          {boundary.human_review_required && (
            <div className="flex items-center gap-2 px-3 py-2 rounded-xl" style={{ backgroundColor: '#fef2f2', border: '1px solid #fecaca' }}>
              <ShieldAlert size={12} style={{ color: '#ef4444' }} />
              <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: '#ef4444' }}>
                Human review recommended
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
