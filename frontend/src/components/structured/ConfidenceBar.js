import React from 'react';

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

/**
 * ConfidenceBar — Visual confidence indicator
 * Professional design — matches Figma design system.
 */
export default function ConfidenceBar({ value = 0, label = '', size = 'md', showLabel = true }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);

  const getColor = (v) => {
    if (v >= 0.7) return '#10b981';
    if (v >= 0.4) return '#f59e0b';
    return '#ef4444';
  };

  const heights = { sm: 'h-1.5', md: 'h-2', lg: 'h-3' };
  const color = getColor(value);

  return (
    <div className="w-full">
      {showLabel && label && (
        <div className="flex justify-between items-center mb-1">
          <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 500, color: '#6e6e73' }}>{label}</span>
          <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color }}>{pct}%</span>
        </div>
      )}
      <div className={`w-full bg-black/5 rounded-full ${heights[size] || heights.md} overflow-hidden`}>
        <div
          className={`${heights[size] || heights.md} rounded-full transition-all duration-700 ease-out`}
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}
