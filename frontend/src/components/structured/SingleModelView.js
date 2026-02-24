import React from 'react';

/**
 * SingleModelView — Minimal clean output for single-model responses.
 * 
 * Shows:
 * - Clean formatted response text
 * - Optional model name label
 * - No cross-model analytics, no boundary, no debate metrics
 */

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

export default function SingleModelView({ content, modelName }) {
  if (!content) return null;

  return (
    <div className="rounded-2xl bg-white border border-black/5 p-4 shadow-sm">
      {modelName && (
        <span style={{
          fontFamily: FONT, fontSize: '10px', fontWeight: 600,
          color: '#aeaeb2', textTransform: 'uppercase', letterSpacing: '0.05em',
        }}>
          {modelName}
        </span>
      )}
      <p className="mt-1.5 whitespace-pre-wrap" style={{
        fontFamily: FONT, fontSize: '15px', lineHeight: 1.6, color: '#1d1d1f',
      }}>
        {content}
      </p>
    </div>
  );
}
