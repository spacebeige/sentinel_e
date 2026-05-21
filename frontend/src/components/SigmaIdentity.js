import React from 'react';
import { Sigma } from 'lucide-react';

export default function SigmaIdentity({
  size = 40,
  label = 'Sentinel-E',
  showLabel = false,
  pulse = false,
  className = '',
}) {
  return (
    <div className={`sentinel-sigma-identity ${pulse ? 'sentinel-sigma-identity--pulse' : ''} ${className}`}>
      <div
        className="sentinel-sigma-mark"
        style={{ width: size, height: size, borderRadius: Math.max(12, size * 0.28) }}
        aria-hidden="true"
      >
        <Sigma style={{ width: size * 0.54, height: size * 0.54 }} />
      </div>
      {showLabel && (
        <div className="sentinel-sigma-label">
          <span>{label}</span>
          <small>cognitive runtime</small>
        </div>
      )}
    </div>
  );
}
