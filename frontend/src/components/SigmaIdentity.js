import React from 'react';

const LOGO_SRC = '/assets/branding/sentinel-logo-icon.png';

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
        <img
          src={LOGO_SRC}
          alt="Sentinel-E"
          className="sentinel-sigma-logo"
          style={{ width: size * 0.64, height: size * 0.64 }}
        />
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
