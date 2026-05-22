import React from 'react';

const LOGO_SRC = '/assets/branding/sentinel_e.png';

export default function SentinelIdentity({
  size = 40,
  label = 'Sentinel-E',
  showLabel = false,
  pulse = false,
  className = '',
}) {
  return (
    <div className={`sentinel-identity ${pulse ? 'sentinel-identity--pulse' : ''} ${className}`}>
      <div
        className="sentinel-mark"
        style={{ width: size, height: size, borderRadius: Math.max(12, size * 0.28) }}
        aria-hidden="true"
      >
        <img
          src={LOGO_SRC}
          alt="Sentinel-E"
          className="sentinel-logo"
          style={{ width: size * 0.64, height: size * 0.64 }}
        />
      </div>
      {showLabel && (
        <div className="sentinel-label">
          <span>{label}</span>
          <small>cognitive runtime</small>
        </div>
      )}
    </div>
  );
}
