import React from 'react';

export default function SentinelIdentity({
  size = 36,
  label = 'Sentinel-E',
  showLabel = false,
  pulse = false,
  className = '',
  rounded = '20px',
}) {
  return (
    <div className={`sentinel-identity ${pulse ? 'sentinel-identity--pulse' : ''} ${className}`}>
      <div
        className="sentinel-mark"
        style={{
          width: size,
          height: size,
          borderRadius: rounded,
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'rgba(255,255,255,0.72)',
          border: '1px solid rgba(255,255,255,0.12)',
          backdropFilter: 'blur(20px)',
          boxShadow: '0 8px 24px rgba(0,0,0,0.08)',
        }}
        aria-hidden="true"
      >
        <img
          src="/assets/branding/sentinel_e.png"
          alt="Sentinel-E"
          draggable={false}
          className="sentinel-logo"
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            userSelect: 'none',
          }}
        />
      </div>
      {showLabel && (
        <div className="sentinel-label ml-3">
          <span style={{ color: 'var(--text-primary)', fontSize: '14px', fontWeight: 700, letterSpacing: '-0.02em' }}>{label}</span>
          <small style={{ color: 'var(--text-tertiary)', fontSize: '9px', fontWeight: 700, letterSpacing: '0.12em', marginTop: '4px', textTransform: 'uppercase', display: 'block' }}>cognitive runtime</small>
        </div>
      )}
    </div>
  );
}