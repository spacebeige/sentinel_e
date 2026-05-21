/**
 * LoadingScreen.js — Visible Loading State
 * Replaces blank screens during:
 * - Initial hydration
 * - Auth state loading
 * - Session initialization
 */

import React from 'react';
import SentinelIdentity from './SentinelIdentity';

export function LoadingScreen({ message = "Booting Runtime...", subtext = "Initializing Cognitive Matrix..." }) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center sentinel-bg-app sentinel-text-primary" style={{ fontFamily: "'Inter', system-ui, -apple-system, sans-serif" }}>
      <div className="flex flex-col items-center gap-6">
        <SentinelIdentity size={58} pulse />
        <div style={{
          width: '128px',
          height: '1px',
          background: 'linear-gradient(90deg, transparent, #3b82f6, transparent)',
          animation: 'runtimeScan 1.6s ease-in-out infinite'
        }}>
          <style>{`
            @keyframes runtimeScan {
              0%, 100% { opacity: 0.25; transform: scaleX(0.6); }
              50% { opacity: 1; transform: scaleX(1); }
            }
          `}</style>
        </div>
        
        {/* Message */}
        <div className="text-center">
          <p className="text-base font-semibold mb-2">
            {message}
          </p>
          <p className="text-sm sentinel-text-muted">
            {subtext}
          </p>
        </div>
      </div>
    </div>
  );
}

export default LoadingScreen;
