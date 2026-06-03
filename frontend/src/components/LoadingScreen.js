/**
 * LoadingScreen.js — Visible Loading State
 * Replaces blank screens during:
 * - Initial hydration
 * - Auth state loading
 * - Session initialization
 */

import React from 'react';
import SigmaIdentity from './SigmaIdentity';

export function LoadingScreen({ message = "Loading..." }) {
  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: '#0f172a',
      color: '#f8fafc',
      fontFamily: "'Inter', system-ui, -apple-system, sans-serif"
    }}>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '24px'
      }}>
        <SigmaIdentity size={58} pulse />
        <div style={{
          width: '128px',
          height: '1px',
          background: 'linear-gradient(90deg, transparent, #38bdf8, transparent)',
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
        <div style={{
          textAlign: 'center'
        }}>
          <p style={{
            fontSize: '16px',
            fontWeight: '500',
            color: '#f8fafc',
            margin: '0 0 8px 0'
          }}>
            {message}
          </p>
          <p style={{
            fontSize: '13px',
            color: '#aeaeb2',
            margin: '0'
          }}>
            Please wait...
          </p>
        </div>
      </div>
    </div>
  );
}

export default LoadingScreen;
