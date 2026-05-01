/**
 * LoadingScreen.js — Visible Loading State
 * Replaces blank screens during:
 * - Initial hydration
 * - Auth state loading
 * - Session initialization
 */

import React from 'react';

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
        {/* Spinner */}
        <div style={{
          width: '48px',
          height: '48px',
          border: '4px solid #38bdf820',
          borderTop: '4px solid #38bdf8',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite'
        }}>
          <style>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
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
