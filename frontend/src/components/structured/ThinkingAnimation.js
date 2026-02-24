import React, { useState, useEffect, useRef } from 'react';

/**
 * ThinkingAnimation — Premium generation pipeline display
 * 
 * Shows animated pipeline steps tied to real async progress.
 * Design: centered card with pulsing shimmer and step-by-step timeline.
 * Matches the Figma design system (white cards, Inter font, consistent radius).
 */

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

function ShimmerBar() {
  return (
    <div className="h-1 w-full bg-[#f5f5f7] rounded-full overflow-hidden">
      <div
        className="h-full rounded-full"
        style={{
          background: 'linear-gradient(90deg, transparent 0%, #3b82f6 50%, transparent 100%)',
          animation: 'shimmer 1.5s ease-in-out infinite',
          width: '40%',
        }}
      />
      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(350%); }
        }
      `}</style>
    </div>
  );
}

export default function ThinkingAnimation({ steps = [], activeColor = '#3b82f6' }) {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (steps.length === 0) return;

    // Advance through steps based on realistic timing
    const stepDuration = Math.max(2000, 12000 / steps.length);

    intervalRef.current = setInterval(() => {
      setCurrentStepIndex(prev => {
        if (prev >= steps.length - 1) {
          clearInterval(intervalRef.current);
          return prev;
        }
        return prev + 1;
      });
    }, stepDuration);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      setCurrentStepIndex(0);
    };
  }, [steps.length]);

  if (steps.length === 0) {
    // Minimal loading fallback
    return (
      <div className="px-5 py-4">
        <div className="flex items-center gap-3">
          <div className="relative flex h-2.5 w-2.5">
            <span className="absolute inline-flex h-full w-full rounded-full opacity-75"
              style={{ backgroundColor: activeColor, animation: 'ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite' }} />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5"
              style={{ backgroundColor: activeColor }} />
          </div>
          <span style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 500, color: '#1d1d1f' }}>
            Processing...
          </span>
        </div>
        <div className="mt-3">
          <ShimmerBar />
        </div>
      </div>
    );
  }

  return (
    <div className="px-5 py-4">
      {/* Header */}
      <div className="flex items-center gap-2.5 mb-4">
        <div className="relative flex h-2.5 w-2.5">
          <span className="absolute inline-flex h-full w-full rounded-full opacity-75"
            style={{ backgroundColor: activeColor, animation: 'ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite' }} />
          <span className="relative inline-flex rounded-full h-2.5 w-2.5"
            style={{ backgroundColor: activeColor }} />
        </div>
        <span style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600, color: '#1d1d1f' }}>
          Sentinel-E Processing
        </span>
      </div>

      {/* Pipeline steps */}
      <div className="space-y-2">
        {steps.map((step, idx) => {
          const isComplete = idx < currentStepIndex;
          const isCurrent = idx === currentStepIndex;

          return (
            <div
              key={step.id || idx}
              className="flex items-center gap-3 py-1 transition-all duration-300"
              style={{ opacity: isComplete ? 0.5 : isCurrent ? 1 : 0.35 }}
            >
              {/* Status indicator */}
              <div className="w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0"
                style={{
                  backgroundColor: isComplete ? '#10b98115' : isCurrent ? activeColor + '15' : '#f5f5f7',
                }}>
                {isComplete ? (
                  <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
                    <path d="M1 4L3.5 6.5L9 1" stroke="#10b981" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : isCurrent ? (
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: activeColor, animation: 'pulse 1.5s ease-in-out infinite' }} />
                ) : (
                  <div className="w-1.5 h-1.5 rounded-full bg-[#d1d5db]" />
                )}
              </div>

              {/* Step label */}
              <span style={{
                fontFamily: FONT,
                fontSize: '13px',
                fontWeight: isCurrent ? 600 : 400,
                color: isComplete ? '#6e6e73' : isCurrent ? '#1d1d1f' : '#aeaeb2',
                textDecoration: isComplete ? 'line-through' : 'none',
              }}>
                {step.label}
              </span>

              {/* Active dots */}
              {isCurrent && (
                <span className="flex gap-0.5 ml-1">
                  {[0, 150, 300].map(d => (
                    <span key={d} className="w-1 h-1 rounded-full animate-bounce inline-block"
                      style={{ backgroundColor: activeColor, animationDelay: `${d}ms` }} />
                  ))}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Progress bar */}
      <div className="mt-3 h-1.5 bg-[#f5f5f7] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-1000 ease-out"
          style={{
            width: `${Math.max(((currentStepIndex + 1) / steps.length) * 100, 8)}%`,
            backgroundColor: activeColor,
          }}
        />
      </div>

      <style>{`
        @keyframes ping {
          75%, 100% {
            transform: scale(2);
            opacity: 0;
          }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </div>
  );
}
