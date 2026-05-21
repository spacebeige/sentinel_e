/**
 * ============================================================
 * CognitionStreamPanel — Live Orchestration Cognition UI
 * ============================================================
 * Sentinel-E v8.0 — Persistent Hybrid Cognitive Runtime
 *
 * Replaces generic "thinking…" with live cognitive phase labels
 * derived from the OrchestrationRun state machine.
 *
 * Props:
 *   isLoading     — boolean: show/hide the panel
 *   currentResult — last API response (reads omega_metadata.orchestration_run)
 *   mode          — current mode string for contextual labels
 *   subMode       — sub-mode string
 *
 * Architecture:
 *   - Reads orchestration_run from omega_metadata (already in response)
 *   - Cycles through cognitive phase labels during loading
 *   - Shows real phase when available, animated labels when not
 *   - Fully additive: does not replace any existing component
 * ============================================================
 */

import React, { useState, useEffect, useRef, useMemo } from 'react';

// ── Cognitive Phase Labels (mirrors backend CognitivePhase) ──
const PHASE_SEQUENCE = [
  { phase: 'observe',          label: 'Observing',           icon: '🔍', detail: 'Parsing semantic intent' },
  { phase: 'analyze',          label: 'Analyzing',           icon: '🧠', detail: 'Routing decision in progress' },
  { phase: 'route',            label: 'Routing',             icon: '🗺️', detail: 'Selecting execution path' },
  { phase: 'retrieve_memory',  label: 'Memory Retrieval',    icon: '💾', detail: 'Injecting cognitive context' },
  { phase: 'spawn_agents',     label: 'Spawning Agents',     icon: '⚡', detail: 'Parallel model execution' },
  { phase: 'debate',           label: 'Debating',            icon: '⚔️', detail: 'Multi-model reasoning active' },
  { phase: 'verify',           label: 'Verifying',           icon: '✅', detail: 'Contradiction analysis' },
  { phase: 'synthesize',       label: 'Synthesizing',        icon: '🔗', detail: 'Ensemble convergence' },
  { phase: 'reflect',          label: 'Reflecting',          icon: '🪞', detail: 'Metacognitive analysis' },
  { phase: 'store_snapshot',   label: 'Storing',             icon: '📦', detail: 'Persisting cognitive state' },
];

const MODE_PHASE_FILTERS = {
  single_model: ['observe', 'analyze', 'route', 'spawn_agents', 'synthesize'],
  standard:     ['observe', 'analyze', 'route', 'retrieve_memory', 'spawn_agents', 'synthesize'],
  experimental: PHASE_SEQUENCE.map(p => p.phase),
  debate:       PHASE_SEQUENCE.map(p => p.phase),
  ensemble:     PHASE_SEQUENCE.map(p => p.phase),
};

// ── Styling constants ─────────────────────────────────────────
const PANEL_BASE = {
  display: 'flex',
  flexDirection: 'column',
  gap: '12px',
  padding: '16px 20px',
  borderRadius: '16px',
  border: '1px solid rgba(59, 130, 246, 0.25)',
  background: 'linear-gradient(135deg, rgba(59,130,246,0.06) 0%, rgba(6,182,212,0.04) 100%)',
  backdropFilter: 'blur(12px)',
  position: 'relative',
  overflow: 'hidden',
};

const PULSE_KEYFRAMES = `
  @keyframes cogPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.97); }
  }
  @keyframes cogScan {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(200%); }
  }
  @keyframes cogDot {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
  }
`;

export default function CognitionStreamPanel({
  isLoading = false,
  currentResult = null,
  mode = 'standard',
  subMode = null,
}) {
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);
  const startTimeRef = useRef(null);
  const intervalRef = useRef(null);
  const elapsedRef = useRef(null);

  // Get real orchestration run from last result
  const orchRun = useMemo(() => {
    if (!currentResult) return null;
    return currentResult?.omega_metadata?.orchestration_run || null;
  }, [currentResult]);

  // Determine phases to show for current mode
  const activePhases = useMemo(() => {
    const key = subMode === 'debate' ? 'debate' : mode;
    const filter = MODE_PHASE_FILTERS[key] || MODE_PHASE_FILTERS.standard;
    return PHASE_SEQUENCE.filter(p => filter.includes(p.phase));
  }, [mode, subMode]);

  // Cycle through phases during loading
  useEffect(() => {
    if (!isLoading) {
      setPhaseIndex(0);
      setElapsedMs(0);
      startTimeRef.current = null;
      clearInterval(intervalRef.current);
      clearInterval(elapsedRef.current);
      return;
    }

    startTimeRef.current = Date.now();

    // Phase cycling — advances every ~2.8s
    intervalRef.current = setInterval(() => {
      setPhaseIndex(prev => (prev + 1) % activePhases.length);
    }, 2800);

    // Elapsed timer
    elapsedRef.current = setInterval(() => {
      setElapsedMs(Date.now() - (startTimeRef.current || Date.now()));
    }, 100);

    return () => {
      clearInterval(intervalRef.current);
      clearInterval(elapsedRef.current);
    };
  }, [isLoading, activePhases.length]);

  if (!isLoading) return null;

  const currentPhase = activePhases[phaseIndex] || activePhases[0];
  const elapsedSec = (elapsedMs / 1000).toFixed(1);

  return (
    <>
      <style>{PULSE_KEYFRAMES}</style>
      <div style={PANEL_BASE} id="cognition-stream-panel">

        {/* Scanning light effect */}
        <div style={{
          position: 'absolute',
          top: 0, left: 0, bottom: 0,
          width: '40%',
          background: 'linear-gradient(90deg, transparent, rgba(59,130,246,0.08), transparent)',
          animation: 'cogScan 2.5s ease-in-out infinite',
          pointerEvents: 'none',
        }} />

        {/* Header row */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {/* Sigma indicator */}
            <div style={{
              width: '28px', height: '28px',
              borderRadius: '8px',
              background: 'linear-gradient(135deg, #3b82f6, #06b6d4)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '14px', fontWeight: 700, color: 'white',
              animation: 'cogPulse 2s ease-in-out infinite',
              flexShrink: 0,
            }}>
              Σ
            </div>

            <div>
              <div style={{
                fontSize: '11px',
                fontWeight: 600,
                letterSpacing: '0.08em',
                color: 'rgba(59,130,246,0.9)',
                textTransform: 'uppercase',
              }}>
                Cognitive Runtime
              </div>
              <div style={{
                fontSize: '13px',
                fontWeight: 600,
                color: 'var(--text-primary, #1d1d1f)',
                marginTop: '1px',
              }}>
                {currentPhase.icon} {currentPhase.label}
              </div>
            </div>
          </div>

          {/* Elapsed time */}
          <div style={{
            fontSize: '11px',
            color: 'rgba(110,110,115,0.8)',
            fontVariantNumeric: 'tabular-nums',
            minWidth: '40px',
            textAlign: 'right',
          }}>
            {elapsedSec}s
          </div>
        </div>

        {/* Phase detail line */}
        <div style={{
          fontSize: '12px',
          color: 'rgba(110,110,115,0.9)',
          paddingLeft: '38px',
          animation: 'cogPulse 2.8s ease-in-out infinite',
        }}>
          {currentPhase.detail}
        </div>

        {/* Phase progress dots */}
        <div style={{
          display: 'flex',
          gap: '5px',
          paddingLeft: '38px',
          alignItems: 'center',
        }}>
          {activePhases.map((p, idx) => (
            <div
              key={p.phase}
              style={{
                width: idx === phaseIndex ? '18px' : '6px',
                height: '6px',
                borderRadius: '3px',
                background: idx === phaseIndex
                  ? 'linear-gradient(90deg, #3b82f6, #06b6d4)'
                  : idx < phaseIndex
                    ? 'rgba(59,130,246,0.4)'
                    : 'rgba(174,174,178,0.3)',
                transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
              }}
            />
          ))}
        </div>

        {/* Real-time event feed — if orchestration data is available */}
        {orchRun && (
          <LiveEventFeed orchRun={orchRun} />
        )}

        {/* Thinking dots */}
        <ThinkingDots />
      </div>
    </>
  );
}


// ── Live Event Feed ────────────────────────────────────────────
function LiveEventFeed({ orchRun }) {
  const events = orchRun?.event_timeline?.slice(-3) || [];
  if (!events.length) return null;

  return (
    <div style={{
      borderTop: '1px solid rgba(59,130,246,0.15)',
      paddingTop: '10px',
      display: 'flex',
      flexDirection: 'column',
      gap: '4px',
    }}>
      {events.map((evt, i) => (
        <div key={i} style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          fontSize: '11px',
          color: 'rgba(110,110,115,0.8)',
          animation: i === events.length - 1 ? 'cogPulse 1.5s ease-in-out 1' : 'none',
        }}>
          <div style={{
            width: '5px', height: '5px',
            borderRadius: '50%',
            background: evt.severity === 'warning' ? '#f59e0b' : 'rgba(59,130,246,0.6)',
            flexShrink: 0,
          }} />
          <span style={{ fontFamily: 'monospace', fontSize: '10px' }}>
            {formatEventType(evt.event_type)}
          </span>
        </div>
      ))}
    </div>
  );
}

function formatEventType(type = '') {
  return type
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}


// ── Animated Thinking Dots ─────────────────────────────────────
function ThinkingDots() {
  return (
    <div style={{
      display: 'flex',
      gap: '4px',
      paddingLeft: '38px',
      alignItems: 'center',
    }}>
      {[0, 1, 2].map(i => (
        <div
          key={i}
          style={{
            width: '5px', height: '5px',
            borderRadius: '50%',
            background: 'rgba(59,130,246,0.7)',
            animation: `cogDot 1.4s ${i * 0.2}s ease-in-out infinite`,
          }}
        />
      ))}
    </div>
  );
}


// ── Quality Badge (shown post-response) ───────────────────────
export function CognitionQualityBadge({ currentResult }) {
  const artifact = currentResult?.omega_metadata?.cognitive_artifact;
  const orchRun = currentResult?.omega_metadata?.orchestration_run;

  if (!artifact && !orchRun) return null;

  const grade = artifact?.quality_indicators?.response_grade || 'B';
  const confidence = orchRun?.final_confidence ?? artifact?.quality_indicators?.confidence ?? null;
  const path = orchRun?.execution_path || 'ensemble';

  const gradeColors = {
    A: { bg: 'rgba(52,199,89,0.12)', text: '#34c759', border: 'rgba(52,199,89,0.25)' },
    B: { bg: 'rgba(59,130,246,0.12)', text: '#3b82f6', border: 'rgba(59,130,246,0.25)' },
    C: { bg: 'rgba(245,158,11,0.12)', text: '#f59e0b', border: 'rgba(245,158,11,0.25)' },
    D: { bg: 'rgba(239,68,68,0.12)', text: '#ef4444', border: 'rgba(239,68,68,0.25)' },
  };
  const colors = gradeColors[grade] || gradeColors.B;

  return (
    <div style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: '8px',
      padding: '4px 10px',
      borderRadius: '20px',
      background: colors.bg,
      border: `1px solid ${colors.border}`,
      fontSize: '11px',
      fontWeight: 600,
      color: colors.text,
    }}>
      <span>Grade {grade}</span>
      {confidence !== null && (
        <span style={{ opacity: 0.8 }}>· {Math.round(confidence * 100)}% confidence</span>
      )}
      <span style={{ opacity: 0.7, textTransform: 'uppercase', fontSize: '10px', letterSpacing: '0.06em' }}>
        {path.replace(/_/g, ' ')}
      </span>
    </div>
  );
}
