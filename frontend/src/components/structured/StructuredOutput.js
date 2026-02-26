import React from 'react';
import { resolveRenderMode } from '../../engines/modeController';
import StandardView from './StandardView';
import DebateView from './DebateView';
import EvidenceView from './EvidenceView';
import GlassView from './GlassView';
import EnsembleView from './EnsembleView';

/**
 * StructuredOutput — Strict mode-isolated output router
 * 
 * ARCHITECTURE:
 *   Each mode renders ONLY its own component.
 *   No shared render blocks. No conditional crossover.
 *   Standard mode NEVER shows debate elements.
 *   Debate mode NEVER shows aggregation summary.
 *
 * Mode routing is determined by omega_metadata from the backend.
 * The frontend sub_mode prop acts as a strict constraint — if present,
 * it overrides metadata-based detection to prevent mode leakage.
 */
export default function StructuredOutput({ result, activeSubMode }) {
  if (!result) return null;

  const renderMode = resolveRenderMode(result);
  const meta = result.omega_metadata || {};

  // STRICT MODE ISOLATION:
  // If the user selected a specific sub_mode, ONLY render that mode's view.
  // This prevents the backend returning debate_result when standard was selected.
  const effectiveMode = activeSubMode
    ? activeSubMode  // User's explicit selection takes precedence
    : renderMode.mode;

  switch (effectiveMode) {
    case 'ensemble':
      return (
        <EnsembleView
          data={renderMode.mode === 'ensemble' ? renderMode.data : meta}
          boundary={renderMode.boundary}
          confidence={renderMode.confidence}
        />
      );

    case 'standard':
    case 'aggregation':
      return (
        <StandardView
          data={renderMode.mode === 'aggregation' ? renderMode.data : meta.aggregation_result || null}
          boundary={renderMode.boundary}
          confidence={renderMode.confidence}
          disagreementScore={meta.boundary_result?.disagreement_score}
        />
      );

    case 'debate':
      // Only render debate if we actually have debate data
      if (!meta.debate_result && !renderMode.data) return null;
      return (
        <DebateView
          data={renderMode.mode === 'debate' ? renderMode.data : meta.debate_result}
          boundary={renderMode.boundary}
          confidence={renderMode.confidence}
        />
      );

    case 'evidence':
      if (!meta.forensic_result && !renderMode.data) return null;
      return (
        <EvidenceView
          data={renderMode.mode === 'evidence' ? renderMode.data : meta.forensic_result}
          boundary={renderMode.boundary}
          confidence={renderMode.confidence}
        />
      );

    case 'glass':
      if (!meta.audit_result && !renderMode.data) return null;
      return (
        <GlassView
          data={renderMode.mode === 'glass' ? renderMode.data : meta.audit_result}
          boundary={renderMode.boundary}
          confidence={renderMode.confidence}
        />
      );

    case 'kill':
      // Kill mode diagnostic — minimal
      return null;

    default:
      return null;
  }
}
