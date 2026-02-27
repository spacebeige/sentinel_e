import React from 'react';
import { resolveRenderMode } from '../../engines/modeController';
import StandardView from './StandardView';
import DebateView from './DebateView';
import EvidenceView from './EvidenceView';
import GlassView from './GlassView';
import EnsembleView from './EnsembleView';

/**
 * StructuredOutput — Sentinel-E Cognitive Engine v7.0
 * 
 * ALWAYS renders EnsembleView (CognitiveDashboard).
 * No mode-based routing. No conditional crossover.
 * All requests go through the cognitive ensemble pipeline.
 *
 * Legacy views preserved as fallback for old cached responses only.
 */
export default function StructuredOutput({ result, activeSubMode }) {
  if (!result) return null;

  const renderMode = resolveRenderMode(result);
  const meta = result.omega_metadata || {};

  // v7.0: Always render ensemble view - the only mode that exists
  const effectiveMode = 'ensemble';

  switch (effectiveMode) {
    case 'ensemble':
      return (
        <EnsembleView
          data={renderMode.data || meta}
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
