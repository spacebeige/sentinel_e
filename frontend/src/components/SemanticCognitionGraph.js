/**
 * ============================================================
 * SemanticCognitionGraph — Cognitive Spatial Visualization
 * ============================================================
 * Sentinel-E v8.0 — Persistent Hybrid Cognitive Runtime
 *
 * Reads from EXISTING response fields — no new backend data needed:
 *   - model_outputs     → claim nodes
 *   - debate_result     → contradiction fields / edges
 *   - agreement_matrix  → edge weights
 *   - confidence_evolution → node intensity
 *   - tactical_map      → cluster topology
 *   - cognitive_artifact → layered cognition structure
 *
 * Rendered as a pure SVG force-directed graph.
 * No external library dependencies.
 *
 * Props:
 *   response — full API response object (omega_metadata)
 *   width    — canvas width (default 560)
 *   height   — canvas height (default 360)
 * ============================================================
 */

import React, { useRef, useMemo, useState, useCallback } from 'react';

// ── Constants ─────────────────────────────────────────────────
const NODE_TYPES = {
  CLAIM:        { color: '#3b82f6', radius: 14, label: 'Claim' },
  SYNTHESIS:    { color: '#06b6d4', radius: 18, label: 'Synthesis' },
  EVIDENCE:     { color: '#34c759', radius: 10, label: 'Evidence' },
  CONTRADICTION:{ color: '#f59e0b', radius: 12, label: 'Contradiction' },
  MEMORY:       { color: '#8b5cf6', radius: 10, label: 'Memory' },
};

const EDGE_TYPES = {
  AGREEMENT:    { color: 'rgba(59,130,246,0.4)', width: 2 },
  CONTRADICTION:{ color: 'rgba(245,158,11,0.6)', width: 2, dashed: true },
  EVIDENCE:     { color: 'rgba(52,199,89,0.35)', width: 1.5 },
  MEMORY:       { color: 'rgba(139,92,246,0.35)', width: 1, dashed: true },
};

// ── Graph Data Builder ─────────────────────────────────────────
function buildGraphData(response) {
  if (!response) return { nodes: [], edges: [] };

  const nodes = [];
  const edges = [];
  const omega = response.omega_metadata || response;
  const artifact = omega.cognitive_artifact || null;

  // ── Central Synthesis Node ─────────────────────────────────
  const synthId = 'synthesis';
  nodes.push({
    id: synthId,
    type: 'SYNTHESIS',
    label: 'Synthesis',
    detail: 'Ensemble consensus',
    confidence: response.confidence || 0.7,
    x: 0, y: 0,  // will be positioned by layout
  });

  // ── Model Output / Claim Nodes ─────────────────────────────
  const modelOutputs = (omega.model_outputs || omega.all_outputs || []).map((output) => ({
    model_id: output.model_id || output.model_name,
    model_name: output.model_name || output.model_id || 'Model',
    confidence: output.confidence ?? output.score?.final_score ?? 0.5,
    position: output.position || output.raw_output || '',
  }));
  modelOutputs.slice(0, 6).forEach((output, i) => {
    const nodeId = `model_${i}`;
    const confidence = output.confidence ?? 0.5;
    nodes.push({
      id: nodeId,
      type: 'CLAIM',
      label: (output.model_name || `Model ${i+1}`).split('-')[0].slice(0, 8),
      detail: (output.position || '').slice(0, 60),
      confidence,
      modelId: output.model_id,
    });

    // Edge: model → synthesis (agreement-weighted)
    const matrixData = omega.agreement_matrix || {};
    const pairAgreement = matrixData.pairwise_scores?.[output.model_id]?.['synthesis'] || confidence;
    edges.push({
      from: nodeId,
      to: synthId,
      type: pairAgreement > 0.6 ? 'AGREEMENT' : 'CONTRADICTION',
      weight: pairAgreement,
    });
  });

  // ── Contradiction Nodes ────────────────────────────────────
  const debateResult = omega.debate_result || {};
  const unresolvedConflicts = debateResult.unresolved_conflicts || [];
  unresolvedConflicts.slice(0, 3).forEach((conflict, i) => {
    const cId = `conflict_${i}`;
    nodes.push({
      id: cId,
      type: 'CONTRADICTION',
      label: `Conflict ${i+1}`,
      detail: String(conflict).slice(0, 60),
      confidence: 0.3,
    });
    // Link to synthesis and random claim
    edges.push({ from: cId, to: synthId, type: 'CONTRADICTION', weight: 0.2 });
    if (nodes.length > 2) {
      edges.push({ from: cId, to: `model_0`, type: 'CONTRADICTION', weight: 0.2 });
    }
  });

  // ── Evidence Nodes ─────────────────────────────────────────
  const evidenceMatrix = artifact?.evidence_matrix || [];
  evidenceMatrix.slice(0, 4).forEach((ev, i) => {
    const eId = `evidence_${i}`;
    nodes.push({
      id: eId,
      type: 'EVIDENCE',
      label: (ev.title || ev.type || 'Evidence').slice(0, 10),
      detail: (ev.content_preview || ev.claim || '').slice(0, 60),
      confidence: ev.reliability || ev.confidence || 0.6,
    });
    // Evidence → synthesis
    edges.push({ from: eId, to: synthId, type: 'EVIDENCE', weight: ev.reliability || 0.6 });
  });

  // ── Memory Nodes ───────────────────────────────────────────
  const orchRun = omega.orchestration_run || {};
  const memRetrievals = orchRun.memory_retrievals || [];
  memRetrievals.slice(0, 3).forEach((mem, i) => {
    const mId = `memory_${i}`;
    nodes.push({
      id: mId,
      type: 'MEMORY',
      label: (mem.layer || 'Memory').slice(0, 8),
      detail: (mem.preview || '').slice(0, 60),
      confidence: mem.relevance || 0.7,
    });
    // Memory → synthesis
    edges.push({ from: mId, to: synthId, type: 'MEMORY', weight: mem.relevance || 0.7 });
  });

  return { nodes, edges };
}


// ── Simple Force Layout ────────────────────────────────────────
function computeLayout(nodes, edges, width, height, iterations = 80) {
  const cx = width / 2;
  const cy = height / 2;

  // Initialize positions
  const positions = {};
  nodes.forEach((node, i) => {
    if (node.id === 'synthesis') {
      positions[node.id] = { x: cx, y: cy };
    } else {
      const angle = (2 * Math.PI * i) / (nodes.length - 1);
      const radius = Math.min(width, height) * 0.32;
      positions[node.id] = {
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle),
      };
    }
  });

  // Force-directed iterations
  for (let iter = 0; iter < iterations; iter++) {
    const forces = {};
    nodes.forEach(n => { forces[n.id] = { fx: 0, fy: 0 }; });

    // Repulsion
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = positions[nodes[i].id];
        const b = positions[nodes[j].id];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = 1800 / (dist * dist);
        forces[nodes[i].id].fx += (dx / dist) * force;
        forces[nodes[i].id].fy += (dy / dist) * force;
        forces[nodes[j].id].fx -= (dx / dist) * force;
        forces[nodes[j].id].fy -= (dy / dist) * force;
      }
    }

    // Attraction along edges
    edges.forEach(edge => {
      const a = positions[edge.from];
      const b = positions[edge.to];
      if (!a || !b) return;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const strength = 0.04 * (edge.weight || 0.5);
      forces[edge.from].fx += dx * strength;
      forces[edge.from].fy += dy * strength;
      forces[edge.to].fx -= dx * strength;
      forces[edge.to].fy -= dy * strength;
    });

    // Apply forces (synthesis node is anchored)
    nodes.forEach(node => {
      if (node.id === 'synthesis') return;
      const pos = positions[node.id];
      const cooling = 1 - iter / iterations;
      pos.x += forces[node.id].fx * cooling * 0.3;
      pos.y += forces[node.id].fy * cooling * 0.3;
      // Boundary clamp
      const margin = 30;
      pos.x = Math.max(margin, Math.min(width - margin, pos.x));
      pos.y = Math.max(margin, Math.min(height - margin, pos.y));
    });
  }

  return positions;
}


// ── Main Component ─────────────────────────────────────────────
export default function SemanticCognitionGraph({
  response,
  width = 560,
  height = 360,
}) {
  const svgRef = useRef(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [tooltip, setTooltip] = useState(null);

  const { nodes, edges } = useMemo(() => buildGraphData(response), [response]);
  const positions = useMemo(
    () => computeLayout(nodes, edges, width, height),
    [nodes, edges, width, height]
  );

  // Compute confidence-based intensity for node glow
  const getNodeStyle = useCallback((node) => {
    const typeSpec = NODE_TYPES[node.type] || NODE_TYPES.CLAIM;
    const intensity = node.confidence || 0.6;
    const isHovered = hoveredNode === node.id;
    const r = typeSpec.radius * (isHovered ? 1.3 : 1);
    return { ...typeSpec, radius: r, intensity, isHovered };
  }, [hoveredNode]);

  if (!response || nodes.length === 0) return null;

  const contradictionDensity = response.omega_metadata?.cognitive_artifact?.contradiction_analysis?.density || 0;
  const modelCount =
    response.omega_metadata?.reasoning_trace?.models_executed
    || response.omega_metadata?.model_count
    || nodes.length;

  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(15,15,25,0.97) 0%, rgba(10,20,40,0.97) 100%)',
      borderRadius: '16px',
      border: '1px solid rgba(59,130,246,0.2)',
      padding: '16px',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '12px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: '6px', height: '6px', borderRadius: '50%',
            background: '#3b82f6',
            boxShadow: '0 0 8px #3b82f6',
            animation: 'cogPulse 2s ease-in-out infinite',
          }} />
          <span style={{
            fontSize: '11px',
            fontWeight: 600,
            color: 'rgba(255,255,255,0.7)',
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
          }}>
            Semantic Cognition Graph
          </span>
        </div>
        <div style={{ display: 'flex', gap: '16px' }}>
          <Stat label="Nodes" value={nodes.length} />
          <Stat label="Models" value={modelCount} />
          <Stat
            label="Contradiction"
            value={`${Math.round(contradictionDensity * 100)}%`}
            color={contradictionDensity > 0.5 ? '#f59e0b' : '#34c759'}
          />
        </div>
      </div>

      {/* SVG Graph */}
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{ display: 'block', maxWidth: '100%' }}
        viewBox={`0 0 ${width} ${height}`}
      >
        <defs>
          {/* Glow filters */}
          <filter id="glow-blue">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="glow-amber">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          {/* Radial grid */}
          <radialGradient id="grid-gradient" cx="50%" cy="50%">
            <stop offset="0%" stopColor="rgba(59,130,246,0.04)" />
            <stop offset="100%" stopColor="transparent" />
          </radialGradient>
        </defs>

        {/* Background grid */}
        <rect width={width} height={height} fill="url(#grid-gradient)" rx="12" />
        {[1, 2, 3].map(i => (
          <circle
            key={i}
            cx={width / 2} cy={height / 2}
            r={i * (Math.min(width, height) * 0.15)}
            fill="none"
            stroke="rgba(59,130,246,0.05)"
            strokeWidth="1"
          />
        ))}

        {/* Edges */}
        {edges.map((edge, i) => {
          const from = positions[edge.from];
          const to = positions[edge.to];
          if (!from || !to) return null;
          const spec = EDGE_TYPES[edge.type] || EDGE_TYPES.AGREEMENT;
          return (
            <line
              key={i}
              x1={from.x} y1={from.y}
              x2={to.x} y2={to.y}
              stroke={spec.color}
              strokeWidth={spec.width * (edge.weight || 0.5) * 2}
              strokeDasharray={spec.dashed ? '4 4' : 'none'}
              opacity={0.7}
            />
          );
        })}

        {/* Nodes */}
        {nodes.map((node) => {
          const pos = positions[node.id];
          if (!pos) return null;
          const style = getNodeStyle(node);
          const isContra = node.type === 'CONTRADICTION';
          return (
            <g
              key={node.id}
              transform={`translate(${pos.x},${pos.y})`}
              style={{ cursor: 'pointer' }}
              onMouseEnter={() => {
                setHoveredNode(node.id);
                setTooltip({ node, x: pos.x, y: pos.y });
              }}
              onMouseLeave={() => {
                setHoveredNode(null);
                setTooltip(null);
              }}
            >
              {/* Outer glow ring */}
              {style.isHovered && (
                <circle
                  r={style.radius + 6}
                  fill="none"
                  stroke={style.color}
                  strokeWidth="1"
                  opacity="0.4"
                  filter={isContra ? 'url(#glow-amber)' : 'url(#glow-blue)'}
                />
              )}
              {/* Confidence intensity ring */}
              <circle
                r={style.radius + 3}
                fill="none"
                stroke={style.color}
                strokeWidth="0.5"
                opacity={style.intensity * 0.5}
              />
              {/* Main node */}
              <circle
                r={style.radius}
                fill={style.color}
                opacity={0.85 + style.intensity * 0.15}
                filter={node.id === 'synthesis' ? 'url(#glow-blue)' : 'none'}
              />
              {/* Label */}
              <text
                textAnchor="middle"
                dy={style.radius + 12}
                style={{
                  fontSize: '9px',
                  fill: 'rgba(255,255,255,0.75)',
                  fontFamily: "'Inter', -apple-system, sans-serif",
                  fontWeight: 600,
                  pointerEvents: 'none',
                }}
              >
                {node.label}
              </text>
            </g>
          );
        })}

        {/* Tooltip */}
        {tooltip && positions[tooltip.node.id] && (
          <GraphTooltip
            node={tooltip.node}
            x={positions[tooltip.node.id].x}
            y={positions[tooltip.node.id].y}
            width={width}
            height={height}
          />
        )}
      </svg>

      {/* Legend */}
      <div style={{
        display: 'flex',
        gap: '12px',
        marginTop: '10px',
        flexWrap: 'wrap',
      }}>
        {Object.entries(NODE_TYPES).map(([key, spec]) => (
          <div key={key} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <div style={{
              width: '8px', height: '8px',
              borderRadius: '50%',
              background: spec.color,
            }} />
            <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.5)' }}>
              {spec.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Stat({ label, value, color = 'rgba(255,255,255,0.6)' }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: '13px', fontWeight: 700, color }}>{value}</div>
      <div style={{ fontSize: '9px', color: 'rgba(255,255,255,0.4)', letterSpacing: '0.05em' }}>
        {label.toUpperCase()}
      </div>
    </div>
  );
}

function GraphTooltip({ node, x, y, width, height }) {
  const typeSpec = NODE_TYPES[node.type] || NODE_TYPES.CLAIM;
  // Keep tooltip in bounds
  const tx = x > width * 0.7 ? x - 150 : x + 20;
  const ty = y > height * 0.7 ? y - 80 : y + 20;

  return (
    <g transform={`translate(${tx},${ty})`}>
      <rect
        x="0" y="0" width="140" height="65"
        rx="8"
        fill="rgba(15,20,35,0.95)"
        stroke={typeSpec.color}
        strokeWidth="1"
        opacity="0.95"
      />
      <text x="10" y="20" style={{ fontSize: '10px', fill: typeSpec.color, fontWeight: 700 }}>
        {typeSpec.label} · {Math.round((node.confidence || 0.5) * 100)}%
      </text>
      <text x="10" y="35" style={{ fontSize: '9px', fill: 'rgba(255,255,255,0.75)' }}>
        {node.label}
      </text>
      <foreignObject x="8" y="42" width="124" height="20">
        <div xmlns="http://www.w3.org/1999/xhtml" style={{
          fontSize: '8px',
          color: 'rgba(255,255,255,0.5)',
          lineHeight: '1.3',
          overflow: 'hidden',
          whiteSpace: 'nowrap',
          textOverflow: 'ellipsis',
        }}>
          {node.detail || ''}
        </div>
      </foreignObject>
    </g>
  );
}
