import React, { useState } from 'react';
import { ChevronDown, ChevronUp, CheckCircle, XCircle, AlertCircle, Quote, Globe, Clock, Eye, EyeOff, FileText } from 'lucide-react';
import ConfidenceBar from './ConfidenceBar';
import BoundaryPanel from './BoundaryPanel';

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

/**
 * EvidenceView — Forensic claim engine output
 *
 * Features:
 * - Claims table with source, agreement, reliability
 * - Bayesian confidence + key metrics
 * - Contradictions with severity
 * - Verbatim citations
 * - Pipeline visualization
 * - Collapsible "Evidence Sources" panel
 * - Raw / Refined output toggle
 * - Dark-mode aware
 */
export default function EvidenceView({ data, boundary, confidence }) {
  const [activeSection, setActiveSection] = useState('claims');
  const [expandedClaim, setExpandedClaim] = useState(null);
  const [showSources, setShowSources] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  if (!data) return null;

  const claims = data.all_claims || [];
  const contradictions = data.contradictions || [];
  const citations = data.verbatim_citations || [];
  const phaseLog = data.phase_log || [];
  const bayesian = data.bayesian_confidence || 0;
  const agreement = data.agreement_score || 0;
  const sourceReliability = data.source_reliability_avg || 0;
  const evidenceSources = data.evidence_sources || data.sources || [];
  const rawOutput = data.raw_output || data.raw || null;
  const refinedOutput = data.refined_output || data.synthesis || data.refined || null;

  const sections = [
    { id: 'claims', label: `Claims (${claims.length})` },
    { id: 'contradictions', label: `Contradictions (${contradictions.length})` },
    ...(citations.length > 0 ? [{ id: 'citations', label: `Citations (${citations.length})` }] : []),
    { id: 'pipeline', label: 'Pipeline' },
  ];

  const verdictIcon = (verdict) => {
    switch (verdict) {
      case 'confirmed': return <CheckCircle size={12} style={{ color: '#10b981' }} />;
      case 'contradicted': return <XCircle size={12} style={{ color: '#ef4444' }} />;
      default: return <AlertCircle size={12} style={{ color: '#f59e0b' }} />;
    }
  };

  return (
    <div className="space-y-3">
      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded-lg" style={{
            fontFamily: FONT, fontSize: '10px', fontWeight: 700,
            color: '#06b6d4', backgroundColor: '#ecfeff',
            letterSpacing: '0.05em', textTransform: 'uppercase',
          }}>
            Forensic Evidence
          </span>
          <span style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73' }}>
            {claims.length} claims · {contradictions.length} contradictions
          </span>
        </div>

        {/* Raw / Refined toggle */}
        {(rawOutput || refinedOutput) && (
          <button
            onClick={() => setShowRaw(v => !v)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg transition-colors bg-[#f5f5f7] dark:bg-white/10 hover:bg-[#e8e8ed] dark:hover:bg-white/15"
          >
            {showRaw ? <Eye size={12} className="text-[#3b82f6]" /> : <EyeOff size={12} className="text-[#6e6e73]" />}
            <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 500, color: showRaw ? '#3b82f6' : '#6e6e73' }}>
              {showRaw ? 'Raw' : 'Refined'}
            </span>
          </button>
        )}
      </div>

      {/* ── Raw / Refined Output Block ── */}
      {showRaw && rawOutput && (
        <div className="rounded-2xl bg-[#1e293b] p-4 overflow-x-auto">
          <pre className="text-[#e2e8f0] whitespace-pre-wrap" style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', lineHeight: 1.6 }}>
            {typeof rawOutput === 'string' ? rawOutput : JSON.stringify(rawOutput, null, 2)}
          </pre>
        </div>
      )}
      {!showRaw && refinedOutput && (
        <div className="rounded-2xl bg-white dark:bg-[#1c1c1e] border border-black/5 dark:border-white/5 p-4 shadow-sm">
          <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#8b5cf6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Refined Synthesis
          </span>
          <p className="mt-2 dark:text-[#e2e8f0]" style={{ fontFamily: FONT, fontSize: '14px', lineHeight: 1.6, color: '#1d1d1f' }}>
            {typeof refinedOutput === 'string' ? refinedOutput : JSON.stringify(refinedOutput)}
          </p>
        </div>
      )}

      {/* ── Key Metrics ── */}
      <div className="grid grid-cols-3 gap-2">
        {[
          { label: 'Bayesian Confidence', value: bayesian, color: '#06b6d4' },
          { label: 'Agreement', value: agreement, color: '#10b981' },
          { label: 'Source Reliability', value: sourceReliability, color: '#8b5cf6' },
        ].map(m => (
          <div key={m.label} className="rounded-2xl bg-[#f5f5f7] dark:bg-[#1c1c1e] p-3">
            <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 500, color: '#6e6e73' }}>
              {m.label}
            </span>
            <div className="mt-1.5">
              <ConfidenceBar value={m.value} size="sm" showLabel={false} />
              <span className="block mt-1 text-right" style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: m.color }}>
                {(m.value * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* ── Collapsible Evidence Sources Panel ── */}
      {evidenceSources.length > 0 && (
        <div className="rounded-2xl border border-black/5 dark:border-white/5 bg-white dark:bg-[#1c1c1e] shadow-sm overflow-hidden">
          <button
            onClick={() => setShowSources(v => !v)}
            className="w-full flex items-center justify-between px-4 py-3 hover:bg-[#f5f5f7] dark:hover:bg-white/10 transition-colors"
          >
            <div className="flex items-center gap-2">
              <FileText size={14} className="text-[#06b6d4]" />
              <span className="dark:text-[#f1f5f9]" style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 600, color: '#1d1d1f' }}>
                Evidence Sources ({evidenceSources.length})
              </span>
            </div>
            {showSources
              ? <ChevronUp size={14} className="text-[#aeaeb2]" />
              : <ChevronDown size={14} className="text-[#aeaeb2]" />
            }
          </button>
          {showSources && (
            <div className="border-t border-black/5 dark:border-white/5 divide-y divide-black/5 dark:divide-white/5">
              {evidenceSources.map((src, i) => (
                <div key={i} className="px-4 py-3">
                  <div className="flex items-start gap-2">
                    <Globe size={12} className="text-[#3b82f6] mt-0.5 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        {src.url ? (
                          <a href={src.url} target="_blank" rel="noopener noreferrer"
                            className="text-[#3b82f6] hover:underline truncate"
                            style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 500 }}>
                            {src.title || src.url}
                          </a>
                        ) : (
                          <span className="dark:text-[#f1f5f9] truncate" style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 500, color: '#1d1d1f' }}>
                            {src.title || src.name || `Source ${i + 1}`}
                          </span>
                        )}
                        {src.reliability != null && (
                          <span className="flex-shrink-0 px-1.5 py-0.5 rounded-md bg-[#f5f5f7] dark:bg-white/10" style={{
                            fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#6e6e73',
                          }}>
                            {(src.reliability * 100).toFixed(0)}% reliable
                          </span>
                        )}
                      </div>
                      {src.timestamp && (
                        <div className="flex items-center gap-1 mt-0.5">
                          <Clock size={10} className="text-[#aeaeb2]" />
                          <span style={{ fontFamily: FONT, fontSize: '10px', color: '#aeaeb2' }}>{src.timestamp}</span>
                        </div>
                      )}
                      {src.snippet && (
                        <p className="mt-1 dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '11px', lineHeight: 1.5, color: '#6e6e73' }}>
                          {src.snippet}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Section Tabs ── */}
      <div className="flex gap-1 bg-[#f5f5f7] dark:bg-[#1c1c1e] rounded-xl p-1">
        {sections.map(section => (
          <button
            key={section.id}
            onClick={() => setActiveSection(section.id)}
            className="flex-1 py-1.5 px-2 rounded-lg transition-colors"
            style={{
              fontFamily: FONT, fontSize: '12px', fontWeight: activeSection === section.id ? 600 : 400,
              backgroundColor: activeSection === section.id ? 'white' : 'transparent',
              color: activeSection === section.id ? '#1d1d1f' : '#6e6e73',
              boxShadow: activeSection === section.id ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
            }}
          >
            {section.label}
          </button>
        ))}
      </div>

      {/* ── Claims Table ── */}
      {activeSection === 'claims' && (
        <div className="space-y-1.5">
          {claims.length === 0 ? (
            <p style={{ fontFamily: FONT, fontSize: '13px', color: '#aeaeb2', textAlign: 'center', padding: '16px 0' }}>
              No claims extracted
            </p>
          ) : (
            claims.map((claim, idx) => (
              <div key={idx} className="rounded-xl border border-black/5 dark:border-white/5 bg-white dark:bg-[#1c1c1e] shadow-sm overflow-hidden">
                <button
                  onClick={() => setExpandedClaim(expandedClaim === idx ? null : idx)}
                  className="w-full flex items-center gap-2 px-3 py-2.5 hover:bg-[#f5f5f7] dark:hover:bg-white/10 transition-colors"
                >
                  <span style={{ fontFamily: FONT, fontSize: '11px', color: '#aeaeb2', fontWeight: 600, width: '24px', flexShrink: 0 }}>
                    #{idx + 1}
                  </span>
                  <span className="flex-1 text-left truncate dark:text-[#f1f5f9]" style={{ fontFamily: FONT, fontSize: '12px', color: '#1d1d1f' }}>
                    {claim.statement}
                  </span>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span className="px-1.5 py-0.5 rounded-md bg-[#f5f5f7] dark:bg-white/10" style={{
                      fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#6e6e73',
                    }}>
                      {claim.model_origin}
                    </span>
                    <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: '#1d1d1f' }} className="dark:text-[#f1f5f9]" >
                      {(claim.final_confidence * 100).toFixed(0)}%
                    </span>
                    {expandedClaim === idx
                      ? <ChevronUp size={12} className="text-[#aeaeb2]" />
                      : <ChevronDown size={12} className="text-[#aeaeb2]" />
                    }
                  </div>
                </button>

                {expandedClaim === idx && (
                  <div className="px-3 pb-3 border-t border-black/5 dark:border-white/5 space-y-2">
                    <div className="grid grid-cols-3 gap-2 mt-2">
                      {[
                        { label: 'Date', value: claim.date || 'unknown' },
                        { label: 'Source', value: claim.source_type || 'unknown' },
                        { label: 'Agreed / Opposed', value: `${claim.agreement_count || 0} / ${claim.contradiction_count || 0}` },
                      ].map(item => (
                        <div key={item.label}>
                          <span style={{ fontFamily: FONT, fontSize: '10px', color: '#aeaeb2' }}>{item.label}</span>
                          <p className="dark:text-[#f1f5f9]" style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 500, color: '#1d1d1f' }}>{item.value}</p>
                        </div>
                      ))}
                    </div>
                    <ConfidenceBar value={claim.final_confidence} label="Bayesian Confidence" size="sm" />
                    {claim.verifications && claim.verifications.length > 0 && (
                      <div className="mt-1.5">
                        <span style={{ fontFamily: FONT, fontSize: '10px', fontWeight: 600, color: '#6e6e73', textTransform: 'uppercase' }}>
                          Verifications
                        </span>
                        {claim.verifications.map((v, vi) => (
                          <div key={vi} className="flex items-center gap-2 mt-1">
                            {verdictIcon(v.verdict)}
                            <span style={{ fontFamily: FONT, fontSize: '11px', color: '#6e6e73' }}>{v.verifier}:</span>
                            <span style={{
                              fontFamily: FONT, fontSize: '11px', fontWeight: 500,
                              color: v.verdict === 'confirmed' ? '#10b981' : v.verdict === 'contradicted' ? '#ef4444' : '#f59e0b',
                            }}>
                              {v.verdict}
                            </span>
                            <span style={{ fontFamily: FONT, fontSize: '10px', color: '#aeaeb2' }}>
                              ({(v.confidence * 100).toFixed(0)}%)
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {/* ── Contradictions ── */}
      {activeSection === 'contradictions' && (
        <div className="space-y-2">
          {contradictions.length === 0 ? (
            <div className="text-center py-4 rounded-2xl bg-[#f0fdf4] dark:bg-[#10b981]/10">
              <CheckCircle size={16} className="mx-auto mb-1" style={{ color: '#10b981' }} />
              <p style={{ fontFamily: FONT, fontSize: '13px', color: '#10b981', fontWeight: 500 }}>No contradictions detected</p>
            </div>
          ) : (
            contradictions.map((c, idx) => (
              <div key={idx} className="rounded-xl border border-[#fecaca] dark:border-[#ef4444]/30 bg-[#fef2f2] dark:bg-[#ef4444]/10 p-3">
                <div className="flex items-center gap-2 mb-1.5">
                  <XCircle size={12} style={{ color: '#ef4444' }} />
                  <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 600, color: '#ef4444' }}>{c.type}</span>
                  <span style={{ fontFamily: FONT, fontSize: '10px', color: '#6e6e73' }}>
                    Severity: {(c.severity * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="dark:text-[#fca5a5]" style={{ fontFamily: FONT, fontSize: '12px', color: '#991b1b', lineHeight: 1.5 }}>
                  {c.model_a}: {c.claim_a}
                </p>
                <p className="dark:text-[#fca5a5]" style={{ fontFamily: FONT, fontSize: '12px', color: '#991b1b', lineHeight: 1.5 }}>
                  {c.model_b}: {c.claim_b}
                </p>
              </div>
            ))
          )}
        </div>
      )}

      {/* ── Verbatim Citations ── */}
      {activeSection === 'citations' && (
        <div className="space-y-2">
          {citations.map((cit, idx) => (
            <div key={idx} className="rounded-xl border border-black/5 dark:border-white/5 bg-white dark:bg-[#1c1c1e] shadow-sm p-3">
              <div className="flex items-start gap-2">
                <Quote size={14} style={{ color: '#06b6d4', marginTop: '2px', flexShrink: 0 }} />
                <div>
                  <p className="dark:text-[#e2e8f0]" style={{ fontFamily: FONT, fontSize: '13px', color: '#1d1d1f', fontStyle: 'italic', lineHeight: 1.5 }}>
                    "{cit.quote}"
                  </p>
                  <div className="flex items-center gap-3 mt-1.5 flex-wrap">
                    <span style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 500, color: '#06b6d4' }}>
                      {cit.source || 'Unknown source'}
                    </span>
                    <span style={{ fontFamily: FONT, fontSize: '10px', color: '#aeaeb2' }}>
                      Reliability: {((cit.reliability || 0) * 100).toFixed(0)}%
                    </span>
                    {cit.model_source && (
                      <span style={{ fontFamily: FONT, fontSize: '10px', color: '#aeaeb2' }}>via {cit.model_source}</span>
                    )}
                    {cit.url && (
                      <a href={cit.url} target="_blank" rel="noopener noreferrer" className="text-[#3b82f6] hover:underline"
                        style={{ fontFamily: FONT, fontSize: '10px' }}>
                        Source link
                      </a>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Pipeline Log ── */}
      {activeSection === 'pipeline' && (
        <div className="space-y-1">
          {phaseLog.map((phase, idx) => (
            <div key={idx} className="flex items-center gap-3 py-2 px-1">
              <div className="w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0" style={{
                backgroundColor: phase.status === 'complete' ? '#f0fdf4' : phase.status === 'failed' ? '#fef2f2' : '#f5f5f7',
              }}>
                {phase.status === 'complete' ? (
                  <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
                    <path d="M1 4L3.5 6.5L9 1" stroke="#10b981" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : phase.status === 'failed' ? (
                  <XCircle size={10} style={{ color: '#ef4444' }} />
                ) : (
                  <span style={{ fontFamily: FONT, fontSize: '9px', fontWeight: 600, color: '#aeaeb2' }}>{phase.phase}</span>
                )}
              </div>
              <span className="flex-1 dark:text-[#f1f5f9]" style={{ fontFamily: FONT, fontSize: '12px', color: '#1d1d1f' }}>{phase.name}</span>
              {phase.detail && <span style={{ fontFamily: FONT, fontSize: '10px', color: '#aeaeb2' }}>{phase.detail}</span>}
            </div>
          ))}
        </div>
      )}

      <BoundaryPanel boundary={boundary} />
    </div>
  );
}
