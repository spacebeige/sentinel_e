import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileSearch, ChevronDown, Link as LinkIcon, AlertTriangle } from 'lucide-react';
import type { OmegaMetadata } from '../api';

interface CinematicEvidencePanelProps {
  metadata?: OmegaMetadata;
}

export const CinematicEvidencePanel: React.FC<CinematicEvidencePanelProps> = ({ metadata }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!metadata?.evidence_result) return null;

  const { sources, contradictions, evidence_confidence, source_agreement } = metadata.evidence_result;
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-4 overflow-hidden rounded-2xl border border-cyan-500/20 bg-[#08090e]/80 backdrop-blur-xl">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-4 py-3 bg-gradient-to-r from-cyan-500/10 to-transparent hover:bg-cyan-500/15 transition-colors"
      >
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/20 text-cyan-500">
            <FileSearch className="w-4 h-4" />
          </div>
          <span className="text-xs font-bold tracking-widest text-cyan-500 uppercase">
            Evidence Verification
          </span>
          <span className="ml-2 px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-300">
            {sources.length} Sources
          </span>
        </div>
        <ChevronDown
          className="w-4 h-4 text-slate-400 transition-transform duration-300"
          style={{ transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)' }}
        />
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-t border-cyan-500/10"
          >
            <div className="p-4 space-y-4">
              {/* Summary Stats */}
              <div className="flex items-center gap-4 mb-4">
                {evidence_confidence != null && (
                  <div className="px-3 py-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center gap-2">
                    <span className="text-[10px] uppercase text-cyan-600/70 font-semibold tracking-wider">Confidence</span>
                    <span className="text-xs font-bold text-cyan-400">{Math.round(evidence_confidence * 100)}%</span>
                  </div>
                )}
                {source_agreement != null && (
                  <div className="px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center gap-2">
                    <span className="text-[10px] uppercase text-emerald-600/70 font-semibold tracking-wider">Agreement</span>
                    <span className="text-xs font-bold text-emerald-400">{Math.round(source_agreement * 100)}%</span>
                  </div>
                )}
              </div>

              {/* Contradictions Alert */}
              {contradictions && contradictions.length > 0 && (
                <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-start gap-3">
                  <AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />
                  <div>
                    <div className="text-[11px] font-bold uppercase tracking-wider text-amber-500 mb-1">
                      {contradictions.length} Contradictions Detected
                    </div>
                    <ul className="list-disc list-inside space-y-1">
                      {contradictions.map((c: any, i: number) => (
                        <li key={i} className="text-[11px] text-amber-200/80">
                          {c.claim || JSON.stringify(c)}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {/* Sources List */}
              <div className="space-y-2">
                {sources.map((source: any, idx: number) => (
                  <div
                    key={idx}
                    className="flex flex-col gap-1.5 p-3 rounded-xl border border-white/5 bg-white/5 hover:bg-white/10 transition-colors"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <LinkIcon className="w-3 h-3 text-cyan-500 shrink-0" />
                        <span className="text-[12px] font-semibold text-slate-200 truncate">
                          {source.title || source.domain || 'Source'}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        {source.reliability_score != null && (
                          <span className="px-1.5 py-0.5 rounded bg-slate-800 text-[9px] font-mono text-cyan-400">
                            {Math.round(source.reliability_score * 100)}% reliable
                          </span>
                        )}
                      </div>
                    </div>
                    {source.content_snippet && (
                      <p className="text-[11px] text-slate-400 leading-relaxed pl-4 border-l-2 border-cyan-500/30 ml-0.5">
                        "{source.content_snippet}"
                      </p>
                    )}
                    {source.url && (
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[10px] text-cyan-500/70 hover:text-cyan-400 truncate ml-5"
                      >
                        {source.url}
                      </a>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
