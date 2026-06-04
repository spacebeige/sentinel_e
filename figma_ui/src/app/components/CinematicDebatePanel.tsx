import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Swords, ChevronDown, ChevronRight, Trophy, BarChart2 } from 'lucide-react';
import type { OmegaMetadata } from '../types';

interface CinematicDebatePanelProps {
  metadata?: OmegaMetadata;
}

export const CinematicDebatePanel: React.FC<CinematicDebatePanelProps> = ({ metadata }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!metadata?.debate_result) return null;

  const { positions, consensus } = metadata.debate_result;
  if (!positions || positions.length === 0) return null;

  const getModelColor = (idx: number) => {
    const palette = ['#6366f1', '#f59e0b', '#10b981', '#ef4444', '#06b6d4', '#8b5cf6'];
    return palette[idx % palette.length];
  };

  return (
    <div className="mt-4 overflow-hidden rounded-2xl border border-rose-500/20 bg-[#08090e]/80 backdrop-blur-xl">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-4 py-3 bg-gradient-to-r from-rose-500/10 to-transparent hover:bg-rose-500/15 transition-colors"
      >
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-rose-500/20 text-rose-500">
            <Swords className="w-4 h-4" />
          </div>
          <span className="text-xs font-bold tracking-widest text-rose-500 uppercase">
            Debate Arena
          </span>
          <span className="ml-2 px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-300">
            {positions.length} Models
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
            className="border-t border-rose-500/10"
          >
            <div className="p-4 space-y-4">
              {/* Positions Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {positions.map((pos: any, idx: number) => {
                  const color = pos.color ?? getModelColor(idx);
                  return (
                    <div
                      key={idx}
                      className="rounded-xl p-3 border"
                      style={{
                        backgroundColor: 'rgba(255,255,255,0.02)',
                        borderColor: `${color}30`,
                      }}
                    >
                      <div className="flex items-center gap-2 mb-2 pb-2 border-b border-white/5">
                        <div
                          className="w-2 h-2 rounded-full shadow-[0_0_8px_currentColor]"
                          style={{ backgroundColor: color, color }}
                        />
                        <span className="text-[11px] font-bold text-slate-200">
                          {pos.model || `Model ${idx + 1}`}
                        </span>
                        {pos.confidence != null && (
                          <span className="ml-auto text-[10px] font-mono text-slate-400">
                            {Math.round(pos.confidence * 100)}% Conf
                          </span>
                        )}
                      </div>
                      
                      <div className="text-[12px] leading-relaxed text-slate-300 mb-3">
                        {pos.position}
                      </div>

                      {pos.key_points && pos.key_points.length > 0 && (
                        <div className="space-y-1">
                          <div className="text-[9px] uppercase tracking-wider text-slate-500 font-semibold mb-1">
                            Key Points
                          </div>
                          <ul className="space-y-1 pl-3 list-disc list-outside marker:text-slate-600">
                            {pos.key_points.map((kp: string, k: number) => (
                              <li key={k} className="text-[10px] text-slate-400">
                                {kp}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Consensus Block */}
              {consensus && (
                <div className="mt-4 p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                  <div className="flex items-center gap-2 mb-2">
                    <Trophy className="w-4 h-4 text-emerald-500" />
                    <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-500">
                      Ensemble Consensus
                    </span>
                  </div>
                  <div className="text-[13px] text-emerald-100/90 leading-relaxed">
                    {consensus}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
