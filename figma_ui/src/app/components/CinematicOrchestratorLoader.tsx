import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Cognitive Phase Labels (mirrors backend CognitivePhase)
const PHASE_SEQUENCE = [
  { phase: 'observe', label: 'Observing', icon: '🔍', detail: 'Parsing semantic intent' },
  { phase: 'analyze', label: 'Analyzing', icon: '🧠', detail: 'Routing decision in progress' },
  { phase: 'route', label: 'Routing', icon: '🗺️', detail: 'Selecting execution path' },
  { phase: 'retrieve_memory', label: 'Memory Retrieval', icon: '💾', detail: 'Injecting cognitive context' },
  { phase: 'spawn_agents', label: 'Spawning Agents', icon: '⚡', detail: 'Parallel model execution' },
  { phase: 'debate', label: 'Debating', icon: '⚔️', detail: 'Multi-model reasoning active' },
  { phase: 'verify', label: 'Verifying', icon: '✅', detail: 'Contradiction analysis' },
  { phase: 'synthesize', label: 'Synthesizing', icon: '🔗', detail: 'Ensemble convergence' },
  { phase: 'reflect', label: 'Reflecting', icon: '🪞', detail: 'Metacognitive analysis' },
  { phase: 'store_snapshot', label: 'Storing', icon: '📦', detail: 'Persisting cognitive state' },
];

const MODE_PHASE_FILTERS = {
  single_model: ['observe', 'analyze', 'route', 'spawn_agents', 'synthesize'],
  standard: ['observe', 'analyze', 'route', 'retrieve_memory', 'spawn_agents', 'synthesize'],
  experimental: PHASE_SEQUENCE.map((p) => p.phase),
  debate: PHASE_SEQUENCE.map((p) => p.phase),
  ensemble: PHASE_SEQUENCE.map((p) => p.phase),
};

interface CinematicOrchestratorLoaderProps {
  isLoading: boolean;
  mode?: string;
  subMode?: string | null;
}

export const CinematicOrchestratorLoader: React.FC<CinematicOrchestratorLoaderProps> = ({
  isLoading,
  mode = 'standard',
  subMode = null,
}) => {
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [elapsedMs, setElapsedMs] = useState(0);

  const startTimeRef = useRef<number | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const elapsedRef = useRef<NodeJS.Timeout | null>(null);

  const activePhases = useMemo(() => {
    const key = subMode === 'debate' ? 'debate' : mode;
    // @ts-ignore
    const filter = MODE_PHASE_FILTERS[key] || MODE_PHASE_FILTERS.standard;
    return PHASE_SEQUENCE.filter((p) => filter.includes(p.phase));
  }, [mode, subMode]);

  useEffect(() => {
    if (!isLoading) {
      setPhaseIndex(0);
      setElapsedMs(0);
      startTimeRef.current = null;
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (elapsedRef.current) clearInterval(elapsedRef.current);
      return;
    }

    startTimeRef.current = Date.now();

    // Advance phase every ~2.8s
    intervalRef.current = setInterval(() => {
      setPhaseIndex((prev) => (prev + 1) % activePhases.length);
    }, 2800);

    // Elapsed timer
    elapsedRef.current = setInterval(() => {
      setElapsedMs(Date.now() - (startTimeRef.current || Date.now()));
    }, 100);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (elapsedRef.current) clearInterval(elapsedRef.current);
    };
  }, [isLoading, activePhases.length]);

  if (!isLoading) return null;

  const currentPhase = activePhases[phaseIndex] || activePhases[0];
  const elapsedSec = (elapsedMs / 1000).toFixed(1);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
      className="relative overflow-hidden flex flex-col gap-3 p-4 rounded-2xl border border-indigo-500/25 bg-gradient-to-br from-indigo-500/5 to-cyan-500/5 backdrop-blur-xl"
    >
      {/* Scanning light effect */}
      <motion.div
        className="absolute top-0 bottom-0 left-0 w-[40%] pointer-events-none"
        style={{
          background: 'linear-gradient(90deg, transparent, rgba(99,102,241,0.08), transparent)',
        }}
        animate={{ x: ['-100%', '250%'] }}
        transition={{ duration: 2.5, ease: 'easeInOut', repeat: Infinity }}
      />

      {/* Header row */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          {/* Pulse Indicator */}
          <motion.div
            animate={{
              boxShadow: [
                '0 0 0 0 rgba(99, 102, 241, 0.4)',
                '0 0 0 8px rgba(99, 102, 241, 0)',
              ],
            }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-8 h-8 rounded-lg bg-slate-900/65 border border-indigo-500/35 flex items-center justify-center shrink-0"
          >
            <div className="w-2 h-2 rounded-full bg-indigo-400" />
          </motion.div>

          <div>
            <div className="text-[10px] font-semibold tracking-widest text-indigo-400 uppercase">
              Cognitive Runtime
            </div>
            <AnimatePresence mode="wait">
              <motion.div
                key={currentPhase.phase}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                className="text-sm font-semibold text-slate-100 mt-0.5 flex items-center gap-2"
              >
                <span>{currentPhase.icon}</span>
                <span>{currentPhase.label}</span>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>

        {/* Elapsed time */}
        <div className="text-xs text-slate-400/80 font-mono min-w-[40px] text-right">
          {elapsedSec}s
        </div>
      </div>

      {/* Phase detail line */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentPhase.detail}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="text-xs text-slate-400/90 pl-11"
        >
          {currentPhase.detail}
        </motion.div>
      </AnimatePresence>

      {/* Phase progress dots */}
      <div className="flex gap-[5px] pl-11 items-center mt-1">
        {activePhases.map((p, idx) => (
          <motion.div
            key={p.phase}
            animate={{
              width: idx === phaseIndex ? 18 : 6,
              backgroundColor:
                idx === phaseIndex
                  ? '#94a3b8' // slate-400
                  : idx < phaseIndex
                  ? 'rgba(99, 102, 241, 0.4)' // active past
                  : 'rgba(148, 163, 184, 0.2)', // inactive future
            }}
            transition={{ duration: 0.4, ease: 'easeOut' }}
            className="h-1.5 rounded-full"
          />
        ))}
      </div>
    </motion.div>
  );
};
