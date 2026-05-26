import React, { useState, useEffect } from 'react';
import { Link } from 'react-router';
import { motion, AnimatePresence } from 'motion/react';
import AgentNode from '../systems/AgentNode';
import { easeCinematic, easeSharp } from '../motion/CognitionMotion';

export default function DebateExperiment() {
  const [phase, setPhase] = useState('thesis'); // thesis, conflict, synthesis

  useEffect(() => {
    const phases = ['thesis', 'conflict', 'synthesis'];
    let idx = 0;
    const interval = setInterval(() => {
      idx = (idx + 1) % phases.length;
      setPhase(phases[idx]);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#050505] text-white overflow-hidden relative">
      <div className="absolute top-8 left-8 z-50">
        <Link to="/lab" className="text-xs font-mono tracking-widest text-zinc-500 hover:text-white transition-colors uppercase">← Back to Lab</Link>
      </div>
      
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-full max-w-6xl aspect-[2/1] relative">
          <motion.div 
            className="absolute inset-0 rounded-[3rem] border border-white/5 backdrop-blur-3xl"
            animate={{
              background: phase === 'conflict' 
                ? 'radial-gradient(ellipse at center, rgba(239,68,68,0.05) 0%, transparent 70%)'
                : phase === 'synthesis'
                ? 'radial-gradient(ellipse at center, rgba(16,185,129,0.05) 0%, transparent 70%)'
                : 'radial-gradient(ellipse at center, rgba(59,130,246,0.05) 0%, transparent 70%)'
            }}
            transition={{ duration: 1.5, ease: easeCinematic }}
          />
          
          <div className="absolute top-12 left-1/2 -translate-x-1/2 text-xs font-mono text-zinc-400 tracking-widest uppercase">
            AI Tribunal Arena — {phase.toUpperCase()}
          </div>
          
          <div className="absolute left-24 top-1/2 -translate-y-1/2 flex flex-col items-center">
            <AgentNode model="gpt" label="THESIS_NODE" state={phase === 'conflict' ? 'debating' : 'thinking'} size="lg" />
            <div className="h-32 relative mt-12 w-64">
              <AnimatePresence>
                {phase === 'conflict' && (
                  <motion.div 
                    initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    className="absolute inset-0 p-4 border border-red-500/20 bg-red-500/5 rounded-xl text-xs font-mono text-zinc-300 flex items-center justify-center text-center"
                  >
                    Semantic conflict detected in primary assumption.
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
          
          <div className="absolute right-24 top-1/2 -translate-y-1/2 flex flex-col items-center">
            <AgentNode model="mistral" label="ANTITHESIS_NODE" state={phase === 'conflict' ? 'debating' : phase === 'synthesis' ? 'idle' : 'thinking'} size="lg" />
            <div className="h-32 relative mt-12 w-64">
              <AnimatePresence>
                {phase === 'conflict' && (
                  <motion.div 
                    initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    className="absolute inset-0 p-4 border border-blue-500/20 bg-blue-500/5 rounded-xl text-xs font-mono text-zinc-300 flex items-center justify-center text-center"
                  >
                    Counter-evidence synthesized and routed.
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
          
          {/* Conflict Lines */}
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-px overflow-hidden">
            <motion.div 
              className="w-full h-full"
              animate={{
                background: phase === 'conflict'
                  ? 'linear-gradient(90deg, transparent, rgba(239,68,68,0.8), rgba(59,130,246,0.8), transparent)'
                  : phase === 'synthesis'
                  ? 'linear-gradient(90deg, transparent, rgba(16,185,129,0.5), transparent)'
                  : 'linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent)'
              }}
              transition={{ duration: 0.5 }}
            />
            {phase === 'conflict' && (
              <motion.div 
                className="absolute inset-0 w-1/4 h-full bg-white blur-sm"
                animate={{ left: ['0%', '100%', '0%'] }}
                transition={{ duration: 0.5, ease: easeSharp, repeat: Infinity }}
              />
            )}
          </div>
          
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
            <motion.div 
              className="w-24 h-24 border rounded-full flex items-center justify-center backdrop-blur-md"
              animate={{
                scale: phase === 'synthesis' ? 1.2 : 1,
                borderColor: phase === 'synthesis' ? 'rgba(16,185,129,0.5)' : phase === 'conflict' ? 'rgba(239,68,68,0.3)' : 'rgba(255,255,255,0.1)',
                backgroundColor: phase === 'synthesis' ? 'rgba(16,185,129,0.1)' : 'rgba(0,0,0,0.5)'
              }}
              transition={{ duration: 1, ease: easeCinematic }}
            >
              <motion.div 
                className="w-4 h-4 rounded-full"
                animate={{
                  scale: phase === 'synthesis' ? [1, 1.5, 1] : phase === 'conflict' ? [1, 0.5, 1] : 1,
                  backgroundColor: phase === 'synthesis' ? '#10b981' : phase === 'conflict' ? '#ef4444' : '#ffffff',
                  opacity: phase === 'thesis' ? 0.2 : 1
                }}
                transition={{ duration: phase === 'synthesis' ? 2 : 0.2, repeat: Infinity }}
              />
            </motion.div>
            <div className="absolute -bottom-10 left-1/2 -translate-x-1/2 text-[10px] font-mono text-zinc-500 tracking-widest whitespace-nowrap uppercase">
              {phase === 'synthesis' ? 'CONSENSUS_REACHED' : 'SYNTHESIS_GRAVITY'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
