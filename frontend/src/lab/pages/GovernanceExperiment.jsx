import React, { useState, useEffect } from 'react';
import { Link } from 'react-router';
import { motion, AnimatePresence } from 'motion/react';

export default function GovernanceExperiment() {
  const [anomaly, setAnomaly] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setAnomaly(a => !a);
    }, 6000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#020202] text-white overflow-hidden relative">
      <div className="absolute top-8 left-8 z-50">
        <Link to="/lab" className="text-xs font-mono tracking-widest text-zinc-500 hover:text-white transition-colors uppercase">← Back to Lab</Link>
      </div>
      
      <div className="absolute inset-0 flex items-center justify-center p-12">
        <div className="w-full max-w-[1200px] h-[700px] flex gap-8">
          
          <div className="flex-1 border border-white/5 bg-[#050508] rounded-[2rem] p-8 relative overflow-hidden flex flex-col shadow-2xl">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-emerald-500/30 to-transparent" />
            <div className="text-[10px] font-mono text-emerald-500 tracking-widest mb-10 uppercase">Tactical Audit System</div>
            
            <div className="space-y-4 flex-1">
              {[1, 2, 3, 4, 5].map((i) => (
                <motion.div 
                  key={i} 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className={`border p-5 rounded-xl flex items-center justify-between transition-colors ${
                    i === 3 && anomaly 
                      ? 'border-red-500/30 bg-red-500/10' 
                      : 'border-white/5 bg-white/[0.02]'
                  }`}
                >
                  <div className="flex items-center gap-4">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${i === 3 && anomaly ? 'bg-red-500/20' : 'bg-white/5'}`}>
                      <span className={`text-[10px] font-mono ${i === 3 && anomaly ? 'text-red-400' : 'text-zinc-500'}`}>{i}</span>
                    </div>
                    <div className={`text-sm tracking-wide ${i === 3 && anomaly ? 'text-red-200' : 'text-zinc-300'}`}>
                      Verification Hash {Math.random().toString(16).slice(2, 10).toUpperCase()}
                    </div>
                  </div>
                  <div className={`text-[10px] font-mono tracking-widest ${i === 3 && anomaly ? 'text-red-400' : 'text-emerald-500/70'}`}>
                    {i === 3 && anomaly ? 'FAILED' : 'VERIFIED'}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
          
          <motion.div 
            className="w-[400px] border rounded-[2rem] p-8 relative overflow-hidden backdrop-blur-xl flex flex-col shadow-2xl"
            animate={{ 
              borderColor: anomaly ? 'rgba(239,68,68,0.3)' : 'rgba(255,255,255,0.05)',
              backgroundColor: anomaly ? 'rgba(239,68,68,0.05)' : 'rgba(255,255,255,0.01)'
            }}
            transition={{ duration: 0.5 }}
          >
            <motion.div 
              className="absolute top-0 left-0 w-full h-1"
              animate={{ background: anomaly ? 'linear-gradient(90deg, rgba(239,68,68,0.5), transparent)' : 'linear-gradient(90deg, rgba(255,255,255,0.1), transparent)' }}
            />
            <div className="text-[10px] font-mono text-red-400 tracking-widest mb-10 uppercase">Hallucination Detect</div>
            
            <div className="space-y-6 flex-1 flex flex-col">
              <AnimatePresence mode="wait">
                {anomaly ? (
                  <motion.div 
                    key="anomaly"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="p-6 border border-red-500/30 bg-red-500/10 rounded-xl"
                  >
                    <div className="text-[10px] font-mono text-red-400 mb-4 flex items-center gap-3">
                      <motion.span className="w-2 h-2 bg-red-500 rounded-full" animate={{ opacity: [1, 0, 1] }} transition={{ duration: 0.5, repeat: Infinity }} />
                      ANOMALY DETECTED
                    </div>
                    <div className="text-xs text-red-200/70 leading-relaxed tracking-wide">
                      Confidence threshold failed at semantic node #748. Tactical audit flagged potential hallucination in reasoning chain.
                    </div>
                  </motion.div>
                ) : (
                  <motion.div 
                    key="stable"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="p-6 border border-white/5 bg-white/[0.02] rounded-xl flex items-center gap-3"
                  >
                    <div className="w-2 h-2 bg-emerald-500/50 rounded-full" />
                    <div className="text-xs text-zinc-500 font-mono uppercase tracking-widest">System Stable</div>
                  </motion.div>
                )}
              </AnimatePresence>
              
              <div className="w-full flex-1 border border-white/5 rounded-xl flex items-center justify-center relative overflow-hidden mt-auto">
                {anomaly && (
                  <motion.div 
                    className="absolute inset-0 bg-[linear-gradient(0deg,transparent_0%,rgba(239,68,68,0.1)_50%,transparent_100%)]"
                    animate={{ top: ['-100%', '100%'] }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  />
                )}
                <div className="text-[10px] font-mono text-zinc-600 text-center px-4 uppercase tracking-widest z-10">
                  {anomaly ? (
                    <span className="text-red-400/50">Adversarial Analysis<br/>Active...</span>
                  ) : (
                    <span>Awaiting<br/>Anomalies</span>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
          
        </div>
      </div>
    </div>
  );
}
