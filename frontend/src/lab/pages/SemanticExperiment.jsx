import React, { useState, useEffect } from 'react';
import { Link } from 'react-router';
import SemanticTopology from '../backgrounds/SemanticTopology';
import { motion } from 'motion/react';

export default function SemanticExperiment() {
  const [conflict, setConflict] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setConflict(c => !c);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#030303] text-white overflow-hidden relative">
      <div className="absolute top-8 left-8 z-50">
        <Link to="/lab" className="text-xs font-mono tracking-widest text-zinc-500 hover:text-white transition-colors uppercase">← Back to Lab</Link>
      </div>
      
      <SemanticTopology active={true} conflict={conflict} />

      <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
        <div className="relative w-full h-full max-w-6xl">
          <div className="absolute top-12 left-12 text-xs font-mono text-zinc-400 tracking-widest uppercase bg-black/50 p-3 rounded-lg backdrop-blur-md border border-white/5">
            Semantic Topology Visualization
          </div>
          
          <div className="absolute right-12 bottom-12 border border-white/10 bg-black/40 backdrop-blur-xl p-8 rounded-2xl w-96">
            <div className="text-xs font-mono text-zinc-400 mb-6 tracking-widest flex items-center justify-between">
              CONTRADICTION FIELD
              <motion.span 
                animate={{ color: conflict ? '#ef4444' : '#10b981' }}
                className="text-[10px]"
              >
                {conflict ? 'ACTIVE_DESTABILIZATION' : 'STABLE_TOPOLOGY'}
              </motion.span>
            </div>
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="text-[10px] font-mono text-zinc-500 flex justify-between">
                  <span>THESIS_NODE_74</span>
                  <span>{conflict ? '82%' : '14%'}</span>
                </div>
                <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                  <motion.div 
                    className="h-full bg-red-500/50" 
                    animate={{ width: conflict ? '82%' : '14%' }}
                    transition={{ duration: 2, ease: "easeInOut" }}
                  />
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-[10px] font-mono text-zinc-500 flex justify-between">
                  <span>ANTITHESIS_NODE_12</span>
                  <span>{conflict ? '45%' : '91%'}</span>
                </div>
                <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                  <motion.div 
                    className="h-full bg-blue-500/50" 
                    animate={{ width: conflict ? '45%' : '91%' }}
                    transition={{ duration: 2, ease: "easeInOut" }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
