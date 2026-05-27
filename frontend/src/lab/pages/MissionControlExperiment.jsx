import React from 'react';
import { Link } from 'react-router';
import { motion } from 'motion/react';
import CognitionStream from '../ui/CognitionStream';

export default function MissionControlExperiment() {
  return (
    <div className="min-h-screen bg-[#020202] text-white overflow-hidden relative font-sans">
      <div className="absolute top-8 left-8 z-50">
        <Link to="/lab" className="text-xs font-mono tracking-widest text-zinc-500 hover:text-white transition-colors uppercase">← Back to Lab</Link>
      </div>
      
      <div className="absolute inset-0 flex items-center justify-center p-12">
        <div className="w-full h-full max-w-[1600px] border border-white/5 rounded-[2rem] bg-[#050508] overflow-hidden flex flex-col shadow-2xl">
          <header className="border-b border-white/5 p-6 flex justify-between items-center bg-white/[0.01]">
            <div className="flex items-center gap-4 text-xs font-mono tracking-widest text-zinc-400 uppercase">
              <motion.span 
                className="w-2 h-2 rounded-sm bg-blue-500" 
                animate={{ opacity: [1, 0.3, 1] }} 
                transition={{ duration: 2, repeat: Infinity }}
              />
              Mission Control
            </div>
            <div className="text-xs font-mono text-zinc-500 tracking-widest">SYS.OP.ACTIVE</div>
          </header>
          
          <div className="flex-1 grid grid-cols-12 gap-px bg-white/[0.02]">
            <div className="col-span-3 bg-[#050508] p-8 space-y-8 border-r border-white/5">
              <div className="text-[10px] font-mono text-zinc-600 uppercase tracking-widest border-b border-white/5 pb-3">Telemetry</div>
              <div className="space-y-8">
                {[1,2,3,4].map(i => (
                  <div key={i} className="space-y-3">
                    <div className="flex justify-between text-[10px] font-mono text-zinc-500">
                      <span>ROUTING_NODE_0{i}</span>
                      <span className="text-blue-500/70">ACTIVE</span>
                    </div>
                    <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                      <motion.div 
                        className="h-full bg-blue-500/40"
                        animate={{ width: [`${30 + i*10}%`, `${80 - i*5}%`, `${30 + i*10}%`] }}
                        transition={{ duration: 3 + i, repeat: Infinity, repeatType: 'mirror', ease: "linear" }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="col-span-6 bg-[#050508] relative flex items-center justify-center overflow-hidden border-r border-white/5">
              <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(59,130,246,0.03)_0%,transparent_70%)]" />
              
              {/* Radar / Orchestration Map */}
              <div className="w-[500px] h-[500px] border border-blue-500/10 rounded-full flex items-center justify-center relative shadow-[0_0_100px_rgba(59,130,246,0.02)]">
                <motion.div 
                  className="absolute inset-0 border border-blue-500/20 rounded-full"
                  animate={{ scale: [1, 1.1, 1], opacity: [0.3, 0, 0.3] }}
                  transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                />
                <div className="w-[300px] h-[300px] border border-blue-500/10 rounded-full flex items-center justify-center">
                   <motion.div 
                     className="w-full h-full rounded-full border-t border-blue-500/50"
                     animate={{ rotate: 360 }}
                     transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                   />
                </div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-blue-500/30 font-mono text-[10px] tracking-widest uppercase">ORCHESTRATION_MAP</span>
                </div>
                
                {/* Simulated Nodes */}
                <motion.div className="absolute top-[25%] left-[35%] w-2 h-2 bg-blue-400 rounded-full shadow-[0_0_10px_rgba(59,130,246,1)]" animate={{ opacity: [0.2, 1, 0.2] }} transition={{ duration: 2, repeat: Infinity }} />
                <motion.div className="absolute bottom-[35%] right-[25%] w-2 h-2 bg-emerald-400 rounded-full shadow-[0_0_10px_rgba(16,185,129,1)]" animate={{ opacity: [0.2, 1, 0.2] }} transition={{ duration: 3, repeat: Infinity }} />
                <motion.div className="absolute top-[45%] right-[20%] w-1.5 h-1.5 bg-amber-400 rounded-full shadow-[0_0_10px_rgba(245,158,11,1)]" animate={{ opacity: [0.1, 0.8, 0.1] }} transition={{ duration: 1.5, repeat: Infinity }} />
              </div>
            </div>
            
            <div className="col-span-3 bg-[#050508] p-8 space-y-8 flex flex-col">
              <div className="text-[10px] font-mono text-zinc-600 uppercase tracking-widest border-b border-white/5 pb-3">Cognition Streams</div>
              <div className="flex-1 space-y-12 mt-4">
                <CognitionStream label="SEMANTIC_PIPELINE" active={true} />
                <CognitionStream label="EVIDENCE_ROUTING" active={true} />
                <CognitionStream label="SYNTHESIS_ENGINE" active={true} />
                <CognitionStream label="GOVERNANCE_AUDIT" active={true} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
