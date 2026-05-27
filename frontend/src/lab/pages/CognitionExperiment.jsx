import React from 'react';
import { Link } from 'react-router';
import { motion } from 'motion/react';
import { easeCinematic } from '../motion/CognitionMotion';

export default function CognitionExperiment() {
  return (
    <div className="min-h-screen bg-[#020202] text-white overflow-hidden relative">
      <div className="absolute top-8 left-8 z-50">
        <Link to="/lab" className="text-xs font-mono tracking-widest text-zinc-500 hover:text-white transition-colors uppercase">← Back to Lab</Link>
      </div>
      
      {/* Atmospheric Depth */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(59,130,246,0.05)_0%,transparent_60%)]" />
      <motion.div 
        className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_60%_at_50%_50%,black_10%,transparent_100%)] pointer-events-none opacity-50"
        animate={{ backgroundPosition: ['0px 0px', '64px 64px'] }}
        transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
      />
      
      <div className="absolute inset-0 flex flex-col items-center justify-center perspective-[1200px]">
        <div className="text-[10px] font-mono text-zinc-500 tracking-widest mb-24 uppercase">Layered Cognition State</div>
        
        <motion.div 
          className="relative w-[700px] h-[450px] transform-style-3d rotate-x-[15deg]"
          animate={{ rotateY: [-5, 5, -5] }}
          transition={{ duration: 15, repeat: Infinity, ease: easeCinematic }}
        >
          {/* Layer 1 - Surface */}
          <motion.div 
            className="absolute inset-0 border border-blue-500/20 bg-blue-500/[0.02] rounded-3xl backdrop-blur-xl flex items-center justify-center transform translate-z-[80px] shadow-[0_20px_50px_rgba(0,0,0,0.5)]"
            animate={{ z: [80, 100, 80] }}
            transition={{ duration: 4, repeat: Infinity, ease: easeCinematic }}
          >
            <div className="text-blue-400/50 font-mono text-xs tracking-widest">SURFACE_LOGIC</div>
            <motion.div className="absolute top-8 left-8 w-16 h-1 bg-blue-500/50 rounded-full" animate={{ opacity: [0.2, 1, 0.2] }} transition={{ duration: 2, repeat: Infinity }} />
          </motion.div>
          
          {/* Layer 2 - Deep Reasoning */}
          <motion.div 
            className="absolute inset-4 border border-purple-500/20 bg-purple-500/[0.01] rounded-3xl backdrop-blur-md flex items-center justify-center transform translate-z-0 shadow-[0_20px_50px_rgba(0,0,0,0.5)]"
            animate={{ z: [0, 10, 0] }}
            transition={{ duration: 5, repeat: Infinity, ease: easeCinematic }}
          >
            <div className="text-purple-400/30 font-mono text-xs tracking-widest">DEEP_REASONING</div>
            <div className="absolute bottom-12 w-[80%] h-px bg-gradient-to-r from-transparent via-purple-500/30 to-transparent">
               <motion.div className="h-full w-1/4 bg-purple-400" animate={{ left: ['0%', '100%'] }} transition={{ duration: 3, repeat: Infinity, ease: "linear" }} style={{ position: 'absolute' }} />
            </div>
          </motion.div>
          
          {/* Layer 3 - Subconscious */}
          <motion.div 
            className="absolute inset-12 border border-emerald-500/10 bg-emerald-500/[0.01] rounded-3xl flex items-center justify-center transform -translate-z-[80px]"
            animate={{ z: [-80, -60, -80] }}
            transition={{ duration: 6, repeat: Infinity, ease: easeCinematic }}
          >
            <div className="text-emerald-400/20 font-mono text-xs tracking-widest">SUBCONSCIOUS_NET</div>
            <motion.div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(16,185,129,0.05)_0%,transparent_50%)]" animate={{ opacity: [0, 1, 0] }} transition={{ duration: 4, repeat: Infinity }} />
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}
