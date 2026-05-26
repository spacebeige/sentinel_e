import React from 'react';
import { Link } from 'react-router';

export default function CouncilExperiment() {
  return (
    <div className="min-h-screen bg-black text-white overflow-hidden relative">
      <div className="absolute top-8 left-8 z-50">
        <Link to="/lab" className="text-xs font-mono tracking-widest text-zinc-500 hover:text-white transition-colors uppercase">← Back to Lab</Link>
      </div>
      
      {/* Visual Sandbox Area */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative w-full max-w-4xl h-[600px] border border-white/10 bg-zinc-950/50 backdrop-blur-3xl rounded-3xl p-12 flex flex-col items-center justify-center overflow-hidden shadow-[0_0_100px_rgba(255,255,255,0.02)]">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(255,255,255,0.03)_0%,transparent_60%)]" />
          
          <h1 className="text-2xl font-light tracking-widest text-zinc-300 mb-8 z-10">COUNCIL CHAMBER</h1>
          
          <div className="flex gap-12 items-center justify-center z-10">
            {[1, 2, 3].map(i => (
              <div key={i} className="relative group">
                <div className={`w-24 h-24 rounded-full border border-white/20 bg-black/50 flex items-center justify-center backdrop-blur-xl ${i === 2 ? 'w-32 h-32 border-white/40 shadow-[0_0_30px_rgba(255,255,255,0.1)]' : ''}`}>
                  <div className="w-2 h-2 rounded-full bg-white/50 animate-ping" />
                </div>
                <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-xs font-mono text-zinc-600 uppercase tracking-widest">Node {i}</div>
              </div>
            ))}
          </div>
          
          <div className="absolute bottom-12 text-sm text-zinc-500 font-mono tracking-widest flex items-center gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
            Awaiting Consensus Pulse...
          </div>
        </div>
      </div>
    </div>
  );
}
