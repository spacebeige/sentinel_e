import React, { useState, useEffect } from 'react';
import { Link } from 'react-router';
import SemanticTopology from '../backgrounds/SemanticTopology';
import AgentNode from '../systems/AgentNode';
import CognitionStream from '../ui/CognitionStream';

const EXPS = [
  { path: '/lab/council', title: 'Council Chamber', desc: 'Floating AI entities' },
  { path: '/lab/mission-control', title: 'Mission Control', desc: 'Operational maps' },
  { path: '/lab/debate', title: 'Debate Arena', desc: 'AI tribunal conflicts' },
  { path: '/lab/cognition', title: 'Cognition', desc: 'Atmospheric depth' },
  { path: '/lab/semantic', title: 'Semantic Flow', desc: 'Contradiction fields' },
  { path: '/lab/governance', title: 'Governance', desc: 'Verification overlays' },
];

export default function LabLandingPage() {
  const [agents, setAgents] = useState({
    gpt: 'idle',
    claude: 'idle',
    gemini: 'idle',
    mistral: 'idle'
  });
  const [conflict, setConflict] = useState(false);

  useEffect(() => {
    const states = ['idle', 'thinking', 'debating', 'synthesis'];
    
    const interval = setInterval(() => {
      const newAgents = {
        gpt: states[Math.floor(Math.random() * states.length)],
        claude: states[Math.floor(Math.random() * states.length)],
        gemini: states[Math.floor(Math.random() * states.length)],
        mistral: states[Math.floor(Math.random() * states.length)]
      };
      
      setAgents(newAgents);
      
      const debatingCount = Object.values(newAgents).filter(s => s === 'debating').length;
      setConflict(debatingCount >= 2);
    }, 3500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-black text-zinc-300 font-sans selection:bg-zinc-800 overflow-hidden relative">
      <SemanticTopology active={true} conflict={conflict} />
      
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(20,20,25,0.7)_0%,rgba(0,0,0,0.95)_80%)] pointer-events-none z-0" />
      
      <div className="relative z-10 p-12 max-w-[1600px] mx-auto h-screen flex flex-col justify-between">
        
        <header className="space-y-4 max-w-xl">
          <div className="flex items-center gap-4 text-xs font-mono tracking-widest text-zinc-500 uppercase">
            <span className={`w-2 h-2 rounded-full animate-pulse ${conflict ? 'bg-red-500/50' : 'bg-blue-500/50'}`} />
            Sentinel-E • UI Laboratory
          </div>
          <h1 className="text-5xl font-light tracking-tight text-white mix-blend-plus-lighter">
            Visible Machine Cognition
          </h1>
          <p className="text-zinc-400 leading-relaxed text-sm">
            Live cognitive sandbox. Orchestration visuals, semantic topology systems, and cinematic AI deliberation.
          </p>
        </header>

        {/* The Council Hero Visualization */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-5xl h-[600px] flex items-center justify-center pointer-events-none">
          <div className="relative w-full h-full flex items-center justify-center">
            
            {/* Central Synthesis Core */}
            <div className="absolute w-96 h-96 border border-white/[0.03] rounded-full flex items-center justify-center">
              <div className="absolute w-64 h-64 border border-white/[0.05] rounded-full" />
               <div className={`w-48 h-48 rounded-full blur-3xl transition-colors duration-1000 ${conflict ? 'bg-red-500/10' : 'bg-blue-500/10'}`} />
            </div>

            {/* Orbiting Agents */}
            <div className="absolute top-[15%] left-[20%]">
              <AgentNode model="gpt" label="GPT-4" state={agents.gpt} size="lg" />
            </div>
            <div className="absolute bottom-[20%] left-[15%]">
              <AgentNode model="claude" label="Claude-3" state={agents.claude} size="lg" />
            </div>
            <div className="absolute top-[20%] right-[15%]">
              <AgentNode model="gemini" label="Gemini-Pro" state={agents.gemini} size="lg" />
            </div>
            <div className="absolute bottom-[25%] right-[25%]">
              <AgentNode model="mistral" label="Mistral-Large" state={agents.mistral} size="md" />
            </div>
            
            {/* Routing Streams */}
            <div className="absolute top-[35%] left-[35%] w-32 -rotate-45">
               <CognitionStream active={agents.gpt === 'thinking' || agents.gpt === 'synthesis'} label="LOGIC_ROUTING" />
            </div>
            <div className="absolute bottom-[40%] right-[35%] w-32 -rotate-45">
               <CognitionStream active={agents.gemini === 'thinking' || agents.gemini === 'synthesis'} label="TOPOLOGY_MAP" />
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mt-auto z-20">
          {EXPS.map(exp => (
            <Link key={exp.path} to={exp.path} className="group block relative p-4 rounded-xl border border-white/5 bg-white/[0.02] hover:bg-white/[0.06] hover:border-white/10 transition-all duration-300 overflow-hidden backdrop-blur-xl">
              <h2 className="text-sm font-medium text-white mb-1 tracking-wide group-hover:translate-x-1 transition-transform duration-300">{exp.title}</h2>
              <p className="text-[10px] text-zinc-500 group-hover:text-zinc-400 transition-colors duration-300">{exp.desc}</p>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
