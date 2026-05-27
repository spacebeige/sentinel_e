import React from 'react';
import { motion } from 'motion/react';
import { agentMotion, nodeHover } from '../motion/CognitionMotion';

export default function AgentNode({ 
  model = 'gpt', 
  state = 'idle', // idle, thinking, debating, synthesis, error
  label, 
  size = 'md', // sm, md, lg
  className = '' 
}) {
  const getPhysics = () => agentMotion[model] || agentMotion.gpt;
  
  const stateColors = {
    idle: 'border-white/20 bg-white/[0.02]',
    thinking: 'border-blue-500/40 bg-blue-500/10 shadow-[0_0_20px_rgba(59,130,246,0.2)]',
    debating: 'border-red-500/40 bg-red-500/10 shadow-[0_0_20px_rgba(239,68,68,0.2)]',
    synthesis: 'border-emerald-500/40 bg-emerald-500/10 shadow-[0_0_20px_rgba(16,185,129,0.2)]',
    error: 'border-amber-500/40 bg-amber-500/10 shadow-[0_0_20px_rgba(245,158,11,0.2)]',
  };

  const sizes = {
    sm: 'w-12 h-12 text-[10px]',
    md: 'w-20 h-20 text-xs',
    lg: 'w-32 h-32 text-sm',
    xl: 'w-48 h-48 text-base',
  };

  const animateProps = {
    scale: state === 'thinking' ? [1, 1.05, 1] : state === 'debating' ? [1, 0.95, 1.02, 1] : 1,
    rotate: state === 'debating' ? [0, -2, 2, 0] : 0,
  };

  const transitionProps = {
    ...getPhysics(),
    ...(state !== 'idle' && { repeat: Infinity, repeatType: "mirror" })
  };

  return (
    <motion.div 
      className={`relative rounded-full border backdrop-blur-xl flex items-center justify-center ${stateColors[state]} ${sizes[size]} ${className}`}
      animate={animateProps}
      transition={transitionProps}
      whileHover={nodeHover}
    >
      <div className="absolute inset-0 rounded-full bg-[radial-gradient(circle_at_50%_0%,rgba(255,255,255,0.1)_0%,transparent_70%)] pointer-events-none" />
      
      {/* Inner Core */}
      <motion.div 
        className={`w-1/4 h-1/4 rounded-full ${state === 'synthesis' ? 'bg-emerald-400' : state === 'debating' ? 'bg-red-400' : 'bg-white/40'}`}
        animate={{ opacity: state !== 'idle' ? [0.4, 1, 0.4] : 0.4 }}
        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
      />

      {label && (
        <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 font-mono text-zinc-400 text-xs tracking-widest uppercase whitespace-nowrap">
          {label}
        </div>
      )}
    </motion.div>
  );
}
