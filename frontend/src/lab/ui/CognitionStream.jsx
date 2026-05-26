import React from 'react';
import { motion } from 'motion/react';

export default function CognitionStream({ active = true, label = 'COGNITION_STREAM' }) {
  return (
    <div className="flex flex-col gap-2 w-full">
      <div className="flex items-center gap-3">
        <motion.div 
          className="w-1.5 h-1.5 rounded-full bg-blue-500"
          animate={{ opacity: active ? [0.2, 1, 0.2] : 0.2 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
        />
        <div className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase">
          {label}
        </div>
      </div>
      <div className="h-0.5 w-full bg-white/5 rounded-full overflow-hidden relative">
        {active && (
          <motion.div 
            className="absolute top-0 bottom-0 w-1/4 bg-gradient-to-r from-transparent via-blue-500/50 to-transparent"
            initial={{ left: '-25%' }}
            animate={{ left: '125%' }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          />
        )}
      </div>
    </div>
  );
}
