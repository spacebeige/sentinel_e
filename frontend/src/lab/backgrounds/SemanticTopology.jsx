import React from 'react';
import { motion } from 'motion/react';

export default function SemanticTopology({ active = false, conflict = false }) {
  // A living SVG background network
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden mix-blend-screen opacity-30 z-0">
      <motion.svg 
        className="w-full h-full"
        animate={{
          scale: active ? 1.05 : 1,
          opacity: conflict ? [0.3, 0.6, 0.3] : active ? 0.6 : 0.3,
          x: conflict ? [0, -5, 5, 0] : 0
        }}
        transition={{ duration: conflict ? 0.2 : 8, repeat: Infinity, repeatType: "mirror", ease: "linear" }}
        viewBox="0 0 1000 1000" preserveAspectRatio="xMidYMid slice"
      >
        <defs>
          <linearGradient id="edgeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor={conflict ? "#ef4444" : "rgba(255,255,255,0.1)"} />
            <stop offset="100%" stopColor={conflict ? "#3b82f6" : "rgba(255,255,255,0)"} />
          </linearGradient>
        </defs>

        <g stroke="url(#edgeGrad)" strokeWidth="1.5" fill="none">
          {/* We create a massive continuous bezier grid that morphs slightly */}
          <motion.path 
            d="M -100 200 C 300 100, 700 400, 1100 300" 
            animate={{ d: active ? "M -100 250 C 300 50, 700 450, 1100 250" : "M -100 200 C 300 100, 700 400, 1100 300" }}
            transition={{ duration: 10, repeat: Infinity, repeatType: "mirror" }}
          />
          <motion.path 
            d="M -100 500 C 400 600, 600 300, 1100 500" 
            animate={{ d: active ? "M -100 450 C 400 650, 600 250, 1100 550" : "M -100 500 C 400 600, 600 300, 1100 500" }}
            transition={{ duration: 12, repeat: Infinity, repeatType: "mirror" }}
          />
          <motion.path 
            d="M -100 800 C 200 700, 800 900, 1100 800" 
            animate={{ d: active ? "M -100 850 C 200 650, 800 950, 1100 750" : "M -100 800 C 200 700, 800 900, 1100 800" }}
            transition={{ duration: 15, repeat: Infinity, repeatType: "mirror" }}
          />
          <motion.path 
            d="M 200 -100 C 100 300, 400 700, 300 1100" 
            animate={{ d: active ? "M 250 -100 C 50 300, 450 700, 250 1100" : "M 200 -100 C 100 300, 400 700, 300 1100" }}
            transition={{ duration: 11, repeat: Infinity, repeatType: "mirror" }}
          />
          <motion.path 
            d="M 800 -100 C 900 400, 600 600, 700 1100" 
            animate={{ d: active ? "M 750 -100 C 950 400, 550 600, 750 1100" : "M 800 -100 C 900 400, 600 600, 700 1100" }}
            transition={{ duration: 14, repeat: Infinity, repeatType: "mirror" }}
          />
        </g>
      </motion.svg>
    </div>
  );
}
