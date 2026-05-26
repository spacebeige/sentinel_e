/**
 * Sentinel-E Motion Language
 * Centralized easing, physics, and variants for the cognition frontend.
 */

// Core Easing Curves
export const easeCinematic = [0.25, 0.1, 0.25, 1];
export const easeStrategic = [0.83, 0, 0.17, 1];
export const easeOrganic = [0.4, 0, 0.2, 1];
export const easeSharp = [0.16, 1, 0.3, 1];

// Agent Personalities
export const agentMotion = {
  gpt: {
    type: "spring",
    stiffness: 200,
    damping: 20,
    mass: 1,
  },
  claude: {
    type: "tween",
    ease: easeStrategic,
    duration: 1.2,
  },
  gemini: {
    type: "spring",
    stiffness: 50,
    damping: 30,
    mass: 1.5,
  },
  mistral: {
    type: "tween",
    ease: easeSharp,
    duration: 0.4,
  }
};

// Global Variants
export const topologyVariants = {
  idle: {
    opacity: 0.3,
    scale: 1,
    transition: { duration: 4, ease: "linear", repeat: Infinity, repeatType: "mirror" }
  },
  active: {
    opacity: 0.6,
    scale: 1.05,
    transition: { duration: 2, ease: easeCinematic }
  },
  conflict: {
    opacity: 0.8,
    scale: 0.98,
    x: [0, -2, 2, -1, 1, 0],
    transition: { duration: 0.3, ease: easeSharp }
  }
};

export const pulseVariants = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: { 
    opacity: [0, 1, 0], 
    scale: [0.8, 1.2, 1.5],
    transition: { duration: 1.5, ease: easeOrganic, repeat: Infinity } 
  }
};

export const nodeHover = {
  scale: 1.05,
  boxShadow: "0px 0px 30px rgba(255, 255, 255, 0.1)",
  transition: { duration: 0.3, ease: easeCinematic }
};
