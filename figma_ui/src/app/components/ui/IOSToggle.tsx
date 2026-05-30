import React from 'react';
import { motion } from 'framer-motion';

export interface IOSToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

export const IOSToggle: React.FC<IOSToggleProps> = ({ checked, onChange, disabled = false }) => {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => !disabled && onChange(!checked)}
      className={`
        relative inline-flex h-[31px] w-[51px] shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent 
        transition-colors duration-300 ease-in-out focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2
        ${checked ? 'bg-[#34C759]' : 'bg-[#E9E9EA] dark:bg-[#39393D]'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      <span className="sr-only">Toggle</span>
      <motion.div
        layout
        initial={false}
        animate={{
          x: checked ? 20 : 0,
          scale: 1,
        }}
        whileTap={!disabled ? { scale: 0.95, x: checked ? 18 : 2 } : {}}
        transition={{
          type: 'spring',
          stiffness: 500,
          damping: 30,
          mass: 1
        }}
        className={`
          pointer-events-none inline-block h-[27px] w-[27px] rounded-full bg-white shadow-sm ring-0
        `}
      />
    </button>
  );
};
