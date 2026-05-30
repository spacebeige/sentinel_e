import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, ChevronDown } from 'lucide-react';

export interface IOSContextMenuOption {
  label: string;
  value: string;
}

export interface IOSContextMenuProps {
  options: IOSContextMenuOption[];
  value: string;
  onChange: (val: string) => void;
  label?: string;
  className?: string;
}

export const IOSContextMenu: React.FC<IOSContextMenuProps> = ({
  options,
  value,
  onChange,
  label,
  className = ''
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const selectedOption = options.find((o) => o.value === value) || options[0];

  // Close when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  return (
    <div className={`relative ${className}`} ref={containerRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 text-[17px] text-black/60 dark:text-white/60 focus:outline-none"
      >
        {label ? <span className="mr-2">{label}</span> : null}
        <span className="capitalize">{selectedOption.label}</span>
        <div className="bg-black/5 dark:bg-white/10 rounded-full p-1 flex items-center justify-center">
          <ChevronDown className="w-3.5 h-3.5" />
        </div>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -10 }}
            transition={{ type: 'spring', stiffness: 500, damping: 30 }}
            className="absolute right-0 top-full mt-2 w-56 z-50 rounded-[14px] overflow-hidden ios-glass-panel shadow-xl border border-black/5 dark:border-white/10"
          >
            <div className="py-1">
              {options.map((option, idx) => {
                const isSelected = option.value === value;
                return (
                  <button
                    key={option.value}
                    onClick={() => {
                      onChange(option.value);
                      setIsOpen(false);
                    }}
                    className={`
                      w-full flex items-center justify-between px-4 py-2.5 text-[15px]
                      ${idx !== options.length - 1 ? 'border-b border-black/5 dark:border-white/10' : ''}
                      hover:bg-black/5 dark:hover:bg-white/10 transition-colors
                      ${isSelected ? 'font-semibold' : 'font-normal'}
                    `}
                  >
                    <span className="capitalize">{option.label}</span>
                    {isSelected && <Check className="w-4 h-4 text-[#007AFF] dark:text-[#0A84FF]" />}
                  </button>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
