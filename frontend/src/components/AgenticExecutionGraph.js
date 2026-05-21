import React from 'react';
import { motion } from 'framer-motion';
import { Globe, CheckCircle2, Clock, AlertCircle, ChevronRight, Activity } from 'lucide-react';

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

export default function AgenticExecutionGraph({ steps = [], className = '' }) {
  if (!steps || steps.length === 0) return null;

  return (
    <div className={`mt-3 p-3 rounded-2xl bg-white/50 dark:bg-[#1c1c1e]/50 border border-black/5 dark:border-white/10 ${className}`}>
      <div className="flex items-center gap-2 mb-3 px-1">
        <Activity className="w-4 h-4 text-[#8b5cf6]" />
        <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600, color: '#8b5cf6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Agentic Execution Trace
        </span>
      </div>
      
      <div className="space-y-2 relative before:absolute before:inset-y-0 before:left-[15px] before:w-0.5 before:bg-black/5 dark:before:bg-white/10">
        {steps.map((step, index) => {
          const isLast = index === steps.length - 1;
          const statusColors = {
            running: 'bg-blue-500 text-white border-blue-600',
            success: 'bg-emerald-500 text-white border-emerald-600',
            error: 'bg-red-500 text-white border-red-600',
            pending: 'bg-gray-200 dark:bg-gray-800 text-gray-500 border-gray-300 dark:border-gray-700'
          };
          const colorClass = statusColors[step.status] || statusColors.pending;

          return (
            <motion.div 
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              key={index} 
              className="relative flex items-start gap-3 pl-1"
            >
              <div className="relative z-10 w-7 h-7 mt-0.5 rounded-full flex items-center justify-center border-2 bg-white dark:bg-[#1c1c1e] shadow-sm">
                <div className={`w-4 h-4 rounded-full flex items-center justify-center ${colorClass}`}>
                  {step.status === 'running' && <Clock className="w-2.5 h-2.5 animate-pulse" />}
                  {step.status === 'success' && <CheckCircle2 className="w-2.5 h-2.5" />}
                  {step.status === 'error' && <AlertCircle className="w-2.5 h-2.5" />}
                  {step.status === 'pending' && <div className="w-1.5 h-1.5 rounded-full bg-current" />}
                </div>
              </div>
              
              <div className="flex-1 bg-white dark:bg-[#202024] border border-black/5 dark:border-white/5 rounded-xl p-2 shadow-sm">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 text-[#1d1d1f] dark:text-[#f1f5f9]" style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 500 }}>
                    {step.type === 'browser' ? <Globe className="w-3.5 h-3.5 text-blue-500" /> : <Activity className="w-3.5 h-3.5 text-purple-500" />}
                    {step.action || 'Executing Action'}
                  </div>
                  {step.duration_ms && (
                    <div className="text-[#6e6e73] dark:text-[#94a3b8]" style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 400 }}>
                      {step.duration_ms}ms
                    </div>
                  )}
                </div>
                {step.details && (
                  <div className="mt-1 text-[#6e6e73] dark:text-[#94a3b8] truncate" style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 400 }}>
                    {step.details}
                  </div>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
