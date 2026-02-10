import React from 'react';
import { History, MessageSquare, Box } from 'lucide-react';

const Sidebar = ({ history, onSelectRun }) => {
  return (
    <div className="w-64 bg-slate-900 text-slate-300 h-screen flex flex-col border-r border-slate-700">
      <div className="p-4 border-b border-slate-700 flex items-center space-x-2">
        <Box className="w-6 h-6 text-emerald-400" />
        <span className="font-bold text-white tracking-wider">SENTINEL</span>
      </div>
      
      <div className="flex-1 overflow-y-auto p-2">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3 ml-2">Recent Runs</h3>
        
        {history.length === 0 && (
            <div className="text-xs text-slate-600 italic ml-2">No logs found.</div>
        )}

        {history.map((run) => (
          <button 
            key={run.id}
            onClick={() => onSelectRun(run)}
            className="w-full text-left p-2 rounded hover:bg-slate-800 transition-colors flex items-center space-x-2 text-sm mb-1 group"
          >
            {run.mode === 'standard' ? (
                <MessageSquare className="w-4 h-4 text-blue-400" />
            ) : (
                <History className="w-4 h-4 text-amber-400" />
            )}
            <span className="truncate group-hover:text-white">{run.summary || run.timestamp}</span>
          </button>
        ))}
      </div>
      
      <div className="p-4 border-t border-slate-800 text-xs text-slate-600">
        v2.0.0 (Sigma-Ready)
      </div>
    </div>
  );
};

export default Sidebar;
